import scannerpy
import scannertools as st
import numpy as np
from scipy.spatial import distance
from esper.prelude import *
from query.models import Video
from rekall.interval_list import IntervalList
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.logical_predicates import *
from rekall.temporal_predicates import *
from rekall.payload_predicates import *
from rekall.list_predicates import *
from rekall.bbox_predicates import *
from rekall.spatial_predicates import *
from rekall.merge_ops import *

video_id = 123
videos = Video.objects.filter(id=123).all()

# Load histograms from Scanner
db = scannerpy.Dtabase()
hists = st.histograms.compute_histograms(
    db,
    videos=[video.for_scannertools() for video in videos]
)

# Do simple outlier detection on histogram differences to get microshot
#   boundaries
WINDOW_SIZE=500
hists_list = [hist for hist in hists[0].load()]
diffs = np.array([
    np.mean([distance.chebyshev(hists_list[i - 1][j], hists_list[i][j]) for j in range(3)])
    for i in range(1, len(hists_list))
])
diffs = np.insert(diffs, 0, 0)
n = len(diffs)
boundaries = []
for i in range(1, n):
    window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
    if diffs[i] - np.mean(window) > 2.5 * np.std(window):
        boundaries.append(i)

# Compute face detection twice a second and before/after every microshot boundary
frames = list(range(0, video.num_frames, int(round(video.fps) / 2)))
frames_set = set(frames)
frames_set = frames_set.union(set(boundaries))
frames_set = frames_set.union(set([boundary - 1 for boundary in boundaries if boundary > 0]))
frames = sorted(list(frames_set))

# Compute face detections in Scanner
faces = st.face_detection.detect_faces(
    db,
    videos=[video.for_scannertools() for video in videos]
    frames = [frames]
)

# One interval for every microshot transition
transitions = IntervalList([(boundary - 1, boundary, 0) for boundary in boundaries])

# One interval for every frame whose payload is list of faces in that frame
faces_at_boundaries = IntervalList([
    (frame, frame, facelist)
    for frame, facelist in zip(frames, faces[0].load()))
]).filter_against(
    transitions,
    predicate=overlaps()
).filter(payload_satisfies(length_at_least(1)))

# Get all boundaries where there are faces before and after the boundary
boundaries_that_have_faces = transitions.filter_against(
    faces_at_boundaries, predicate=starts_inv() # Faces at the start of this transition
).filter_against(
    transitions.filter_against(
        faces_at_boundaries, predicate=finishes_inv() # Faces at the end of this transition
    ),
    predicate=equal()
)

# Annotate boundary payloads with the faces at the start/end of each transition
boundaries_starting_faces = boundaries_that_have_faces.merge(
    faces_at_boundaries, predicate = starts_inv(),
    payload_merge_op = payload_second
)
boundaries_ending_faces = boundaries_that_have_faces.merge(
    faces_at_boundaries, predicate = finishes_inv(),
    payload_merge_op = payload_second
)

# Finally, annotate boundary with a payload of the faces that start/end the
#   transition
boundaries_transition_faces = boundaries_starting_faces.merge(
    boundaries_ending_faces, predicate=equal(),
    payload_merge_op = lambda starts_payload, finishes_payload: {
        'starts': starts_payload, 'finishes': finishes_payload
    }
)

# Get all the boundaries where the faces are very similar before and after the
#   boundary
def similar_face_lists(faces):
    graph = {
        'nodes': [{
            'name': 'face{}'.format(idx),
            'predicates': [
                position(face.x1, face.y1, face.x2, face.y2, epsilon=.05),
                lambda face: face['score'] > 0.9
            ]
        } for idx, face in enumerate(faces['starts']) if face.score > 0.9],
        'edges': []
    }
    return scene_graph(graph, exact=True)([{ 
        'x1': face.x1, 
        'y1': face.y1, 
        'x2': face.x2, 
        'y2': face.y2, 
        'score': face.score } for face in faces['finishes']])

bad_boundaries = boundaries_transition_faces.filter(
    payload_satisfies(similar_face_lists))

# Finally, generate shots from the boundaries
def boundaries_to_shots_fold(acc, frame):
    if acc == []:
        return [frame.copy()]
    top = acc[-1]
    top.end = frame.start - 1
    if top.length() > 0:
        acc.append(frame.copy())
    else:
        top.end = frame.start
    return acc

def boundaries_to_shots(boundaries):
    boundaries = [0] + boundaries
    boundary_list = IntervalList([(boundary, boundary, 0) for boundary in boundaries])
    shots = boundary_list.fold_list(boundaries_to_shots_fold, [])
    
    return shots

# Generate microshots
microshots = boundaries_to_shots(boundaries)

# Filter out short microshots
short_microshots = microshots.filter_length(max_length=10)
shots = microshots.set_union(
    short_microshots.map(lambda i: (i.start, i.end + 1, i.payload)).coalesce()
).coalesce()

# Remove the bad boundaries we identified earlier
bad_shots = shots.filter_against(
    bad_boundaries.map(lambda i: (i.start+1, i.end, i.payload)),
    predicate=starts_inv()
)
shot_boundaries = shots.map(lambda i: (i.start, i.start, i.payload))
shot_boundaries_without_bad_shots = shot_boundaries.minus(bad_shots)
shots = shot_boundaries_without_bad_shots.fold_list(boundaries_to_shots_fold, [])
