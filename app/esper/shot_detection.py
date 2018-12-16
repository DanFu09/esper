import scannertools as st
import scannerpy
from scipy.spatial import distance
import numpy as np
import math
from esper.prelude import Notifier, par_for
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from django.db.models import Q, F
from django.db import transaction
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from rekall.interval_list import IntervalList
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.logical_predicates import *
from rekall.temporal_predicates import *
from rekall.payload_predicates import *
from rekall.list_predicates import *
from rekall.bbox_predicates import *
from rekall.spatial_predicates import *
from rekall.merge_ops import *
from tqdm import tqdm

# Parameters for histogram outlier detection
WINDOW_SIZE = 500
OUTLIER_THRESHOLD = 2.5

# Parameters for face detection
FACE_FPS = 2

# Parameters for shot generation
MINIMUM_FACE_PROBABILITY = 0.9
POSITION_EPSILON = 0.05
MINIMUM_SHOT_DURATION = 0.42 # 10 frames at 24 FPS

def microshot_boundaries_from_histograms(histogram):
    """Get microshot boundaries from histograms using outlier detection."""
    histogram=list(histogram)

    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([
        np.mean([distance.chebyshev(histogram[i - 1][j], histogram[i][j]) for j in range(3)])
        for i in range(1, len(histogram))
    ])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Do simple outlier detection to find boundaries between shots
    boundaries = []
    for i in range(1, n):
        window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if diffs[i] - np.mean(window) > OUTLIER_THRESHOLD * np.std(window):
            boundaries.append(i)
    
    return boundaries

def frames_to_detect_faces(microshot_boundaries, video):
    # Detect faces FACE_FPS times a second
    sampling_rate = int(round(video.fps) / FACE_FPS)
    frames = set(range(0, video.num_frames, sampling_rate))

    # Detect faces at every microshot boundary
    frames = frames.union(set(microshot_boundaries))

    # Also detect faces the frame before every microshot boundary
    frames = frames.union(set([boundary - 1
        for boundary in microshot_boundaries
        if boundary > 0]))

    return sorted(list(frames))

def new_frame_objs(frames, video):
    frames_existing = [
        frame.number
        for frame in Frame.objects.filter(video_id=video.id).all()
    ]
    
    new_frames = [frame for frame in frames if frame not in frames_existing]
    return [
        Frame(number=frame, video_id=video.id)
        for frame in new_frames
    ]

def update_database_with_faces(faces, frames, video, labeler, tag):
    # First create new frames
    Frame.objects.bulk_create(new_frame_objs(frames, video))

    # Get the frames in this video that have already been labeled
    frames_labeled_already = set([
        face.frame_id
        for face in Face.objects.filter(frame__video_id=video.id)
    ])

    # Next get all the Frame objects out of the database
    frame_objs = Frame.objects.filter(number__in=frames, video_id=video.id).order_by('number').all()

    # Create new Face objects and new Frame tags
    new_faces = []
    new_frame_tags = []
    for bbox_list, frame, frame_obj in zip(faces, frames, frame_objs):
        if frame_obj.id in frames_labeled_already:
            continue
        for bbox in bbox_list:
            new_faces.append(Face(
                frame=frame_obj,
                bbox_x1=bbox.x1,
                bbox_x2=bbox.x2,
                bbox_y1=bbox.y1,
                bbox_y2=bbox.y2,
                probability=bbox.score,
                labeler=labeler
            ))
        #new_frame_tags.append(Frame.tags.through(
        #    frame_id=frame_obj.pk,
        #    tag_id=tag.pk
        #))
    
    with transaction.atomic():
        Frame.tags.through.objects.bulk_create(new_frame_tags)
        Face.objects.bulk_create(new_faces)
        VideoTag.objects.get_or_create(video=video, tag=tag)

def compute_shots(microshot_boundaries, faces_scanner, frames, video):
    faces_per_frame = IntervalList([
        (frame, frame, facelist)
        for frame, facelist in zip(frames, faces_scanner)
    ])

    transitions = IntervalList([(boundary - 1, boundary, 0)
        for boundary in microshot_boundaries])

    faces_at_boundaries = faces_per_frame.filter_against(
        transitions,
        predicate=overlaps()
    ).filter(payload_satisfies(length_at_least(1)))

    # Get all transitions where there are faces before and after the transition
    # This IntervalList's payload is stil 0
    transitions_with_faces = transitions.filter_against(
        faces_at_boundaries, predicate=starts_inv()
    ).filter_against(
        transitions.filter_against(faces_at_boundaries, predicate=finishes_inv()),
        predicate=equal()
    )

    # Annotate transitions_with_faces with the list of faces before and after
    #   every transition
    transitions_with_faces_at_start_of_transition = transitions_with_faces.merge(
        faces_at_boundaries, predicate = starts_inv(),
        payload_merge_op = payload_second
    )
    transitions_with_faces_at_end_of_transition = transitions_with_faces.merge(
        faces_at_boundaries, predicate=finishes_inv(),
        payload_merge_op = payload_second
    )
    transitions_with_faces = transitions_with_faces_at_start_of_transition.merge(
        transitions_with_faces_at_end_of_transition,
        predicate = equal(),
        payload_merge_op = lambda starting_faces, ending_faces: {
            'starts': starting_faces,
            'finishes': ending_faces
        }
    )

    # Get all the transitions where the faces at the start and the end are
    # the same
    def face_list_stays_the_same(start_finishes_payload):
        """ Define a scene graph by the face positions at the start and check
        if the face positions at the end satisfy it. """
        graph = {
            'nodes': [
                {
                    'name': 'face{}'.format(idx),
                    'predicates': [ position(face.x1, face.y1, face.x2, face.y2, epsilon=POSITION_EPSILON),
                          lambda face: face['score'] > MINIMUM_FACE_PROBABILITY ]
                }
                for idx, face in enumerate(start_finishes_payload['starts'])
                if face.score > MINIMUM_FACE_PROBABILITY
            ],
            'edges': []
        }
        return scene_graph(graph, exact=True)([
            { 'x1': face.x1, 'y1': face.y1, 'x2': face.x2, 'y2': face.y2, 'score': face.score }
            for face in start_finishes_payload['finishes']
        ])
    bad_transitions = transitions_with_faces.filter(
        payload_satisfies(face_list_stays_the_same))

    # Finally, compute shot boundaries
    def convert_shot_boundaries_to_shots(shot_boundary_list):
        """
        Helper function to convert an IntervalList of shot boundaries to an
        IntervalList of shots.
        
        shot_boundary_list should have the start and end of the movie as
        boundaries.
        """
        def fold_boundaries_to_shots(acc, frame):
            if acc == []:
                return [frame.copy()]
            top = acc[-1]
            top.end = frame.start - 1
            if top.length() > 0:
                acc.append(frame.copy())
            else:
                top.end = frame.start
            return acc

        return shot_boundary_list.fold_list(fold_boundaries_to_shots, [])

    # Convert microshot boundaries to IntervalList
    shot_boundaries = IntervalList([
        (boundary, boundary, 0)
        for boundary in list(set([0, video.num_frames] + microshot_boundaries))
    ])
    microshots = convert_shot_boundaries_to_shots(shot_boundaries)

    # Filter out short microshots
    short_microshots = microshots.filter_length(max_length=math.floor(
        MINIMUM_SHOT_DURATION * video.fps))
    shots = microshots.set_union(
        short_microshots.map(lambda i: (i.start, i.end + 1, i.payload)).coalesce()
    ).coalesce()

    # Remove shots that start with the bad boundaries we found earlier
    bad_shots = shots.filter_against(
        bad_transitions.map(lambda i: (i.start+1, i.end, i.payload)),
        predicate=starts_inv()
    )
    shot_boundaries = shots.map(lambda i: (i.start, i.start, i.payload))
    shot_boundaries_without_bad_shots = shot_boundaries.minus(bad_shots)
    shots = convert_shot_boundaries_to_shots(shot_boundaries_without_bad_shots)

    return shots

def save_shots_to_database(shots, video, labeler, tag):
    new_shots = shots.fold(lambda acc, shot: acc + [
        Shot(min_frame=shot.get_start(),
            max_frame=shot.get_end(),
            labeler=labeler,
            video=video)], [])

    with transaction.atomic():
        Shot.objects.bulk_create(new_shots)
        VideoTag(video=video, tag=tag).save()

# Labeler for HSV histogram shot detection
LABELER_HIST, _ = Labeler.objects.get_or_create(name='shot-hsvhist-face')
LABELED_HIST_TAG, _ = Tag.objects.get_or_create(name='shot-hsvhist-face:labeled')

# Labeler for Face detection
LABELER_FACE, _ = Labeler.objects.get_or_create(name='mtcnn')
LABELED_FACE_TAG, _ = Tag.objects.get_or_create(name='mtcnn:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_HIST_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
#video_ids=sorted(list(all_videos.difference(ids_to_exclude)))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

# Cluster parameters
cfg = cluster_config(num_workers=80, worker=worker_config('n1-standard-32'))
with make_cluster(cfg, no_delete=True) as db_wrapper:
    db = db_wrapper.db
#if True:
#    db = scannerpy.Database() 

    print("Loading histograms from Scanner")

    # Make sure the histograms have been computed already!
    hsv_histograms = st.histograms.compute_hsv_histograms(
        db,
        videos=[video.for_scannertools() for video in list(videos)]
    )

    for idx, hist in enumerate(hsv_histograms):
        if hist is None:
            print(videos[idx].id, 'is None')

    #hsv_histograms_loaded = [hist.load() for hist in tqdm(hsv_histograms)]

    print("Computing microshot boundaries")

    # Compute microshot boundaries
    microshot_boundaries = st.shot_detection.compute_shot_boundaries(
        db,
        videos=[video.for_scannertools() for video in list(videos)],
        histograms=hsv_histograms
    )

    bad_boundaries = []
    for idx, boundaries in enumerate(microshot_boundaries):
        if boundaries is None or boundaries is []:
            bad_boundaries.append(videos[idx].id)
    print("{} movies fail on boundary detection".format(bad_boundaries))

    print("Computing faces")

    # Compute frames FACE_FPS times a second and before and after every microshot
    #   boundary
    frames = [
        frames_to_detect_faces(list(boundaries), video)
        for boundaries, video in zip(microshot_boundaries, videos)
    ]

    # Compute the faces
    faces = st.face_detection.detect_faces(
        db,
        videos = [video.for_scannertools() for video in videos],
        frames=frames,
        run_opts = {'work_packet_size': 20, 'io_packet_size': 1000,
            'checkpoint_frequency': 5}
    )

    bad_movies = []
    for idx, face in enumerate(faces):
        if face is None:
            bad_movies.append(videos[idx].id)
    print('{} movies fail on face detection'.format(bad_movies))

    print('Putting faces into the database')

    # Update the database with all the new faces
    for facelist, framelist, video in tqdm(zip(faces, frames, videos), total=len(videos)):
        if video.id in bad_movies:
            continue
        update_database_with_faces(facelist.load(), framelist, video,
            LABELER_FACE, LABELED_FACE_TAG)

    print("Computing shots")
    
    # Compute shots
    shotlist = [
        compute_shots(list(boundaries), facelist.load(), framelist, video)
        for boundaries, facelist, framelist, video in tqdm(
            zip(microshot_boundaries, faces, frames, videos),
            total=len(videos))
        if video.id not in bad_movies
    ]

    print("Putting shots into the database")

    # Save shots to the database
    for shots, video in tqdm(zip(shotlist, videos), total=len(videos)):
        if video.id in bad_movies:
            continue
        save_shots_to_database(shots, video, LABELER_HIST, LABELED_HIST_TAG)

Notifier().notify("Done with shot detection!")
