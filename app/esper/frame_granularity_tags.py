import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag
from esper.prelude import Notifier
import json
from tqdm import tqdm

# Faces were computed at this many FPS
FACE_FPS = 2

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

TAG, _ = Tag.objects.get_or_create(name="face_computed")

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

all_videos = set([video.id for video in Video.objects.all()])
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))

print(video_ids, len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

db = scannerpy.Database()


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

print("Loading microshot boundaries")

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

print("Computing frames to compute on")
# Compute frames FACE_FPS times a second and before and after every microshot
#   boundary
frames = [
    frames_to_detect_faces(list(boundaries), video)
    for boundaries, video in zip(microshot_boundaries, videos)
]

print("Saving frames to database")

frames_in_db_already = set([
    (f.video_id, f.number)
    for f in Frame.objects.filter(tags=TAG).all()
])

# Put frame objects in database
new_frames = []
for video, framelist in tqdm(zip(videos, frames), total=len(videos)):
    frames_existing = set([f.number for f in Frame.objects.filter(video_id=video.id)])
    new_frames += [
        Frame(video=video, number=num)
        for num in framelist
        if num not in frames_existing
    ]
Frame.objects.bulk_create(new_frames)

print("Saving frame tags to database")

# Tag all the frames as being labeled
new_frame_tags = []
for video, framelist in tqdm(zip(videos, frames), total=len(videos)):
    frame_objs = Frame.objects.filter(video_id=video.id).filter(number__in=framelist)
    frame_obj_nums = set([f.number for f in frame_objs])
    if frame_objs.count() != len(framelist):
        print('Not all frames in Database for video {}'.format(video.id))
        print('{} frames in DB, {} frames wanted'.format(len(frame_obj_nums), len(framelist)))
    for frame in frame_objs:
        new_frame_tags.append(
                Frame.tags.through(frame_id=frame.pk, tag_id=TAG.pk))
Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=100000)
