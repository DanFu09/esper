import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, Shot
from esper.prelude import Notifier

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(id=123)
db = scannerpy.Database()

shots = [
    Shot.objects.filter(video_id=video.id).all()
    for video in videos
]

# Calculate at 2 fps
frames_2fps = [
    list(range(0, video.num_frames, int(round(video.fps) / 2)))
    for video in videos
]

frames = []
for shot_enumerator, framelist in zip(shots, frames_2fps):
    shot_boundaries = [
        (shot.min_frame, shot.max_frame)
        for shot in shot_enumerator
    ]
    starts = set([bounds[0] for bounds in shot_boundaries])
    ends = set([bounds[1] for bounds in shot_boundaries])
    framelist = set(framelist).union(starts).union(ends)
    frames.append(sorted(list(framelist)))

# Detect faces
faces = st.face_detection.detect_faces(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    megabatch=1
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='mtcnn')
LABELED_TAG, _ = Tag.objects.get_or_create(name='mtcnn:labeled')

# Insert new frames
new_frame_objs = []
for video, framelist in zip(videos, frames):
    frames_existing = Frame.objects.filter(video_id=video.id).all()
    frame_nums_existing = [frame.number for frame in frames_existing]
    new_frames = [frame for frame in framelist
            if frame not in frame_nums_existing]

    # Annotate frames with shot metadata
    new_frames_with_metadata = []
    shots = Shot.objects.filter(video_id=video.id).order_by('min_frame').all() 
    shot_iter = iter(shots)
    cur_shot = next(shot_iter)
    for f in new_frames:
        while f > cur_shot.max_frame:
            cur_shot = next(shot_iter)
        boundary = (f == cur_shot.min_frame or f == cur_shot.max_frame)
        new_frames_with_metadata.append((f, boundary))

    new_frame_objs = new_frame_objs + [
            Frame(number=frame, shot_boundary=boundary, video_id=video.id)
            for frame, boundary in new_frames_with_metadata]

Frame.objects.bulk_create(new_frame_objs)

# Tag all the frames as being labeled
new_frame_tags = []
for video, framelist in zip(videos, frames):
    frame_objs = Frame.objects.filter(video_id=video.id).filter(number__in=framelist)
    for frame in frame_objs:
        new_frame_tags.append(
                Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))

Frame.tags.through.objects.bulk_create(new_frame_tags)

# Tag this video as being labeled
new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
VideoTag.objects.bulk_create(new_videotags)

# Put faces in database
new_faces = []
for video, framelist, facelist in zip(videos, frames, faces):
    frame_objs = Frame.objects.filter(video_id=video.id).filter(
            number__in=framelist).order_by('number')
    
    # Annotate frame objects with shot metadata
    frame_objs_with_metadata = []
    shots = Shot.objects.filter(video_id=video.id).order_by('min_frame').all() 
    shot_iter = iter(shots)
    cur_shot = next(shot_iter)
    for frame in frame_objs:
        while frame.number > cur_shot.max_frame:
            cur_shot = next(shot_iter)
        frame_objs_with_metadata.append((frame, cur_shot))

    for frame_with_shot, bboxlist in zip(frame_objs_with_metadata,
            facelist.load()):
        for bbox in bboxlist:
            new_faces.append(Face(
                frame=frame_with_shot[0],
                shot=frame_with_shot[1],
                bbox_x1=bbox.x1,
                bbox_x2=bbox.x2,
                bbox_y1=bbox.y1,
                bbox_y2=bbox.y2,
                probability=bbox.score,
                labeler=LABELER))
Face.objects.bulk_create(new_faces)

Notifier().notify('Done with face detection')
