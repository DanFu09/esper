import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, FaceFeatures
from esper.prelude import Notifier
import json

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(name__contains='godfather')
db = scannerpy.Database()

# Calculate at 2 fps
frames = [
    list(range(0, video.num_frames, int(round(video.fps) / 2)))
    for video in videos
]

# Load faces
faces = st.face_detection.detect_faces(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    megabatch=1
)

# Compute face embeddings
features = st.face_embedding.embed_faces(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    bboxes=faces,
    megabatch=1
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='facenet')
LABELED_TAG, _ = Tag.objects.get_or_create(name='facenet:labeled')

new_features = []
for video, framelist, facelist, featurelist in zip(videos, frames, faces, features):
    for frame, faces_in_frame, features_in_frame in zip(framelist, facelist.load(), featurelist.load()):
        if len(faces_in_frame) == 0:
            continue
        face_objs = list(Face.objects.filter(frame__video_id=video.id).filter(frame__number=frame).all())
        for bbox, feature_vec in zip(faces_in_frame, features_in_frame):
            face_obj = None
            for obj in face_objs:
                if (abs(obj.bbox_x1 - bbox.x1) < .000001 and
                    abs(obj.bbox_x2 - bbox.x2) < .000001 and
                    abs(obj.bbox_y1 - bbox.y1) < .000001 and
                    abs(obj.bbox_y2 - bbox.y2) < .000001 and
                    abs(obj.probability - bbox.score) < .000001):
                    face_obj = obj
                    break
            if face_obj is None:
                print("Couldn't find face {} in {}".format(bbox, face_objs))
            new_features.append(FaceFeatures(
                face=face_obj,
                features=json.dumps(feature_vec.tolist()).encode(),
                labeler=LABELER
            ))
FaceFeatures.objects.bulk_create(new_features)

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

Notifier().notify('Done with face embeddings')
