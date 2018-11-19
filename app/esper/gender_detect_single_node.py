import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, FaceGender, Gender
from esper.prelude import Notifier

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(
        Q(name__contains="star wars") | Q(name__contains="harry potter")) \
                .filter(~Q(name="star wars episode i the phantom menace"))
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
    frames=frames
)

# Detect genders
genders = st.gender_detection.detect_genders(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    bboxes=faces
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='rudecarnie')
LABELED_TAG, _ = Tag.objects.get_or_create(name='rudecarnie:labeled')

gender_set = set()
for resultlist in genders:
    for result in resultlist.load():
        for gender_label, score in result:
            gender_set.add(gender_label)
gender_objs = { 
        gender: Gender.objects.get_or_create(name=gender)[0] 
        for gender in list(gender_set) }

# Put facegender labels in database
new_facegenders = []
for video, framelist, facelist, genderlist in zip(videos, frames, faces, genders):
    for frame, faces_in_frame, genders_in_frame in zip(framelist, facelist.load(), genderlist.load()):
        if len(faces_in_frame) == 0:
            continue
        face_objs = list(Face.objects.filter(frame__video_id=video.id).filter(frame__number=frame).all())
        for bbox, gender in zip(faces_in_frame, genders_in_frame):
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
            new_facegenders.append(FaceGender(
                face=face_obj,
                gender=gender_objs[gender[0]],
                probability=gender[1],
                labeler=LABELER
            ))
FaceGender.objects.bulk_create(new_facegenders)

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

Notifier().notify('Done with gender detection')
