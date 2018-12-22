import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, FaceGender, Gender
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from tqdm import tqdm

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='rudecarnie')
LABELED_TAG, _ = Tag.objects.get_or_create(name='rudecarnie:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))
#video_ids = sorted(list([
#    video.id
#    for video in Video.objects.filter(small_dataset=True).all()
#]))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

print("Getting frames to compute on")
# Get frames that we computed faces on
frames = [
    [
        f.number
        for f in Frame.objects.filter(video_id=video,
            tags__name='face_computed').order_by('number')
    ]
    for video in tqdm(video_ids)
]

# Cluster parameters
cfg = cluster_config(num_workers=80, worker=worker_config('n1-standard-32'),
    pipelines=[st.gender_detection.GenderDetectionPipeline])
#with make_cluster(cfg, no_delete=True) as db_wrapper:
#    db = db_wrapper.db
if True:
    db = scannerpy.Database() 

    print("Loading faces from Scanner")
    # Load faces
    faces = st.face_detection.detect_faces(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames
    )

    print("Detecting genders")

    # Detect genders
    genders = st.gender_detection.detect_genders(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames,
        bboxes=faces
    )

    print("Putting genders in database")
    # Figure out which frames we've already computed face embeddings for
    frames_in_db_already = set([
        (f.video_id, f.number)
        for f in Frame.objects.filter(tags=LABELED_TAG).all()
    ])

    gender_objs = {}

    # Get faces that already have genders
    faces_with_genders = set([
        fg.face_id
        for fg in FaceGender.objects.filter(face__frame__video_id__in=video_ids)
    ])

    # Put facegender labels in database
    for video, framelist, facelist, genderlist in tqdm(zip(videos, frames, faces, genders), total=len(videos)):
        new_facegenders = []
        for frame, faces_in_frame, genders_in_frame in zip(framelist, facelist.load(), genderlist.load()):
            if (video.id, frame) in frames_in_db_already:
                continue
            if len(faces_in_frame) == 0:
                continue
            face_objs = list(Face.objects.filter(frame__video_id=video.id, frame__number=frame).all())
            for bbox, gender in zip(faces_in_frame, genders_in_frame):
                face_obj = None
                for obj in face_objs:
                    if (abs(obj.bbox_x1 - bbox.x1) < .00001 and
                        abs(obj.bbox_x2 - bbox.x2) < .00001 and
                        abs(obj.bbox_y1 - bbox.y1) < .00001 and
                        abs(obj.bbox_y2 - bbox.y2) < .00001 and
                        abs(obj.probability - bbox.score) < .00001):
                        face_obj = obj
                        break
                if face_obj is None:
                    print("Couldn't find face {} in {}".format(bbox, face_objs))
                    continue
                if face_obj.id in faces_with_genders:
                    continue
                
                if gender[0] not in gender_objs:
                    gender_objs[gender[0]] = Gender.objects.get_or_create(name=gender[0])[0]

                new_facegenders.append(FaceGender(
                    face=face_obj,
                    gender=gender_objs[gender[0]],
                    probability=gender[1],
                    labeler=LABELER
                ))
        FaceGender.objects.bulk_create(new_facegenders, batch_size=100000)

         Tag all the frames as being labeled
        new_frame_tags = []
        frame_objs = Frame.objects.filter(video_id=video.id, number__in=framelist)
        for frame in frame_objs:
            if (video.id, frame.number) in frames_in_db_already:
                continue
            new_frame_tags.append(
                    Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
        Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=100000)

    # Get the videos that already have the tag
    videos_tagged_already = set([
        vtag.video_id
        for vtag in VideoTag.objects.filter(tag=LABELED_TAG).all()
    ])

    # Tag this video as being labeled
    new_videotags = [
        VideoTag(video=video, tag=LABELED_TAG)
        for video in videos
        if video.id not in videos_tagged_already
    ]
    VideoTag.objects.bulk_create(new_videotags)

    print("Finished putting everything in the database")

Notifier().notify('Done with gender detection')
