import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from django.db import transaction
from query.models import Video, Frame, Labeler, Tag, VideoTag
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from tqdm import tqdm

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='avgintensity')
LABELED_TAG, _ = Tag.objects.get_or_create(name='avgintensity:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))

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
#with make_cluster(cfg) as db_wrapper:
#    db = db_wrapper.db
if True:
    db = scannerpy.Database() 

    print("Detecting brightness")
    # Calculate brightness
    brightness = st.imgproc.compute_brightness_cpp(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames
    )

    print("Saving brightness values in database")
    # Figure out which frames we've already computed face embeddings for
    frames_in_db_already = set([
        (f.video_id, f.number)
        for f in Frame.objects.filter(tags=LABELED_TAG).all()
    ])

    # Tag all the frames as being labeled, and update frames with brightness val
    for video, framelist, brightness_list in tqdm(zip(videos, frames, brightness), total=len(videos)):
        new_frame_tags = []
        frames_to_update = []
        frame_objs = Frame.objects.filter(video_id=video.id,
                number__in=framelist).order_by('number')
        for (frame_obj, brightness_val) in zip(frame_objs, brightness_list.load()):
            if (video.id, frame_obj.number) in frames_in_db_already:
                continue
            new_frame_tags.append(
                    Frame.tags.through(frame_id=frame_obj.pk, tag_id=LABELED_TAG.pk))
            frame_obj.brightness = brightness_val
            frames_to_update.append(frame_obj)
        with transaction.atomic():
            for frame in frames_to_update:
                frame.save()
            Frame.tags.through.objects.bulk_create(new_frame_tags)

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

Notifier().notify('Done with brightness computation')
