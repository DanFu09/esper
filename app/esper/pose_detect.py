import scannerpy 
import scannertools as st
import os
import sys
from django.db.models import Q
from query.models import Video, Frame, Labeler, Tag, VideoTag, Pose
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from tqdm import tqdm

# Labeler for this pipeline
FRAME_TO_COMPUTE = Tag.objects.get(name='face_computed')
LABELER, _ = Labeler.objects.get_or_create(name='openpose')
LABELED_TAG, _ = Tag.objects.get_or_create(name='openpose:labeled')

# Get all the videos that haven't been labeled with this pipeline
decode_errors = set([
    video.id
    for video in Video.objects.filter(decode_errors=True)
])

small_dataset_ids = set([
    video.id
    for video in Video.objects.filter(small_dataset=True)
])

ids_to_exclude = decode_errors

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

print("Loading frames from Database")
frames = [
    [
        f.number
        for f in Frame.objects.filter(video_id=video.id, tags=FRAME_TO_COMPUTE).order_by('number').all()]
    for video in tqdm(videos)
]

cfg = st.kube.ClusterConfig(
    num_workers=50,
    id='df-test',
    autoscale=True,
    master=st.kube.MachineConfig(
        'gcr.io/visualdb-1046/esper-base-dan:gpu',
        type=st.kube.MachineTypeName(name='n1-highmem-32'),
        disk=250,
        gpu=2,
        gpu_type='nvidia-tesla-p100',
        preemptible=False
    ),
    worker=st.kube.MachineConfig(
        'gcr.io/visualdb-1046/esper-base-dan:gpu',
        type=st.kube.MachineTypeName(name='n1-standard-32'),
        disk=250,
        gpu=2,
        gpu_type='nvidia-tesla-p100',
        preemptible=True
    ),
    pipelines=[st.pose_detection.PoseDetectionPipeline])
with make_cluster(cfg, no_delete=True) as db_wrapper:
    db = db_wrapper.db
#if True:
#    db = scannerpy.Database() 

    print("Computing poses")

    poses = st.pose_detection.detect_poses(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames
    )

    print("Putting poses in database")
    # Figure out which frames we've already computed face embeddings for
    frames_in_db_already = set([
        (f.video_id, f.number)
        for f in Frame.objects.filter(tags=LABELED_TAG).all()
    ])

    for video, framelist, poseframelist in tqdm(zip(videos, frames, poses), total=len(videos)):
        new_poses = []
        frame_objs = Frame.objects.filter(video_id=video.id, number__in=framelist).order_by('number')
        for frame, posesinframe in zip(frame_objs, poseframelist.load()):
            if (video.id, frame) in frames_in_db_already:
                continue
            for pose in posesinframe:
                new_pose = Pose(
                    keypoints=pose.keypoints.tobytes(),
                    labeler=LABELER,
                    frame=frame
                )
                new_poses.append(new_pose)
        Pose.objects.bulk_create(new_poses, batch_size=10000)

    print("Tagging frames as being labeled")
    # Tag all the frames as being labeled
    for video, framelist in tqdm(zip(videos, frames), total=len(videos)):
        new_frame_tags = []
        frame_objs = Frame.objects.filter(video_id=video.id, number__in=framelist)
        for frame in frame_objs:
            if (video.id, frame.number) in frames_in_db_already:
                continue
            new_frame_tags.append(
                Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
        Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=10000)

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

Notifier().notify('Done with pose detection')
