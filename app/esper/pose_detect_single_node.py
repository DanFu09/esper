import scannerpy 
import scannertools as st
import os
import sys
from django.db.models import Q
from query.models import Video, Frame, Labeler, Tag, VideoTag, Pose
from esper.prelude import Notifier

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(name="star wars episode i the phantom menace")
db = scannerpy.Database()

# Calculate at 2 fps
frames = [
    list(range(0, video.num_frames, int(round(video.fps) / 2)))
    for video in videos
]

poses = st.pose_detection.detect_poses(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    cache=False
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='openpose')
LABELED_TAG, _ = Tag.objects.get_or_create(name='openpose:labeled')

new_poses = []
for video, framelist, poseframelist in zip(videos, frames, poses):
    frame_objs = Frame.objects.filter(video_id=video.id).filter(number__in=framelist).order_by('number')
    for frame, posesinframe in zip(frame_objs, poseframelist.load()):
        for pose in posesinframe:
            new_pose = Pose(
                    keypoints=pose.keypoints.tobytes(),
                    labeler=LABELER,
                    frame=frame
            )
            new_poses.append(new_pose)
Pose.objects.bulk_create(new_poses, batch_size=100000)

# Tag all the frames as being labeled
#new_frame_tags = []
#for video, framelist in zip(videos, frames):
#    frame_objs = Frame.objects.filter(video_id=video.id).filter(number__in=framelist)
#    for frame in frame_objs:
#        new_frame_tags.append(
#                Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
#Frame.tags.through.objects.bulk_create(new_frame_tags)

# Tag this video as being labeled
new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with pose detection')
