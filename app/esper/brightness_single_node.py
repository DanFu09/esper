import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from django.db import transaction
from query.models import Video, Frame, Labeler, Tag, VideoTag
from esper.prelude import Notifier

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(name__contains='godfather')
db = scannerpy.Database()

# Calculate at 2 fps
frames = [
    list(range(0, video.num_frames, int(round(video.fps) / 2)))
    for video in videos
]

# Calculate brightness
brightness = st.imgproc.compute_brightness_cpp(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    megabatch=1
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='avgintensity')
LABELED_TAG, _ = Tag.objects.get_or_create(name='avgintensity:labeled')

# Tag all the frames as being labeled, and update frames with brightness val
new_frame_tags = []
frames_to_update = []
for video, framelist, brightness_list in zip(videos, frames, brightness):
    frame_objs = Frame.objects.filter(video_id=video.id,
            number__in=framelist).order_by('number')
    for (frame_obj, brightness_val) in zip(frame_objs, brightness_list.load()):
        new_frame_tags.append(
                Frame.tags.through(frame_id=frame_obj.pk, tag_id=LABELED_TAG.pk))
        frame_obj.brightness = brightness_val
        frames_to_update.append(frame_obj)
with transaction.atomic():
    for frame in frames_to_update:
        frame.save()
Frame.tags.through.objects.bulk_create(new_frame_tags)

# Tag this video as being labeled
new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with brightness computation')
