from esper.prelude import *
from esper.shot_scale import ShotScale as ShotScaleEnum, label_videos_with_shot_scale
from query.models import Frame, ShotScale, Face, Pose, VideoTag
from tqdm import tqdm

shot_scale_enum_to_row = {}
for scale in ShotScaleEnum:
    shot_scale_enum_to_row[scale] = ShotScale.objects.get(name=scale.name.lower())
print(shot_scale_enum_to_row)

video_ids = [videotag.video_id
    for videotag in VideoTag.objects.filter(tag_id=2).order_by('video_id').all()]

print(video_ids)

for video_id in tqdm(video_ids):
    frames = label_videos_with_shot_scale([video_id]).get_intervallist(video_id)
    for frame in frames.get_intervals():
        frame_id = frame.payload['frame_id']
        shot_scale = shot_scale_enum_to_row[frame.payload['shot_scale']]
        frame_obj = Frame.objects.get(pk=frame_id)
        frame_obj.shot_scale = shot_scale
        frame_obj.save()

Notifier().notify('Done with shot scale labeling!')
