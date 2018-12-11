from esper.prelude import *
from esper.shot_scale import ShotScale as ShotScaleEnum, label_videos_with_shot_scale
from query.models import Frame, ShotScale, Face, Pose, VideoTag
from tqdm import tqdm

shot_scale_names = ['UNK', 'XL', 'L', 'ML', 'M', 'CU', 'XCU']
shot_scales = [ShotScale.objects.get_or_create(name=name)[0]
        for name in shot_scale_names]

video_ids = [videotag.video_id
    for videotag in VideoTag.objects.filter(tag_id=2).exclude(video_id=123).all()]
print(video_ids)
frames_with_shot_scales = label_videos_with_shot_scale(video_ids)
for video_id in tqdm(video_ids):
    frames = frames_with_shot_scales.get_intervallist(video_id)
    for frame in frames.get_intervals():
        frame_id = frame.payload['frame_id']
        shot_scale = shot_scales[frame.payload['shot_scale']]

        frame_obj = Frame.objects.get(pk=frame_id)
        frame_obj.shot_scale = shot_scale
        frame_obj.save()

Notifier().notify('Done with shot scale labeling!')
