from esper.prelude import *
from query.models import Video, Shot, Labeler, Face, PoseMeta, Frame
import numpy as np
from django.db.models import Avg
from tqdm import tqdm
import esper.pose_wrapper as pw
from esper.shot_scale import ShotScale as ShotScaleEnum
import rekall
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.interval_list import IntervalList
from rekall.merge_ops import payload_plus, payload_second
from rekall.temporal_predicates import overlaps
from esper.rekall import intrvllists_to_result_with_objects

def count_in_region(poses, region):
    def in_region(pose):
        # Take only detected keypoints
        xs = pose[pose[:,2]>0,0]
        result = (xs >= region[0]) & (xs < region[1])
        return np.any(result)
    return len([pose for pose in poses if in_region(pose.pose_keypoints())])

def truncate(val, maxval):
    return val if val < maxval else maxval

# Find the scale for shot from scales of sampled frames
def scale_for_shot(scales):
    scales = [scale for scale in scales if (scale != ShotScaleEnum.UNKNOWN)]
    if len(scales) == 0:
        return ShotScaleEnum.UNKNOWN
    counter={}
    for s in ShotScaleEnum:
        counter[s]=0
    for scale in scales:
        counter[scale] += 1
    best_c = 0
    best = ShotScaleEnum.UNKNOWN
    for s in ShotScaleEnum:
        if counter[s] >= best_c:
            best_c = counter[s]
            best = s
    return best

# Find the poses for shot from pose_metas in sampled frames
def poses_for_shot(pose_metas_for_frames):
    pose_metas = max(pose_metas_for_frames, key=len)
    return pw.get(pose_metas)

class ShotFeatures():
    MAX_COUNT = 5
    REGIONS = [(0,1/3),(1/3,2/3),(2/3,1)]
    def __init__(self, scale, poses):
        self.scale = scale
        self.n_people = truncate(len(poses), ShotFeatures.MAX_COUNT)
        self.counts = tuple(truncate(count_in_region(poses, r), ShotFeatures.MAX_COUNT) for r in ShotFeatures.REGIONS)
        self.pose_ids = [pose.id for pose in poses]
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__) 

def get_shots_for_video(vid):

    shots_qs = Shot.objects.filter(
        video__id=vid,
        labeler=Labeler.objects.get(name='shot-hsvhist-face')
    ).all()
    total = shots_qs.count()
    print("Total shots:", total)
    shots = VideoIntervalCollection.from_django_qs(
        shots_qs,
        with_payload=lambda row:[],
        progress=False,
        total=total
    )
    
    # Take all sampled frames: every 12th.
    frames_qs = Frame.objects.filter(video__id=vid).annotate(numbermod=F('number') % 12).filter(
            numbermod=0).annotate(scale=F("shot_scale__name"))
    total = frames_qs.count()
    print("Total frames with scale:", total)
    shot_scales = VideoIntervalCollection.from_django_qs(
        frames_qs,
        schema={
            "start": "number",
            "end": "number",
        },
        with_payload=lambda f: [ShotScaleEnum[f.scale.upper()]],
        progress=False, total=total)
    
    # Take all poses
    poses_qs = PoseMeta.objects.filter(frame__video__id=vid).annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id')
    )
    total = poses_qs.count()
    print("Total Poses:", total)
    poses = VideoIntervalCollection.from_django_qs(
        poses_qs,
        with_payload=lambda row: [row],
        progress=False,
        total=total
    ).coalesce(payload_merge_op=payload_plus)
    
    print("Merging scales into shots")
    # Merge scales into shots
    shots_with_scale = shots.merge(
        shot_scales,
        payload_merge_op = payload_second,
        predicate=overlaps(),
        working_window=1
    ).coalesce(
        payload_merge_op=payload_plus
    ).map(
        lambda shot_interval: (shot_interval.get_start(), shot_interval.get_end(),
                              {"scale": scale_for_shot(shot_interval.get_payload())})
    )
    
    print("Merging poses into shots")
    # Merge poses into shots
    shots_with_poses = shots.merge(
        poses.map(lambda shot_interval: (shot_interval.get_start(), shot_interval.get_end(), [shot_interval.get_payload()])),
        payload_merge_op = payload_second,
        predicate=overlaps(),
        working_window=1
    ).coalesce(
        # Get a list of list of poses for each shot
        payload_merge_op = payload_plus
    ).map(lambda shot_interval: (shot_interval.get_start(), shot_interval.get_end(),
                                 {"poses": poses_for_shot(shot_interval.get_payload())}))
    
    print("Computing shot features")
    # Get shots with shot features
    shots = shots_with_scale.merge(
        shots_with_poses,
        payload_merge_op = lambda d1, d2: {**d1,**d2},
        predicate=overlaps(),
        working_window=1
    ).coalesce().map(
        lambda intv: (intv.get_start(), intv.get_end(), ShotFeatures(
            intv.get_payload()["scale"], intv.get_payload()["poses"])))
    return shots

if __name__ == '__main__':
    import pickle
    all_videos = Video.objects.filter(decode_errors=False).exclude(id=344).order_by("id").all()
    vids = [vid.id for vid in all_videos]
    for vid in tqdm(vids):
       shots = get_shots_for_video(vid).get_intervallist(vid) 
       pickle.dump(shots, open("data/shot_features/{0:03d}_intervallist.p".format(vid), "wb"))
