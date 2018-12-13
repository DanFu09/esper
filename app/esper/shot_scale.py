from esper.prelude import *
import esper.stdlib as stdlib
from django.db.models import ExpressionWrapper, F
import rekall as rk
import rekall.parsers
import rekall.payload_predicates
import esper.rekall
from rekall.video_interval_collection import VideoIntervalCollection
from query.models import *
import numpy as np

# Using https://filmanalysis.coursepress.yale.edu/cinematography/ as reference.
from enum import IntEnum
class ShotScale(IntEnum):
    """ L=Long CU=Close-up X=Extreme M=Medium """
    UNK = 0
    XL = 1
    L = 2
    ML = 3
    M = 4
    CU = 5
    XCU = 6

# Limitations:
# Extreme Close-up is difficult to get since face detector does not work on those frames.
# Long shots and Extreme Long shots are difficult to get since faces are too small for face detectors.
def face_height_to_shot_scale(face_height):
    if face_height >= 0.95:
        return ShotScale.XCU
    if face_height >= 0.5:
        return ShotScale.CU
    if face_height >= 0.25:
        return ShotScale.M
    if face_height >= 0.12:
        return ShotScale.ML
    # Faces are usually not detected anymore in L and XL shots.
    return ShotScale.UNK

# Heuristics to guess scale from pose keypoints.
def pose_keypoints_to_shot_scale(keypoints):
    # If any of the key points in `positions` are detected
    def visible(pose, positions):
        return np.any(pose[positions, 2]>0)
    # Get the y-value of detected key points in `positions` then apply
    # the `reduce` function
    def get_y(pose, positions, reduce):
        rows = pose[positions, :]
        heights = rows[rows[:,2]>0, 1] # only consider existing keypoints
        return reduce(heights)
    # Get the maximum y-value difference between detected keypoints in
    # `upper_pos` and `lower_pos`
    def get_height(pose, upper_pos, lower_pos):
        return get_y(pose, lower_pos, max) - get_y(pose, upper_pos, min)

    pose = np.array(keypoints).reshape((-1,3))
    ankles = [Pose.RAnkle, Pose.LAnkle]
    knees = [Pose.RKnee, Pose.LKnee]
    hips = [Pose.RHip, Pose.LHip]
    shoulders = [Pose.RShoulder, Pose.LShoulder, Pose.Neck]
    head = [Pose.Nose, Pose.LEye, Pose.REye, Pose.REar, Pose.LEar]
    all_pts = head+shoulders+hips+knees+ankles # Excludes arm and hands
    show_ankle = visible(pose, ankles)
    show_knee = visible(pose, knees)
    show_hip = visible(pose, hips)
    show_shoulder = visible(pose, shoulders)
    show_head = visible(pose, head)
    height = get_height(pose, all_pts, all_pts)
    if show_head and show_shoulder and show_hip and show_knee and show_ankle:
        if height >= 0.5:
            return ShotScale.L
        return ShotScale.XL
    if show_head and show_shoulder and show_hip and show_knee:
        if height >= 0.75:
            return ShotScale.ML
        elif height >= 0.4:
            return ShotScale.L
        return ShotScale.XL
    if show_head and show_shoulder and show_hip:
        if height >= 0.75:
            return ShotScale.M
        elif height >= 0.5:
            return ShotScale.ML
        elif height >= 0.2:
            return ShotScale.L
        return ShotScale.XL
    if show_head and show_shoulder:
        if height >= 0.4:
            return ShotScale.CU
        elif height >= 0.15:
            return ShotScale.M
        return ShotScale.UNK
    if show_head:
        if height >= 0.25:
            return ShotScale.XCU
        elif height >= 0.1:
            return ShotScale.CU
    return ShotScale.UNK    

def pose_payload_parser():
    def get_pose(row):
        return {
            'hand_left': row.hand_keypoints()[0].tolist(),
            'hand_right': row.hand_keypoints()[1].tolist(),
            'pose': row.pose_keypoints().tolist(),
            'face': row.face_keypoints().tolist()
        }
    return get_pose

def with_face():
    return rk.parsers.named_payload('face',
             rk.parsers.in_array(
               rk.parsers.bbox_payload_parser(VideoIntervalCollection.django_accessor)))

def with_pose():
    return rk.parsers.named_payload('pose',
            rk.parsers.in_array(
               pose_payload_parser()))

def with_named_empty_list(name):
    return rk.parsers.named_payload(name,
            lambda obj: [])

def merge_named_payload(name_to_merge_op):
    def merge(p1,p2):
        p = {}
        for name, op in name_to_merge_op.items():
            p[name] = op(p1[name], p2[name])
        return p
    return merge

def payload_to_shot_scale(p):
    s = ShotScale.UNK
    if 'face' in p:
        for bbox in p['face']:
            s = max(s, face_height_to_shot_scale(bbox['y2']-bbox['y1']))
    if 'pose' in p:
        for all_pose in p['pose']:
            s = max(s, pose_keypoints_to_shot_scale(all_pose['pose']))
    return s

def label_videos_with_shot_scale(video_ids):
    faces = Face.objects.annotate(
            min_frame=F('frame__number'),
            max_frame=F('frame__number'),
            video_id=F('frame__video__id')).filter(video_id__in=video_ids)
    poses = Pose.objects.annotate(
            min_frame=F('frame__number'),
            max_frame=F('frame__number'),
            video_id=F('frame__video__id')).filter(video_id__in=video_ids)
    face_frames = VideoIntervalCollection.from_django_qs(faces,
            with_payload=rk.parsers.merge_dict_parsers([
                with_face(),
                with_named_empty_list('pose'),
                rk.parsers.dict_payload_parser(
                    VideoIntervalCollection.django_accessor,
                    { 'frame_id': 'frame_id' })
                ]))
    pose_frames = VideoIntervalCollection.from_django_qs(poses,
            with_payload=rk.parsers.merge_dict_parsers([
                with_pose(),
                with_named_empty_list('face'),
                rk.parsers.dict_payload_parser(
                    VideoIntervalCollection.django_accessor,
                    { 'frame_id': 'frame_id' })
                ]))
    faces_with_pose = face_frames.set_union(pose_frames).coalesce(
            merge_named_payload({'pose': rk.merge_ops.payload_plus,
                                 'face': rk.merge_ops.payload_plus,
                                 'frame_id': rk.merge_ops.payload_first}))
    frames_with_shot_scale = faces_with_pose.map(
        lambda intrvl: (intrvl.start, intrvl.end, {
            'pose': intrvl.payload['pose'],
            'face': intrvl.payload['face'],
            'frame_id': intrvl.payload['frame_id'],
            'shot_scale': payload_to_shot_scale(intrvl.payload)
        }))
    return frames_with_shot_scale

def get_all_frames_with_shot_scale(video_id, scale):
    return label_videos_with_shot_scale([video_id]).filter(
            rk.payload_predicates.payload_satisfies(
                lambda p: p['shot_scale']==scale))
