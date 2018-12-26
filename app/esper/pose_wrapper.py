import os
import numpy as np
from rs_embed import EmbeddingData

POSE_DIR = '/app/data/pose'
POSE_PATH = os.path.join(POSE_DIR, 'pose_binary.bin')
ID_PATH = os.path.join(POSE_DIR, 'pose_ids.bin')
POSE_DIM = 390


def _load():
    id_file_size = os.path.getsize(ID_PATH)
    assert id_file_size % 8 == 0, \
        'Id file size is not a multiple of sizeof(u64)'
    n = int(id_file_size / 8)
    emb_file_size = os.path.getsize(POSE_PATH)
    assert emb_file_size % 4 == 0, \
        'Embedding file size is a multiple of sizeof(f32)'
    d = int((emb_file_size / 4) / (id_file_size / 8))
    assert emb_file_size % d == 0, \
        'Embedding file size is a multiple of d={}'.format(d)
    emb_data = EmbeddingData(ID_PATH, POSE_PATH, POSE_DIM)
    assert emb_data.count() == n, \
        'Count does not match expected: {} != {}'.format(n, emb_data.count())
    return emb_data


_POSE_DATA = _load()


class PoseWrapper():
    def __init__(self, keypoints, pose_id, labeler):
        self.kp = np.array(keypoints).reshape(130, 3)
        self.id = pose_id
        self.labeler = labeler

    POSE_KEYPOINTS = 18
    FACE_KEYPOINTS = 70
    HAND_KEYPOINTS = 21

    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

    def pose_keypoints(self):
        return self.kp[:self.POSE_KEYPOINTS, :]

    def face_keypoints(self):
        return self.kp[self.POSE_KEYPOINTS:(self.POSE_KEYPOINTS + self.FACE_KEYPOINTS), :]

    def hand_keypoints(self):
        base = self.kp[self.POSE_KEYPOINTS + self.FACE_KEYPOINTS:, :]
        return [base[:self.HAND_KEYPOINTS, :], base[self.HAND_KEYPOINTS:, :]]


def get(pose_meta_qs):
    """Generator of PoseMeta objects -> list of PoseWrapper objects."""
    pose_meta_qs = list(pose_meta_qs)
    ids = [p.id for p in pose_meta_qs]

    # get returns list of (id, pose bytes)
    result = _POSE_DATA.get(ids)
    assert len(result) == len(pose_meta_qs), "{} != {}".format(
        len(result), len(pose_meta_qs))

    return [
        PoseWrapper(np.array(pose_id_bytes[1]), pose_id_bytes[0], pm.labeler)
        for pm, pose_id_bytes in zip(pose_meta_qs, result)
    ]


def exists(pose_meta_qs):
    """Generator of PoseMeta objects -> List of bools"""
    pose_meta_qs = list(pose_meta_qs)
    ids = [p.id for p in pose_meta_qs]
    return _POSE_DATA.exists(ids)

