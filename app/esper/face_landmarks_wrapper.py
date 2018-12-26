import os
import numpy as np
from rs_embed import EmbeddingData
from query.models import Labeler

LANDMARKS_DIR = '/app/data/landmarks'
LANDMARKS_PATH = os.path.join(LANDMARKS_DIR, 'landmarks_binary.bin')
ID_PATH = os.path.join(LANDMARKS_DIR, 'landmarks_ids.bin')
LANDMARKS_DIM = 272

def _load():
    id_file_size = os.path.getsize(ID_PATH)
    assert id_file_size % 8 == 0, \
        'Id file size is not a multiple of sizeof(u64)'
    n = int(id_file_size / 8)
    emb_file_size = os.path.getsize(LANDMARKS_PATH)
    assert emb_file_size % 4 == 0, \
        'Embedding file size is a multiple of sizeof(f32)'
    d = int((emb_file_size / 4) / (id_file_size / 8))
    assert emb_file_size % d == 0, \
        'Embedding file size is a multiple of d={}'.format(d)
    emb_data = EmbeddingData(ID_PATH, LANDMARKS_PATH, LANDMARKS_DIM)
    assert emb_data.count() == n, \
        'Count does not match expected: {} != {}'.format(n, emb_data.count())
    return emb_data


_LANDMARKS_DATA = _load()
LABELER = Labeler.objects.get(name='fan2d')


class LandmarksWrapper():
    def __init__(self, landmarks, landmarks_id, labeler):
        self.landmarks = np.frombuffer(
            np.array(landmarks, dtype=np.float32).tobytes(),
            dtype=np.float64).reshape(68, 2)
        self.id = landmarks_id
        self.labeler = labeler

    # Slice values for each set of landmarks
    FACE_OUTLINE = (0, 17)
    RIGHT_EYEBROW = (17, 22)
    LEFT_EYEBROW = (22, 27)
    NOSE_BRIDGE = (27, 31)
    NOSE_BOTTOM = (31, 36)
    RIGHT_EYE = (36, 42)
    LEFT_EYE = (42, 48)
    OUTER_LIPS = (48, 60)
    INNER_LIPS = (60, 68)

    def _get_landmarks(self, slice_values):
        return self.landmarks[slice_values[0]:slice_values[1]]

    def face_outline(self):
        return self._get_landmarks(self.FACE_OUTLINE)

    def right_eyebrow(self):
        return self._get_landmarks(self.RIGHT_EYEBROW)

    def left_eyebrow(self):
        return self._get_landmarks(self.LEFT_EYEBROW)

    def nose_bridge(self):
        return self._get_landmarks(self.NOSE_BRIDGE)

    def nose_bottom(self):
        return self._get_landmarks(self.NOSE_BOTTOM)

    def right_eye(self):
        return self._get_landmarks(self.RIGHT_EYE)

    def left_eye(self):
        return self._get_landmarks(self.LEFT_EYE)

    def outer_lips(self):
        return self._get_landmarks(self.OUTER_LIPS)

    def inner_lips(self):
        return self._get_landmarks(self.INNER_LIPS)

def get(faces_qs):
    """Generator of Face objects -> list of LandmarksWrapper objects."""
    faces_qs = list(faces_qs)
    ids = [f.id for f in faces_qs]

    # get returns list of (id, landmarks bytes)
    result = _LANDMARKS_DATA.get(ids)
    assert len(result) == len(faces_qs), "{} != {}".format(
        len(result), len(faces_qs))

    return [
        LandmarksWrapper(np.array(landmarks_id_bytes[1]), landmarks_id_bytes[0], LABELER)
        for landmarks_id_bytes in result
    ]
