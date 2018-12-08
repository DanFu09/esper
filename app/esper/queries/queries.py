from esper.prelude import *
from esper.rekall import *
from collections import defaultdict
from functools import reduce
import inspect
import os

queries = []


def query(name):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__

    def wrapper(f):
        lines = inspect.getsource(f).split('\n')
        lines = lines[:-1]  # Seems to include a trailing newline

        # Hacky way to get just the function body
        i = 0
        while True:
            if "():" in lines[i]:
                break
            i = i + 1

        fn = lines[i:]
        fn += ['FN = ' + f.__name__]
        queries.append([name, '\n'.join(fn)])

        return f

    return wrapper

from .all_faces import *
from .all_faces_rekall import *
from .all_videos import *
from .all_poses import *
from .all_face_landmarks import *
from .shot_reverse_shot import *
from .man_woman_up_close import *
from .faces_from_poses import *
from .frames_with_two_women import *
from .three_people import *
