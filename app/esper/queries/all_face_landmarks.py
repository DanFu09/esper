from esper.prelude import *
from .queries import query

@query("All face landmarks")
def all_face_landmarks():
    from query.models import FaceLandmarks
    from esper.stdlib import qs_to_result
    return qs_to_result(FaceLandmarks.objects.all(), stride=1000)
