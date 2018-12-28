from esper.prelude import *
from .queries import query

@query("Bright frames")
def bright_frames():
    from query.models import Frame
    from esper.stdlib import qs_to_result
    return qs_to_result(Frame.objects.filter(brightness__gt=200).all(), stride=100)
