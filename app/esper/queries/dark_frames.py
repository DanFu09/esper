from esper.prelude import *
from .queries import query

@query("Dark frames")
def dark_frames():
    from query.models import Frame
    from esper.stdlib import qs_to_result
    return qs_to_result(Frame.objects.filter(brightness__lt=50).all(), stride=100)
