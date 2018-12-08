from esper.prelude import *
from .queries import query

@query("All poses")
def all_poses():
    from query.models import Pose
    from esper.stdlib import qs_to_result
    return qs_to_result(Pose.objects.all(), stride=1000)
