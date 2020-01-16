from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Manual shots (rekall)")
def manual_shots_rekall():
    from query.models import Shot, Labeler
    from rekall.video_interval_collection import VideoIntervalCollection
    from esper.rekall import intrvllists_to_result
    from esper.stdlib import qs_to_result
        
    shots_qs = Shot.objects.filter(
        labeler__name__contains='manual')

    shots = VideoIntervalCollection.from_django_qs(shots_qs)

    return intrvllists_to_result_with_objects(shots.get_allintervals())
