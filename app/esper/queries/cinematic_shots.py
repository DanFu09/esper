from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Cinematic shots (rekall)")
def cinematic_shots_rekall():
    from query.models import Shot, Labeler
    from rekall.video_interval_collection import VideoIntervalCollection
    from esper.rekall import intrvllists_to_result_with_objects
    from esper.stdlib import qs_to_result
        
    video_ids = [1]
    shots_qs = Shot.objects.filter(
        video_id__in=video_ids,
        labeler=Labeler.objects.get(name='shot-hsvhist-face'))

    shots = VideoIntervalCollection.from_django_qs(shots_qs)

    return intrvllists_to_result_with_objects(shots.get_allintervals(),
        lambda payload, video: [])
