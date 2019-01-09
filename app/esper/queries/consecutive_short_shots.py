from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Consecutive_short_shots (rekall)")
def consecutive_short_shots():
    from query.models import Shot
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.temporal_predicates import meets_before
    from esper.rekall import intrvllists_to_result_with_objects
    from django.db.models import ExpressionWrapper, FloatField

    NUM_SHOTS=3
    MAX_SHOT_DURATION=0.5

    short_shots = VideoIntervalCollection.from_django_qs(Shot.objects.annotate(
        duration = ExpressionWrapper((F('max_frame') - F('min_frame')) / F('video__fps'), output_field=FloatField())
    ).filter(
        duration__lt=MAX_SHOT_DURATION,
        duration__gt=0.,
        labeler__name='shot-hsvhist-face'
    ).all())

    n_shots = short_shots
    for n in range(2, NUM_SHOTS + 1):
        print('Constructing {} consecutive short shots'.format(n))
            
        n_shots = n_shots.merge(
            short_shots, predicate=meets_before(epsilon=1), working_window=1
        ).coalesce().filter_length(min_length=1)

        print('There are {} videos with {} consecutive short shots'.format(
            len(n_shots.get_allintervals().keys()), n)
        )

    return intrvllists_to_result_with_objects(n_shots, lambda a, b: [], limit=100, stride=1)
