from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Shot/Reverse Shot with Context (rekall)")
def shot_reverse_shot_with_context():
    from rekall.bbox_predicates import height_at_least
    from rekall.list_predicates import length_at_most
    from rekall.logical_predicates import and_pred, or_pred
    from rekall.merge_ops import payload_plus
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph, make_region
    from rekall.temporal_predicates import before, after, overlaps_before
    from rekall.video_interval_collection import VideoIntervalCollection
    from esper.rekall import intrvllists_to_result, add_intrvllists_to_result, intrvllists_to_result_with_objects

    # If True, visualize results in a timeline
    TIMELINE_OUTPUT = False

    RIGHT_HALF_MIN_X = 0.45
    LEFT_HALF_MAX_X = 0.55
    MIN_FACE_HEIGHT = 0.4
    MAX_FACES_ON_SCREEN = 2
    # faces are sampled every 12 frames
    SAMPLING_RATE = 12
    ONE_SECOND = 24
    FOUR_SECONDS = 96
    TEN_SECONDS = 240
    MIN_SKELETONS_IN_WIDE_SHOT = 2
    MAX_SKELETONS_IN_WIDE_SHOT = 2

    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    right_half = make_region(RIGHT_HALF_MIN_X, 0.0, 1.0, 1.0)
    left_half = make_region(0.0, 0.0, LEFT_HALF_MAX_X, 1.0)


    graph = {
        'nodes': [ { 'name': 'face', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] } ],
        'edges': []
    }

    vids = VideoIntervalCollection.from_django_qs(
        faces,
        with_payload=in_array(
            bbox_payload_parser(VideoIntervalCollection.django_accessor))
    ).coalesce(payload_merge_op=payload_plus)

    # Get sequences where there's a face on the right half of the screen and
    #   there are at most two faces
    faces_on_right = vids.filter(
        and_pred(
            payload_satisfies(length_at_most(MAX_FACES_ON_SCREEN)),
            payload_satisfies(scene_graph(graph, region=right_half))
        )
    ).dilate(SAMPLING_RATE / 2).coalesce()

    # Get sequences where there's a face on the left half of the screen and
    #   there are at most two faces
    faces_on_left = vids.filter(
        and_pred(
            payload_satisfies(length_at_most(MAX_FACES_ON_SCREEN)),
            payload_satisfies(scene_graph(graph, region=left_half))
        )
    ).dilate(SAMPLING_RATE / 2).coalesce()

    # Sequences where faces on left up to one second before/after faces on left
    # Four seconds of buffer time between left-then-right/right-then-left
    #   segments
    # Only keep remaining sequences that last longer than ten seconds
    shot_reverse_shot = faces_on_right.merge(
        faces_on_left,
        predicate=or_pred(before(max_dist=ONE_SECOND), after(max_dist=ONE_SECOND), arity=2)
    ).dilate(FOUR_SECONDS).coalesce().dilate(-1 * FOUR_SECONDS).filter_length(min_length=TEN_SECONDS)
    
    poses = Pose.objects.filter(frame__shot_scale__name__in=["M","ML","L","XL"]).annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))
    # Only count the poses
    pose_collection = VideoIntervalCollection.from_django_qs(
        poses,
        with_payload=lambda obj: 1
    ).coalesce(payload_merge_op=payload_plus)
    contexts = pose_collection.filter(
        payload_satisfies(lambda count: (count >= MIN_SKELETONS_IN_WIDE_SHOT
            and count <= MAX_SKELETONS_IN_WIDE_SHOT))
    ).dilate(SAMPLING_RATE / 2).coalesce()
    
    shot_countershot_start_wide = contexts.merge(
        shot_reverse_shot, 
        predicate=or_pred(before(max_dist=24), overlaps_before(),arity=2)
    ).dilate(96).coalesce().dilate(-96).filter_length(min_length=240)

    # Post-process to display in Esper widget
    if TIMELINE_OUTPUT:
        results = intrvllists_to_result(shot_countershot_start_wide.get_allintervals())
        add_intrvllists_to_result(results, shot_reverse_shot.get_allintervals(), color='black')
    else:
        results = intrvllists_to_result_with_objects(
            shot_countershot_start_wide.get_allintervals(),
            lambda payload, video: [])
    return results
