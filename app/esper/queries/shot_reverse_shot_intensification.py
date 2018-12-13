from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Shot/Reverse Shot Conversations with Intensification (rekall)")
def shot_reverse_shot_intensification():
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, named_payload
    from rekall.merge_ops import payload_plus, merge_named_payload, payload_first
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies, on_name
    from rekall.list_predicates import length_at_most
    from rekall.logical_predicates import and_pred, or_pred
    from rekall.spatial_predicates import scene_graph, make_region
    from rekall.temporal_predicates import before, after
    from rekall.bbox_predicates import height_at_least
    from esper.rekall import intrvllists_to_result, intrvllists_to_result_with_objects, add_intrvllists_to_result
        
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

    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'),
        shot_scale=F('frame__shot_scale'))

    right_half = make_region(RIGHT_HALF_MIN_X, 0.0, 1.0, 1.0)
    left_half = make_region(0.0, 0.0, LEFT_HALF_MAX_X, 1.0)

    graph = {
        'nodes': [ { 'name': 'face', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] } ],
        'edges': []
    }

    vids = VideoIntervalCollection.from_django_qs(
        faces,
        with_payload=merge_dict_parsers([
            named_payload('faces', 
                in_array(
                    bbox_payload_parser(VideoIntervalCollection.django_accessor))
            ),
            named_payload('shot_scale',
                in_array(lambda obj: obj.shot_scale))
        ])
    ).coalesce(payload_merge_op=merge_named_payload({
        'faces': payload_plus,
        'shot_scale': payload_first
    }))
    
    def shot_scales_decreasing(scales):
        if len(scales) <= 1:
            return True
        cur_scale = scales[0]
        for scale in scales:
            if cur_scale == 0:
                cur_scale = scale
                continue
            if scale == 0:
                continue
            if scale > cur_scale:
                # Shot scale has gotten father here
                return False
        return True

    # Get sequences where there's a face on the right half of the screen and
    #   there are at most two faces
    # Payload is the faces in the first frame, and a list of the shot scales
    #   throughout the sequence
    # Filter out any sequences where the shot scale gets farther away over the sequence
    faces_on_right = vids.filter(
        and_pred(
            payload_satisfies(on_name('faces', length_at_most(MAX_FACES_ON_SCREEN))),
            payload_satisfies(on_name('faces', scene_graph(graph, region=right_half)))
        )
    ).dilate(SAMPLING_RATE / 2).coalesce(
        payload_merge_op=merge_named_payload({
            'faces': payload_first,
            'shot_scale': payload_plus
        })
    ).filter(lambda intrvl: shot_scales_decreasing(intrvl.get_payload()['shot_scale']))

    # Get sequences where there's a face on the left half of the screen and
    #   there are at most two faces
    # Payload is the faces in the first frame, and a list of the shot scales
    #   throughout the sequence
    faces_on_left = vids.filter(
        and_pred(
            payload_satisfies(on_name('faces', length_at_most(MAX_FACES_ON_SCREEN))),
            payload_satisfies(on_name('faces', scene_graph(graph, region=left_half)))
        )
    ).dilate(SAMPLING_RATE / 2).coalesce(
        payload_merge_op=merge_named_payload({
            'faces': payload_first,
            'shot_scale': payload_plus})
    ).filter(lambda intrvl: shot_scales_decreasing(intrvl.get_payload()['shot_scale']))

    # Sequences where faces on left up to one second before/after faces on left
    # Four seconds of buffer time between left-then-right/right-then-left
    #   segments
    # Filter sequences by decreasing shot sequences
    # Only keep remaining sequences that last longer than ten seconds
    shot_reverse_shot_intensification = faces_on_right.merge(
        faces_on_left,
        predicate=before(max_dist=ONE_SECOND)
    ).set_union(faces_on_left.merge(
        faces_on_right,
        predicate=before(max_dist=ONE_SECOND)
    )).dilate(FOUR_SECONDS).coalesce(payload_merge_op=merge_named_payload({
        'faces': payload_first,
        'shot_scale': payload_plus
    })).dilate(-1 * FOUR_SECONDS).filter(
        lambda intrvl: shot_scales_decreasing(intrvl.get_payload()['shot_scale']) 
    ).filter_length(min_length=TEN_SECONDS)

    def non_uniform(shot_scales):
        return (len(set(shot_scales)) > 2 if 0 in set(shot_scales) else
            len(set(shot_scales)) > 1)

    # Finally, filter out any shot sequences where the shot scales are uniform
    shot_reverse_shot_intensification = shot_reverse_shot_intensification.filter(
        lambda intrvl: non_uniform(intrvl.get_payload()['shot_scale']) 
    )

    # Post-process to display in Esper widget
    if TIMELINE_OUTPUT:
        results = intrvllists_to_result(shot_reverse_shot_intensification.get_allintervals())
        add_intrvllists_to_result(results, faces_on_left.get_allintervals(), color='black')
        add_intrvllists_to_result(results, faces_on_right.get_allintervals(), color='green')
    else:
        results = intrvllists_to_result_with_objects(
            shot_reverse_shot_intensification.get_allintervals(), lambda payload, video: [])
    return results
