from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Shot/Reverse Shot Conversations (rekall)")
def shot_reverse_shot():
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies
    from rekall.list_predicates import length_at_most
    from rekall.logical_predicates import and_pred, or_pred
    from rekall.spatial_predicates import scene_graph, make_region
    from rekall.temporal_predicates import before, after
    from rekall.bbox_predicates import height_at_least
    from esper.rekall import intrvllists_to_result, add_intrvllists_to_result
        
    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    right_half = make_region(0.45, 0.0, 1.0, 1.0)
    left_half = make_region(0.0, 0.0, .55, 1.0)

    graph = {
        'nodes': [ { 'name': 'face', 'predicates': [ height_at_least(0.4) ] } ],
        'edges': []
    }

    vids = VideoIntervalCollection.from_django_qs(
        faces,
        with_payload=in_array(
            bbox_payload_parser(VideoIntervalCollection.django_accessor))
    ).coalesce(payload_merge_op=payload_plus)

    faces_on_right = vids.filter(
        and_pred(
            payload_satisfies(length_at_most(2)),
            payload_satisfies(scene_graph(graph, region=right_half))
        )
    ).dilate(6).coalesce()

    faces_on_left = vids.filter(
        and_pred(
            payload_satisfies(length_at_most(2)),
            payload_satisfies(scene_graph(graph, region=left_half))
        )
    ).dilate(6).coalesce()

    shot_reverse_shot = faces_on_right.merge(
        faces_on_left,
        predicate=or_pred(before(max_dist=24), after(max_dist=24), arity=2)
    ).dilate(96).coalesce().dilate(-96).filter_length(min_length=240)

    # Post-process to display in Esper widget
    results = intrvllists_to_result(shot_reverse_shot.get_allintervals())
    add_intrvllists_to_result(results, faces_on_left.get_allintervals(), color='black')
    add_intrvllists_to_result(results, faces_on_right.get_allintervals(), color='green')
    return results
