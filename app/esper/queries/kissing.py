from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Kissing (rekall)")
def two_faces_up_close():
    # Takes 2min to run!
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from rekall.bbox_predicates import height_at_least, same_height
    from esper.rekall import intrvllists_to_result_bbox
    
    MIN_FACE_CONFIDENCE = 0.8
    MIN_FACE_HEIGHT = 0.5
    MAX_FACE_HEIGHT_DIFF = 0.1
    MIN_FACE_OVERLAP_X = 0.05
    MIN_FACE_OVERLAP_Y = 0.2

    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.filter(probability__gte=MIN_FACE_CONFIDENCE).annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        height = F('bbox_y2')-F('bbox_y1'),
        video_id=F('frame__video_id')).filter(height__gte=MIN_FACE_HEIGHT)

    faces = VideoIntervalCollection.from_django_qs(
        faces,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor)
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    graph = {
        'nodes': [
            { 'name': 'face_left', 'predicates': [] },
            { 'name': 'face_right', 'predicates': [] },
        ],
        'edges': [
            {'start': 'face_left', 'end':'face_right', 'predicates': [
                lambda f1, f2: f1['x2'] < f2['x2'] and f1['x1']<f2['x1'], # Left face on the left
                lambda f1, f2: f1['x2'] - f2['x1'] > MIN_FACE_OVERLAP_X, # Faces overlap
                lambda f1, f2: min(f1['y2'], f2['y2'])-max(f1['y1'], f1['y1']) > MIN_FACE_OVERLAP_Y,
                lambda f1, f2: f1['y2'] > f2['y1'] and f1['y1'] < f2['y2'],  # No face is entirely above another
                same_height(MAX_FACE_HEIGHT_DIFF),
            ]},
        ]
    }

    faces_up_close = faces.filter(payload_satisfies(
        scene_graph(graph, exact=True)))
    return intrvllists_to_result_bbox(faces_up_close)
