from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Hermione in the center (rekall)")
def hermione_in_the_center():
    from query.models import FaceCharacterActor
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from rekall.bbox_predicates import height_at_least, left_of, same_value, same_height
    from esper.rekall import intrvllists_to_result_bbox

    STRIDE=10
    LIMIT=100
    MIN_FACE_HEIGHT = 0.12
    EPSILON = 0.15
    NAMES = [ 'ron weasley', 'hermione granger', 'harry potter' ]

    # Annotate face rows with start and end frames and the video ID
    faces_with_character_actor_qs = FaceCharacterActor.objects.annotate(
        min_frame=F('face__frame__number'),
        max_frame=F('face__frame__number'),
        video_id=F('face__frame__video_id'),
        bbox_x1=F('face__bbox_x1'),
        bbox_y1=F('face__bbox_y1'),
        bbox_x2=F('face__bbox_x2'),
        bbox_y2=F('face__bbox_y2'),
        character_name=F('characteractor__character__name')
    ).filter(face__frame__video__name__contains="harry potter")

    faces_with_identity = VideoIntervalCollection.from_django_qs(
        faces_with_character_actor_qs,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'character': 'character_name' }),
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    harry_ron_hermione_scene_graph = {
        'nodes': [
            { 'name': 'face1', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT),
                lambda f: f['character'] == NAMES[0] or f['character'] == NAMES[2]
            ] },
            { 'name': 'face2', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT),
                lambda f: f['character'] == NAMES[1]
            ] },
            { 'name': 'face3', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT),
                lambda f: f['character'] == NAMES[0] or f['character'] == NAMES[2]
            ] }
        ],
        'edges': [
            { 'start': 'face1', 'end': 'face2', 'predicates': [
                lambda f1, f2: f1['x1'] < f2['x1'],
                same_value('y1', epsilon=EPSILON),
                same_height(epsilon=EPSILON) 
            ] },
            { 'start': 'face2', 'end': 'face3', 'predicates': [
                lambda f1, f2: f1['x1'] < f2['x1'],
                same_value('y1', epsilon=EPSILON),
                same_height(epsilon=EPSILON) 
            ] },
            { 'start': 'face1', 'end': 'face3', 'predicates': [
                lambda f1, f2: f1['x1'] < f2['x1'],
                same_value('y1', epsilon=EPSILON),
                same_height(epsilon=EPSILON) 
            ] }
        ]
    }

    harry_ron_hermione = faces_with_identity.filter(payload_satisfies(scene_graph(
        harry_ron_hermione_scene_graph,
        exact=True
    )))
    
    return intrvllists_to_result_bbox(harry_ron_hermione, limit=LIMIT, stride=STRIDE)
