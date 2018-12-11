from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Three people (rekall)")
def three_people():
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from rekall.bbox_predicates import height_at_least, left_of, same_value
        
    MIN_FACE_HEIGHT = 0.3
    EPSILON = 0.05

    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    # Materialize all the faces and load them into rekall with bounding box payloads
    # Then coalesce them so that all faces in the same frame are in the same interval
    # NOTE that this is slow right now since we're loading all faces!
    face_lists = VideoIntervalCollection.from_django_qs(
        faces,
        with_payload=in_array(
            bbox_payload_parser(VideoIntervalCollection.django_accessor))
        ).coalesce(payload_merge_op=payload_plus)

    three_people_scene_graph = {
        'nodes': [
            { 'name': 'face1', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] },
            { 'name': 'face2', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] },
            { 'name': 'face3', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] }
        ],
        'edges': [
            { 'start': 'face1', 'end': 'face2', 'predicates': [ left_of(), same_value('y1', epsilon=EPSILON) ] },
            { 'start': 'face2', 'end': 'face3', 'predicates': [ left_of(), same_value('y1', epsilon=EPSILON) ] }
        ]
    }

    three_people = face_lists.filter(payload_satisfies(scene_graph(
        three_people_scene_graph, exact=True
    )))

    # Post-process to display in Esper widget
    return intrvllists_to_result_bbox(three_people.get_allintervals(), limit=100, stride=100)
