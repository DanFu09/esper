from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Frames with two women (rekall)")
def frames_with_two_women():
    from query.models import FaceGender
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    
    MIN_FACE_CONFIDENCE = 0.95
    MIN_GENDER_CONFIDENCE = 0.95
    VIDEO_NAME_CONTAINS = "harry potter"

    # Annotate face rows with start and end frames and the video ID
    faces_with_gender= FaceGender.objects.annotate(
        min_frame=F('face__frame__number'),
        max_frame=F('face__frame__number'),
        video_id=F('face__frame__video_id'),
        bbox_x1=F('face__bbox_x1'),
        bbox_y1=F('face__bbox_y1'),
        bbox_x2=F('face__bbox_x2'),
        bbox_y2=F('face__bbox_y2'),
        gender_name=F('gender__name'),
        face_probability=F('face__probability')
    ).filter(face__frame__video__name__contains=VIDEO_NAME_CONTAINS)

    faces = VideoIntervalCollection.from_django_qs(
        faces_with_gender,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'gender': 'gender_name' }),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'gender_probability': 'probability' }),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'face_probability': 'face_probability' })
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    graph = {
        'nodes': [
            { 'name': 'face1', 'predicates': [
                lambda payload: payload['gender'] is 'F',
                lambda payload: payload['face_probability'] > MIN_FACE_CONFIDENCE,
                lambda payload: payload['gender_probability'] > MIN_GENDER_CONFIDENCE
                ] },
            { 'name': 'face2', 'predicates': [
                lambda payload: payload['gender'] is 'F',
                lambda payload: payload['face_probability'] > MIN_FACE_CONFIDENCE,
                lambda payload: payload['gender_probability'] > MIN_GENDER_CONFIDENCE
                ] },
        ],
        'edges': []
    }

    two_women = faces.filter(payload_satisfies(
        scene_graph(graph, exact=False)))

    return intrvllists_to_result_bbox(two_women.get_allintervals(), limit=100, stride=10)
