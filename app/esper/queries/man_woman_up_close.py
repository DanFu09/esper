from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Man and woman up close (rekall)")
def man_woman_up_close():
    from query.models import FaceGender
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from rekall.bbox_predicates import height_at_least
    
    MIN_FACE_CONFIDENCE = 0.95
    MIN_GENDER_CONFIDENCE = 0.95
    MIN_FACE_HEIGHT = 0.6

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
        face_probability=F('face__probability'))

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
            { 'name': 'face_male', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT),
                lambda payload: payload['gender'] is 'M',
                lambda payload: payload['face_probability'] > MIN_FACE_CONFIDENCE,
                lambda payload: payload['gender_probability'] > MIN_GENDER_CONFIDENCE
                ] },
            { 'name': 'face_female', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT),
                lambda payload: payload['gender'] is 'F',
                lambda payload: payload['face_probability'] > MIN_FACE_CONFIDENCE,
                lambda payload: payload['gender_probability'] > MIN_GENDER_CONFIDENCE
                ] },
        ],
        'edges': []
    }

    mf_up_close = faces.filter(payload_satisfies(
        scene_graph(graph, exact=True)))

    return intrvllists_to_result_bbox(mf_up_close.get_allintervals(), limit=100, stride=100)
