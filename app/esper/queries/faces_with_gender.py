from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Faces with gender (rekall)")
def faces_with_gender():
    from query.models import FaceGender
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    
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
        gender_name=F('gender__name')
    ).filter(face__frame__video__name__contains=VIDEO_NAME_CONTAINS)

    faces = VideoIntervalCollection.from_django_qs(
        faces_with_gender,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'gender': 'gender_name' })
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    return intrvllists_to_result_bbox(faces.get_allintervals(), limit=100, stride=1000)
