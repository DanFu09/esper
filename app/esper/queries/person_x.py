from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Frames with actor X (rekall)")
def frames_with_actor_x():
    from query.models import FaceCharacterActor
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from esper.rekall import intrvllists_to_result_bbox

    actor_name = "daniel radcliffe"

    # Annotate face rows with start and end frames and the video ID
    faces_with_character_actor_qs = FaceCharacterActor.objects.annotate(
        min_frame=F('face__frame__number'),
        max_frame=F('face__frame__number'),
        video_id=F('face__frame__video_id'),
        bbox_x1=F('face__bbox_x1'),
        bbox_y1=F('face__bbox_y1'),
        bbox_x2=F('face__bbox_x2'),
        bbox_y2=F('face__bbox_y2'),
        actor_name=F('characteractor__actor__name')
    )

    faces_with_identity = VideoIntervalCollection.from_django_qs(
        faces_with_character_actor_qs,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'actor': 'actor_name' }),
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    faces_with_actor = faces_with_identity.filter(payload_satisfies(scene_graph({
        'nodes': [ { 'name': 'face1', 'predicates': [ lambda f: f['actor'] == actor_name ] } ],
        'edges': []
    })))

    return intrvllists_to_result_bbox(faces_with_actor.get_allintervals(), limit=100, stride=1000)

@query("Frames with character X (rekall)")
def frames_with_character_x():
    from query.models import FaceCharacterActor
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from esper.rekall import intrvllists_to_result_bbox

    character_name = "harry potter"

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
    )

    faces_with_identity = VideoIntervalCollection.from_django_qs(
        faces_with_character_actor_qs,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'character': 'character_name' }),
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    faces_with_actor = faces_with_identity.filter(payload_satisfies(scene_graph({
        'nodes': [ { 'name': 'face1', 'predicates': [ lambda f: f['character'] == character_name ] } ],
        'edges': []
    })))

    return intrvllists_to_result_bbox(faces_with_actor.get_allintervals(), limit=100, stride=1000)
