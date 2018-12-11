from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Hero shot (rekall)")
def hero_shot():
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import named_payload, in_array, bbox_payload_parser
    from rekall.parsers import merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus, payload_first, merge_named_payload
    from rekall.payload_predicates import payload_satisfies, on_name
    from rekall.spatial_predicates import scene_graph
    from rekall.logical_predicates import and_pred
    from rekall.bbox_predicates import height_at_least, left_of, same_value
    from esper.rekall import intrvllists_to_result_with_objects, bbox_to_result_object
   
    # We're going to look for frames that would be good "hero shot" frames --
    #   potentially good frames to show in a Netflix preview, for instance.
    # We're going to look for frames where there's exactly one face of a
    #   certain height, and the frame has certain minimum brightness,
    #   sharpness, and contrast properties.
    MIN_FACE_HEIGHT = 0.2
    MIN_BRIGHTNESS = 50
    MIN_SHARPNESS = 50
    MIN_CONTRAST = 30
    FILM_NAME = "star wars the force awakens"

    # Annotate face rows with start and end frames, video ID, and frame image
    #   information
    faces_qs = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'),
        brightness=F('frame__brightness'),
        contrast=F('frame__contrast'),
        sharpness=F('frame__sharpness')
    ).filter(frame__video__name=FILM_NAME)

    # Load bounding boxes and faces into rekall, and put all faces in one frame
    faces = VideoIntervalCollection.from_django_qs(
        faces_qs,
        with_payload=merge_dict_parsers([named_payload('face',
            in_array(bbox_payload_parser(VideoIntervalCollection.django_accessor))),
            dict_payload_parser(VideoIntervalCollection.django_accessor, {
                'brightness': 'brightness',
                'contrast': 'contrast',
                'sharpness': 'sharpness'
            })])
    ).coalesce(merge_named_payload({
        'face': payload_plus,
        'brightness': payload_first,
        'contrast': payload_first,
        'sharpness': payload_first
    }))

    # Hero shots are shots where there is exactly one face of at least a
    #   certain height, and brightness, contrast, and sharpness are at least
    #   some amount
    hero_shots = faces.filter(payload_satisfies(and_pred(
        on_name('face', scene_graph({
            'nodes': [{ 'name': 'face', 'predicates': [
                height_at_least(MIN_FACE_HEIGHT) ] }],
            'edges': []
        }, exact=True)),
        lambda payload: (payload['brightness'] > MIN_BRIGHTNESS and
            payload['contrast'] > MIN_CONTRAST and
            payload['sharpness'] > MIN_SHARPNESS)
    )))
    
    return intrvllists_to_result_with_objects(hero_shots.get_allintervals(), 
        lambda payload, video_id: [
            bbox_to_result_object(bbox, video_id) for bbox in payload['face']],
        limit=100, stride=10)
