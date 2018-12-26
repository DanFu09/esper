from esper.prelude import *
from .queries import query

@query("All face landmarks (rekall)")
def all_face_landmarks():
    from query.models import Face
    from esper.stdlib import face_landmarks_to_dict, simple_result
    import esper.face_landmarks_wrapper as flw
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_with_objects

    STRIDE = 1000
    LIMIT = 100

    # Face landmarks are keyed by Face ID's.
    faces_qs = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    # Use coalesce to get a list of frames we want
    # We store Video ID and frame number in the payload
    frames = VideoIntervalCollection.from_django_qs(
        faces_qs[:LIMIT*STRIDE:STRIDE],
        with_payload=lambda face_obj: (face_obj.video_id, face_obj.min_frame)
    ).coalesce()

    # face_landmarks_wrapper.get takes in a Face queryset or list of Face
    #   objects and returns a list of LandmarksWrapper objects.
    landmarks = frames.map(lambda interval: (
        interval.start, interval.end,
        flw.get(faces_qs.filter(
            video_id=interval.payload[0],
            min_frame=interval.payload[1]).all())
    ))

    # We use face_landmarks_to_dict to draw LandmarksWrapper objects.
    return intrvllists_to_result_with_objects(
        landmarks,
        lambda landmarks_wrappers, video_id: [face_landmarks_to_dict(wrapper) for wrapper in landmarks_wrappers]
    )
