from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("All poses (rekall)")
def all_poses():
    from query.models import PoseMeta
    from esper.stdlib import pose_to_dict, simple_result
    import esper.pose_wrapper as pw
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_with_objects

    STRIDE = 1000
    LIMIT = 100

    # PoseMeta is a table that contains pose ID, labeler, and a pointer to
    #   a Frame.
    # NOTE that PoseMeta ID's have NO RELATION to Pose ID's.
    pose_meta_qs = PoseMeta.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    # Use coalesce to get a list of frames we want
    # We store Video ID and frame number in the payload
    frames = VideoIntervalCollection.from_django_qs(
        pose_meta_qs[:LIMIT*STRIDE:STRIDE],
        with_payload=lambda pose_meta_obj: (pose_meta_obj.video_id, pose_meta_obj.min_frame)
    ).coalesce()

    # pose_wrapper.get takes in a PoseMeta queryset or list of PoseMeta objects
    #   and returns a list of PoseWrapper objects.
    poses = frames.map(lambda interval: (
        interval.start, interval.end,
        pw.get(pose_meta_qs.filter(
            video_id=interval.payload[0],
            min_frame=interval.payload[1]).all())
    ))

    # We use pose_to_dict to draw PoseWrapper objects.
    return intrvllists_to_result_with_objects(
        poses,
        lambda pose_wrappers, video_id: [pose_to_dict(wrapper) for wrapper in pose_wrappers]
    )
