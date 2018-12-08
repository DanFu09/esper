from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Faces from poses (rekall)")
def faces_from_poses_rekall():
    from query.models import Pose
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.merge_ops import payload_plus
    from esper.rekall import intrvllists_to_result_bbox
    from esper.stdlib import qs_to_result
        
    # Annotate pose rows with start and end frames and the video ID
    poses = Pose.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id'))

    # Parse the pose keypoints and get a bounding box around the face
    def get_face_bbox(pose):
        pose_keypoints = pose.pose_keypoints()
        face_indices = [Pose.Nose, Pose.Neck, Pose.REye,
                        Pose.LEye, Pose.REar, Pose.LEar]
        x_vals = [pose_keypoints[index][0] for index in face_indices
                     if pose_keypoints[index][2] is not 0.0]
        y_vals = [pose_keypoints[index][1] for index in face_indices
                     if pose_keypoints[index][2] is not 0.0]
        x1 = min(x_vals)
        y1 = min(y_vals)
        x2 = max(x_vals)
        y2 = max(y_vals)
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }

    # Materialize all the faces and load them into rekall with bounding box payloads
    # Then coalesce them so that all faces in the same frame are in the same interval
    # NOTE that this is slow right now since we're loading all poses!
    vids = VideoIntervalCollection.from_django_qs(poses,
        with_payload=in_array(get_face_bbox)) \
        .coalesce(payload_merge_op=payload_plus)

    # Post-process to display in Esper widget
    return intrvllists_to_result_bbox(vids.get_allintervals(), limit=100, stride=100)
