from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("All Extreme Close-up frames (rekall)")
def extreme_close_up_frames():
    from esper.shot_scale import ShotScale, get_all_frames_with_shot_scale
    from esper.rekall import intrvllists_to_result_with_objects, bbox_to_result_object

    def pose_payload_to_object(pose, video):
    	return {
          'id': video,
          'type': 'pose',
          'keypoints': pose
    	}
    def payload_to_objects(payload, video_id):
    	result = []
    	result += [bbox_to_result_object(x, video_id) for x in payload.get('face', [])]
    	result += [pose_payload_to_object(x, video_id) for x in payload.get('pose', [])]
    	return result

    video_id=123
    return intrvllists_to_result_with_objects(
        get_all_frames_with_shot_scale(
            video_id, ShotScale.EXTREME_CLOSE_UP).get_allintervals(),
        payload_to_objects, limit=1000, stride=1)
