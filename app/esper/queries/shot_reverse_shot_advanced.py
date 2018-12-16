from esper.prelude import *
from esper.rekall import *
from .queries import query

@query("Shot/Reverse Shot Conversations Complex (rekall)")
def shot_reverse_shot_complex():
    from query.models import Face, Shot
    from rekall.temporal_predicates import overlaps
    from rekall.merge_ops import payload_second, payload_plus
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.interval_list import Interval, IntervalList
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.payload_predicates import payload_satisfies
    from rekall.list_predicates import length_at_most
    from rekall.logical_predicates import and_pred
    from rekall.spatial_predicates import scene_graph, make_region
    from rekall.temporal_predicates import before, after
    from rekall.bbox_predicates import height_at_least
    from esper.rekall import intrvllists_to_result_with_objects
            
    VIDEO_NAME='godfather part iii'

    MAX_FACE_MOVEMENT=0.15
    MIN_FACE_HEIGHT=0.2
    MAX_FACES_ON_SCREEN=4
    RIGHT_HALF_MIN_X=0.33
    LEFT_HALF_MAX_X=0.66
    SHOTS_LABELER_ID=64
    # faces are sampled every 12 frames
    SAMPLING_RATE = 12
    # Annotate face rows with start and end frames and the video ID
    faces = Face.objects.annotate(
            min_frame=F('frame__number'),
            max_frame=F('frame__number'),
            video_id=F('frame__video_id')).filter(frame__video__name__contains=VIDEO_NAME)
    
    shots = VideoIntervalCollection.from_django_qs(
        Shot.objects.filter(video__name__contains=VIDEO_NAME, labeler_id=SHOTS_LABELER_ID),
        with_payload=lambda obj:[]
    )
    # vids are all faces for each frame
    vids = VideoIntervalCollection.from_django_qs(
            faces.filter(probability__gte=0.99),
            with_payload=in_array(
                bbox_payload_parser(VideoIntervalCollection.django_accessor))
        ).coalesce(payload_merge_op=payload_plus)
    
    right_half = make_region(RIGHT_HALF_MIN_X, 0.0, 1.0, 1.0)
    left_half = make_region(0.0, 0.0, LEFT_HALF_MAX_X, 1.0)
    graph = {
            'nodes': [ { 'name': 'face', 'predicates': [ height_at_least(MIN_FACE_HEIGHT) ] } ],
            'edges': []
        }
    
    
    faces_on_right = vids.filter(
            and_pred(
                payload_satisfies(length_at_most(MAX_FACES_ON_SCREEN)),
                payload_satisfies(scene_graph(graph, region=right_half))
            )
        )
    faces_on_left = vids.filter(
            and_pred(
                payload_satisfies(length_at_most(MAX_FACES_ON_SCREEN)),
                payload_satisfies(scene_graph(graph, region=left_half))
            )
        )
    
    def wrap_list(intvl):
        intvl.payload = [intvl.payload]
        return intvl
    
    def get_height(box):
        return box['y2'] - box['y1']
    
    def get_center(box):
        return ((box['x1'] + box['x2']) / 2, (box['y1']+box['y2']) / 2)
    
    def get_distance(pt1, pt2):
        return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    
    def find_highest_box(boxes):
        if len(boxes) == 0:
            return None
        result = boxes[0]
        best = get_height(result)
        for i in range(1, len(boxes)):
            h = get_height(boxes[i])
            if h > best:
                best = h
                result= boxes[i]
        return result
    
    def take_highest_in_frame(intvl):
        result = []
        for faces_in_frame in intvl.payload:
            largest = find_highest_box(faces_in_frame)
            if largest is not None:
                result.append(largest)
        intvl.payload = result
        return intvl        
    
    # Check if displacement of box center between frames are within `dist`
    def inter_frame_movement_less_than(dist):
        def check(boxes):
            for b1, b2 in zip(boxes, boxes[1:]):
                if get_distance(get_center(b1), get_center(b2)) > dist:
                    return False
            return True
        return check
        
    # Payload is a list, each element is a list of faces for a frame
    shots_with_face_on_right = shots.merge(
        faces_on_right, predicate=overlaps(), payload_merge_op=payload_second
        ).map(wrap_list).coalesce(payload_merge_op=payload_plus).map(take_highest_in_frame).filter(
        payload_satisfies(inter_frame_movement_less_than(MAX_FACE_MOVEMENT)))
    shots_with_face_on_left = shots.merge(
        faces_on_left, predicate=overlaps(), payload_merge_op=payload_second
        ).map(wrap_list).coalesce(payload_merge_op=payload_plus).map(take_highest_in_frame).filter(
        payload_satisfies(inter_frame_movement_less_than(MAX_FACE_MOVEMENT)))
    
    # Right-Left-Right sequences
    shot_reverse_shot_1 = shots_with_face_on_right.merge(
            shots_with_face_on_left,
            predicate=before(max_dist=1)
        ).merge(
            shots_with_face_on_right,
            predicate=before(max_dist=1)
        )
    
    # Left-Right-Left sequences
    shot_reverse_shot_2 = shots_with_face_on_left.merge(
            shots_with_face_on_right,
            predicate=before(max_dist=1)
        ).merge(
            shots_with_face_on_left,
            predicate=before(max_dist=1)
        )
    
    shot_reverse_shot = shot_reverse_shot_1.set_union(shot_reverse_shot_2).coalesce()
    result = intrvllists_to_result_with_objects(shot_reverse_shot.get_allintervals(), payload_to_objs=lambda p,v:[])
    return result
