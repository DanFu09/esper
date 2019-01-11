from esper.prelude import *
from .queries import query

@query("Kissing (rekall)")
def kissing():
    # Takes 7min to run!
    from query.models import Face, Shot
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from rekall.temporal_predicates import overlaps
    from rekall.face_landmark_predicates import looking_left, looking_right
    from rekall.bbox_predicates import height_at_least, same_height
    import esper.face_landmarks_wrapper as flw
    from esper.captions import get_all_segments
    from esper.rekall import intrvllists_to_result_with_objects, bbox_to_result_object
    from esper.stdlib import face_landmarks_to_dict
    
    MAX_MOUTH_DIFF = 0.12
    MIN_FACE_CONFIDENCE = 0.8
    MIN_FACE_HEIGHT = 0.4
    MAX_FACE_HEIGHT_DIFF = 0.1
    MIN_FACE_OVERLAP_X = 0.05
    MIN_FACE_OVERLAP_Y = 0.2
    MAX_FACE_OVERLAP_X_FRACTION = 0.7
    MIN_FACE_ANGLE = 0.1
    
    def map_payload(func):
        def map_fn(intvl):
            intvl.payload = func(intvl.payload)
            return intvl
        return map_fn
    
    def get_landmarks(faces):
        ids = [face['id'] for face in faces]
        landmarks = flw.get(Face.objects.filter(id__in=ids))
        for face, landmark in zip(faces, landmarks):
            face['landmarks'] = landmark
        return faces

    # Annotate face rows with start and end frames and the video ID
    faces_qs = Face.objects.filter(probability__gte=MIN_FACE_CONFIDENCE).annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        height = F('bbox_y2')-F('bbox_y1'),
        video_id=F('frame__video_id')).filter(height__gte=MIN_FACE_HEIGHT)

    faces = VideoIntervalCollection.from_django_qs(
        faces_qs,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, {'id': 'id'})
        ]))
    ).coalesce(payload_merge_op=payload_plus)

    graph = {
        'nodes': [
            { 'name': 'face_left', 'predicates': [] },
            { 'name': 'face_right', 'predicates': [] },
        ],
        'edges': [
            {'start': 'face_left', 'end':'face_right', 'predicates': [
                lambda f1, f2: f1['x2'] < f2['x2'] and f1['x1']<f2['x1'], # Left face on the left
                lambda f1, f2: f1['x2'] - f2['x1'] > MIN_FACE_OVERLAP_X, # Faces overlap
                lambda f1, f2: min(f1['y2'], f2['y2'])-max(f1['y1'], f1['y1']) > MIN_FACE_OVERLAP_Y,
                lambda f1, f2: f1['y2'] > f2['y1'] and f1['y1'] < f2['y2'],  # No face is entirely above another
                same_height(MAX_FACE_HEIGHT_DIFF),
                lambda f1, f2: (f1['x2']-f2['x1'])/max(f1['x2']-f1['x1'], f2['x2']-f2['x1']) < MAX_FACE_OVERLAP_X_FRACTION
            ]},
        ]
    }
    
    def mouths_are_close(lm1, lm2):
        select_outer=[2,3,4,8,9,10]
        select_inner=[1,2,3,5,6,7]
        mouth1 = np.concatenate((lm1.outer_lips()[select_outer], lm1.inner_lips()[select_inner]))
        mouth2 = np.concatenate((lm2.outer_lips()[select_outer], lm2.inner_lips()[select_inner]))
        mean1 = np.mean(mouth1, axis=0)
        mean2 = np.mean(mouth2, axis=0)
        return np.linalg.norm(mean1-mean2) <= MAX_MOUTH_DIFF
    
    # Face is profile if both eyes are on the same side of the nose bridge horizontally.
    def is_left_profile(f):
        lm = f['landmarks']
        nose_x = min(lm.nose_bridge()[:,0])
        left = np.all(lm.left_eye()[:,0] >= nose_x)
        right = np.all(lm.right_eye()[:,0] >= nose_x)
        return left and right
    def is_right_profile(f):
        lm = f['landmarks']
        nose_x = max(lm.nose_bridge()[:,0])
        left = np.all(lm.left_eye()[:,0] <= nose_x)
        right = np.all(lm.right_eye()[:,0] <= nose_x)
        return left and right
    
    # Line is ax+by+c=0
    def project_point_to_line(pt, a, b, c):
        x0,y0=pt[0], pt[1]
        d=a*a+b*b
        x=(b*(b*x0-a*y0)-a*c)/d
        y=(a*(-b*x0+a*y0)-b*c)/d
        return np.array([x,y])
    
    # Positive if facing right
    def signed_face_angle(lm):
        center_line_indices = [27,28,32,33,34,51,62,66,57]
        data = lm.landmarks[center_line_indices]
        fit = np.polyfit(data[:,0], data[:,1], 1)
        # y = ax+b
        a,b = fit[0], fit[1]
        A = project_point_to_line(lm.landmarks[center_line_indices[0]], a,-1,b)
        B = project_point_to_line(lm.landmarks[center_line_indices[-1]], a,-1,b)
        AB = B-A
        AB = AB / np.linalg.norm(AB)
        C = np.mean(lm.nose_bridge()[2:4], axis=0)
        AC = C-A
        AC = AC / np.linalg.norm(AC)
        return np.cross(AB, AC)

        
    graph2 = {
        'nodes': [
            {'name': 'left', 'predicates': [
                lambda f: signed_face_angle(f['landmarks']) > MIN_FACE_ANGLE
#                 is_right_profile
            ]},
            {'name': 'right', 'predicates': [
                lambda f: signed_face_angle(f['landmarks']) < -MIN_FACE_ANGLE
#                 is_left_profile
            ]},
        ],
        'edges': [
            {'start': 'left', 'end':'right', 'predicates':[
                lambda l, r: mouths_are_close(l['landmarks'], r['landmarks']),
            ]}
        ]
    }

    mf_up_close = faces.filter(payload_satisfies(
        scene_graph(graph, exact=True))).map(map_payload(get_landmarks)).filter(
        payload_satisfies(scene_graph(graph2, exact=True)))
    vids = mf_up_close.get_allintervals().keys()
    # Merge with shots
    shots_qs = Shot.objects.filter(
        video_id__in = vids,
        labeler=Labeler.objects.get(name='shot-hsvhist-face')
    ).all()
    total = shots_qs.count()
    print("Total shots:", total)
    # use emtpy list as payload
    shots = VideoIntervalCollection.from_django_qs(
        shots_qs,
        with_payload=lambda row:[],
        progress=True,
        total=total
    )
    kissing_shots = mf_up_close.join(
      shots,
      lambda kiss, shot: [(kiss.get_start(), shot.get_end(), kiss.get_payload())],
      predicate=overlaps(),
      working_window=1
    ).coalesce()
    
    # Getting faces in the shot
    def wrap_in_list(intvl):
        intvl.payload = [intvl.payload]
        return intvl
    
    print("Getting faces...")
    faces_qs2 = Face.objects.filter(frame__video_id__in=vids,probability__gte=MIN_FACE_CONFIDENCE)
    total = faces_qs2.count()
    faces2 = VideoIntervalCollection.from_django_qs(
        faces_qs2.annotate(
            min_frame=F('frame__number'),
            max_frame=F('frame__number'),
            video_id=F('frame__video_id')
        ),
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, {'frame': 'min_frame'})
        ])),
        progress=True,
        total = total
    ).coalesce(payload_merge_op=payload_plus).map(wrap_in_list)
    
    def clip_to_last_frame_with_two_faces(intvl):
        faces = intvl.get_payload()[1]
        two_faces = [(f[0], f[1]) for f in faces if len(f)==2]
        two_high_faces = [(a, b) for a, b in two_faces if min(a['y2']-a['y1'],b['y2']-b['y1'])>=MIN_FACE_HEIGHT]
        frame = [a['frame'] for a,b in two_high_faces]
        
        if len(frame) > 0:
            intvl.end = frame[-1]
        return intvl
    
    clipped_kissing_shots = kissing_shots.merge(
        faces2,
        payload_merge_op = lambda p1, p2: (p1, p2),
        predicate=overlaps(),
        working_window=1
    ).coalesce(payload_merge_op=lambda p1, p2: (p1[0], p1[1]+p2[1])).map(
        clip_to_last_frame_with_two_faces).filter_length(min_length=12)
    
    results = get_all_segments(vids)
    fps_map = dict((i, Video.objects.get(id=i).fps) for i in vids)
    caption_results = VideoIntervalCollection({
        video_id: [(
            word[0] * fps_map[video_id], # start frame
            word[1] * fps_map[video_id], # end frame
            word[2]) # payload is the word
            for word in words]
        for video_id, words in results
    })
    kissing_without_words = clipped_kissing_shots.minus(
            caption_results)
    kissing_final = kissing_without_words.map(
            lambda intvl: (int(intvl.start),
                int(intvl.end), intvl.payload)
            ).coalesce().filter_length(min_length=12)

    def payload_to_objects(p, video_id):
        return [face_landmarks_to_dict(face['landmarks']) for face in p[0]] + [
                    bbox_to_result_object(face, video_id) for face in p[0]]
    
    
    return intrvllists_to_result_with_objects(kissing_final.get_allintervals(),
                lambda p, vid: payload_to_objects(p, vid), stride=1)
