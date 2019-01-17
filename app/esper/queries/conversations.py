from esper.prelude import *
from .queries import query

@query("Conversations")
def conversations_for_display():
    from query.models import FaceCharacterActor, Shot
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser, merge_dict_parsers, dict_payload_parser
    from rekall.merge_ops import payload_plus
    from rekall.payload_predicates import payload_satisfies
    from rekall.spatial_predicates import scene_graph
    from esper.rekall import intrvllists_to_result_bbox
    from query.models import Face
    from rekall.video_interval_collection import VideoIntervalCollection
    from rekall.parsers import in_array, bbox_payload_parser
    from rekall.merge_ops import payload_plus, merge_named_payload, payload_second
    from esper.rekall import intrvllists_to_result_bbox
    from rekall.payload_predicates import payload_satisfies
    from rekall.list_predicates import length_at_most
    from rekall.logical_predicates import and_pred, or_pred, true_pred
    from rekall.spatial_predicates import scene_graph, make_region
    from rekall.temporal_predicates import before, after, overlaps, equal
    from rekall.bbox_predicates import height_at_least
    from esper.rekall import intrvllists_to_result, intrvllists_to_result_with_objects, add_intrvllists_to_result
    from esper.prelude import esper_widget
    from rekall.interval_list import Interval, IntervalList
    import esper.face_embeddings as face_embeddings
    
    video_id=15
    EMBEDDING_EQUALITY_THRESHOLD = 1.
    ONE_FRAME = 1
    
    faces_qs = Face.objects.annotate(
        min_frame=F('frame__number'),
        max_frame=F('frame__number'),
        video_id=F('frame__video_id')
    ).filter(frame__video_id=video_id, frame__regularly_sampled=True)
    
    faces_per_frame = VideoIntervalCollection.from_django_qs(
        faces_qs,
        with_payload=in_array(merge_dict_parsers([
            bbox_payload_parser(VideoIntervalCollection.django_accessor),
            dict_payload_parser(VideoIntervalCollection.django_accessor, { 'face_id': 'id' }),
        ]))
    ).coalesce(payload_merge_op=payload_plus)
    
    shots_qs = Shot.objects.filter(cinematic=True)
    shots = VideoIntervalCollection.from_django_qs(shots_qs)
    
    shots_with_faces = shots.merge(
        faces_per_frame, 
        predicate=overlaps(), 
        payload_merge_op=lambda shot_id, faces_in_frame: (shot_id, [faces_in_frame])
    ).coalesce(payload_merge_op=lambda p1, p2: (p1[0], p1[1] + p2[1]))
   
    def cluster_center(face_ids):
#         print("About to compute mean")
        mean_embedding = face_embeddings.mean(face_ids)
#         print("About to compute dist", face_ids)
        dists = face_embeddings.dist(face_ids, [mean_embedding])
#         print("Done computing dist")
        return min(zip(dists, face_ids))[1]

    def cluster_and_compute_centers(faces_in_frame_list, shot_id):
        num_people = max(len(faces_in_frame) for faces_in_frame in faces_in_frame_list)
        face_ids = [face['face_id'] for faces_in_frame in faces_in_frame_list for face in faces_in_frame]
        face_heights = [face['y2'] - face['y1']
                        for faces_in_frame in faces_in_frame_list for face in faces_in_frame]
        print(num_people)
        if num_people == 1:
            clusters = [(fid, 0) for fid in face_ids]
        else:
            clusters = face_embeddings.kmeans(face_ids, num_people)
#         print("Done clustering")
        centers = [
            (
                cluster_center([
                    face_id
                    for face_id, cluster_id in clusters
                    if cluster_id == i
                ]), [
                    face_id
                    for face_id, cluster_id in clusters
                    if cluster_id == i
                ],
                shot_id,
                max([
                    face_heights[face_ids.index(face_id)]
                    for face_id, cluster_id in clusters if cluster_id == i
                ])
            )
            for i in range(num_people)
        ]
#         print("Done computing the center")
        return centers

#     print("About to compute clusters")
    
    shots_with_centers = shots_with_faces.map(
        lambda intrvl: (intrvl.start, intrvl.end, 
                        (intrvl.payload[0],
                         cluster_and_compute_centers(intrvl.payload[1], intrvl.payload[0]))
                       )
    )
    
#     print("Clusters computed")
    
    def same_face(center1, center2):
        return face_embeddings.dist([center1], target_ids=[center2])[0] < EMBEDDING_EQUALITY_THRESHOLD

    def cross_product_faces(intrvl1, intrvl2):
        payload1 = intrvl1.get_payload()
        payload2 = intrvl2.get_payload()
        payload = []
        for cluster1 in payload1[1]:
            for cluster2 in payload2[1]:
                if not same_face(cluster1[0], cluster2[0]):
                    new_payload = {'A': cluster1, 'B': cluster2}
                    payload.append(new_payload)

        return [(min(intrvl1.get_start(), intrvl2.get_start()),
                 max(intrvl1.get_end(), intrvl2.get_end()), {
            'chrs': payload,
            'shots': [payload1[0], payload2[0]]
        })]
    
    two_shots = shots_with_centers.join(
        shots_with_centers,
        predicate=after(max_dist=ONE_FRAME, min_dist=ONE_FRAME), 
        merge_op=cross_product_faces
    )
 
#     print("Cross product done")

    def faces_equal(payload1, payload2):
        for face_pair1 in payload1['chrs']:
            for face_pair2 in payload2['chrs']:
                if (same_face(face_pair1['A'][0], face_pair2['A'][0]) and
                    same_face(face_pair1['B'][0], face_pair2['B'][0])):
                    return True
                if (same_face(face_pair1['A'][0], face_pair2['B'][0]) and
                    same_face(face_pair1['B'][0], face_pair2['A'][0])):
                    return True
        return False
    
    convs = two_shots.coalesce(
        predicate=payload_satisfies(faces_equal, arity=2),
        payload_merge_op = lambda payload1, payload2: {
            'chrs': payload1['chrs'] + payload2['chrs'],
            'shots': payload1['shots'] + payload2['shots']
        }
    )
    
#     print("Coalesce done")    
        
    adjacent_seq = convs.merge(
        convs,
        predicate=and_pred(
            after(max_dist=ONE_FRAME, min_dist=ONE_FRAME),
            payload_satisfies(faces_equal, arity=2),
            arity=2),
        payload_merge_op = lambda payload1, payload2: {
            'chrs': payload1['chrs'] + payload2['chrs'],
            'shots': payload1['shots'] + payload2['shots']
        },
        working_window=1
    )
    convs = convs.set_union(adjacent_seq)
    # convs = convs.coalesce(predicate=times_equal, payload_merge_op=shots_equal)
    
#     print("Two-shot adjacencies done")

    def filter_fn(intvl):
        payload = intvl.get_payload()
        if type(payload) is dict and 'shots' in payload:
            return len(set(payload['shots'])) >= 3
        return False 
    
    convs = convs.filter(filter_fn)
    convs = convs.coalesce()
    
#     print("Final filter done")

#     for video_id in convs.intervals.keys():
#         print(video_id)
#         intvllist = convs.get_intervallist(video_id)
#         for intvl in intvllist.get_intervals():
#             print(intvl.payload)
#             print(str(intvl.start) + ':' + str(intvl.end))
    
    return convs