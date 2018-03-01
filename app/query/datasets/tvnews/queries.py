from query.datasets.prelude import *
from query.datasets.queries import *


@query("Non-handlabeled random faces/genders")
def not_handlabeled():
    import random
    l = Labeler.objects.get(name='rudecarnie')
    t = Tag.objects.get(name='handlabeled-face:labeled')
    i = random.randint(0, FaceGender.objects.aggregate(Max('id'))['id__max'])
    return qs_to_result(
        FaceGender.objects.filter(labeler=l, id__gte=i).exclude(
            Q(face__person__frame__tags=t)
            | Q(face__shot__in_commercial=True)
            | Q(face__shot__video__commercials_labeled=False)
            | Q(face__shot__isnull=True)),
        stride=1000)


@query("Handlabeled faces/genders")
def handlabeled():
    return qs_to_result(
        FaceGender.objects.filter(labeler__name='handlabeled-gender').annotate(
            identity=F('face__faceidentity__identity')))


@query("Donald Trump")
def donald_trump():
    return qs_to_result(FaceIdentity.objects.filter(identity__name='donald trump'))


@query("Commercials")
def commercials():
    return qs_to_result(Commercial.objects.filter(labeler__name='haotian-commercials'))


@query("Positive segments")
def positive_segments():
    return qs_to_result(
        Segment.objects.filter(labeler__name='haotian-segments',
                               polarity__isnull=False).order_by('-polarity'))


@query("Negative segments")
def negative_segments():
    return qs_to_result(
        Segment.objects.filter(labeler__name='haotian-segments',
                               polarity__isnull=False).order_by('polarity'))


@query("Segments about Donald Trump")
def segments_about_donald_trump():
    from query.datasets.tvnews.models import ThingType
    return qs_to_result(
        Segment.objects.filter(
            labeler__name='haotian-segments',
            things__type=ThingType.PERSON,
            things__name='donald trump'))


@query("Segments about immigration")
def segments_about_immigration():
    from query.datasets.tvnews.models import ThingType
    return qs_to_result(
        Segment.objects.filter(
            labeler__name='haotian-segments',
            things__type=ThingType.TOPIC,
            things__name='immigration'))


@query("Fox News videos")
def fox_news_videos():
    return qs_to_result(Frame.objects.filter(number=0, video__channel='FOXNEWS'))


@query("Talking heads face tracks")
def talking_heads_tracks():
    return qs_to_result(
        PersonTrack.objects.filter(
            id__in=Person.objects.filter(frame__video__id=791) \
            .annotate(
                c=Subquery(
                    Face.objects.filter(person=OuterRef('pk')) \
                    .annotate(height=F('bbox_y2') - F('bbox_y1')) \
                    .filter(labeler__name='mtcnn', height__gte=0.3) \
                    .values('person') \
                    .annotate(c=Count('*')) \
                    .values('c'),
                    models.IntegerField())) \
            .filter(c__gt=0) \
            .values('tracks')),
        segment=True)


@query("Faces on Poppy Harlow")
def faces_on_poppy_harlow():
    return qs_to_result(
        Face.objects.filter(person__frame__video__show='CNN Newsroom With Poppy Harlow'),
        group=True,
        stride=24)


@query("Female faces on Poppy Harlow")
def female_faces_on_poppy_harlow():
    return qs_to_result(
        Face.objects.filter(
            person__frame__video__show='CNN Newsroom With Poppy Harlow',
            facegender__gender__name='female'),
        group=True,
        stride=24)


@query("Talking heads on Poppy Harlow")
def talking_heads_on_poppy_harlow():
    return qs_to_result(
        Face.objects.annotate(height=F('bbox_y2') - F('bbox_y1')).filter(
            height__gte=0.3,
            person__frame__video__show='CNN Newsroom With Poppy Harlow',
            facegender__gender__name='female'),
        group=True,
        stride=24)


@query("Two female faces on Poppy Harlow")
def two_female_faces_on_poppy_harlow():
    r = []
    try:
        for video in Video.objects.filter(show__name='CNN Newsroom With Poppy Harlow'):
            for frame in Frame.objects.filter(video=video):
                faces = list(
                    Face.objects.annotate(height=F('bbox_y2') - F('bbox_y1')).filter(
                        labeler__name='mtcnn',
                        person__frame=frame,
                        facegender__gender__name='F',
                        height__gte=0.2))
                if len(faces) == 2:
                    r.append({
                        'video': frame.video.id,
                        'start_frame': frame.number,
                        'objects': [bbox_to_dict(f) for f in faces]
                    })
                if len(r) > 100:
                    raise Break()
    except Break:
        pass
    return simple_result(r, 'Frame')


@query("Faces like Poppy Harlow")
def faces_like_poppy_harlow():
    id = 4457280
    FaceFeatures.compute_distances(id)
    return qs_to_result(
        Face.objects.filter(facefeatures__distto__isnull=False).order_by('facefeatures__distto'))


@query("Faces unlike Poppy Harlow")
def faces_unlike_poppy_harlow():
    id = 4457280
    FaceFeatures.compute_distances(id)
    return qs_to_result(
        Face.objects.filter(facefeatures__distto__isnull=False).order_by('-facefeatures__distto'))


@query("MTCNN missed face bboxes vs. handlabeled")
def mtcnn_vs_handlabeled():
    labeler_names = [l['labeler__name'] for l in Face.objects.values('labeler__name').distinct()]

    videos = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for frame in Frame.objects.filter(
            Q(video__show='Situation Room With Wolf Blitzer') | \
            Q(video__show='Special Report With Bret Baier')) \
        .filter(person__face__labeler__name='handlabeled') \
        .select_related('video') \
        .order_by('id')[:50000:5]:
        faces = list(Face.objects.filter(person__frame=frame).select_related('labeler'))
        has_mtcnn = any([f.labeler.name == 'mtcnn' for f in faces])
        has_handlabeled = any([f.labeler.name == 'handlabeled' for f in faces])
        if not has_mtcnn or not has_handlabeled:
            continue
        for face in faces:
            videos[frame.video.id][frame.id][face.labeler.name].append(face)

    AREA_THRESHOLD = 0.02
    DIST_THRESHOLD = 0.10

    mistakes = defaultdict(lambda: defaultdict(tuple))
    for video, frames in videos.iteritems():
        for frame, labelers in frames.iteritems():
            labeler = 'handlabeled'
            faces = labelers[labeler]
            for face in faces:
                if bbox_area(face) < AREA_THRESHOLD:
                    continue

                mistake = True
                for other_labeler in labeler_names:
                    if labeler == other_labeler: continue
                    other_faces = labelers[other_labeler] if other_labeler in labelers else []
                    for other_face in other_faces:
                        if bbox_dist(face, other_face) < DIST_THRESHOLD:
                            mistake = False
                            break

                    if mistake:
                        mistakes[video][frame] = (faces, other_faces)
                        break
                else:
                    continue
                break

    result = []
    for video, frames in list(mistakes.iteritems())[:100]:
        for frame, (faces, other_faces) in frames.iteritems():
            result.append({
                'video': video,
                'start_frame': frame,
                'objects': [bbox_to_dict(f) for f in faces + other_faces]
            })

    return simple_result(result, 'Frame')


@query("MTCNN missed face bboxes vs. OpenPose")
def mtcnn_vs_openpose():
    labeler_names = ['mtcnn', 'openpose']

    videos = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    frames = Frame.objects.all() \
        .annotate(c=Subquery(
            Pose.objects.filter(person__frame=OuterRef('pk')).values('person__frame') \
            .annotate(c=Count('*')).values('c'), models.IntegerField())) \
        .filter(c__gt=0) \
        .select_related('video') \
        .order_by('id')
    for frame in frames[:50000:5]:
        faces = list(Face.objects.filter(person__frame=frame))
        poses = list(Pose.objects.filter(person__frame=frame))
        for face in faces:
            videos[frame.video.id][frame.id]['mtcnn'].append(face)
        for pose in poses:
            videos[frame.video.id][frame.id]['openpose'].append(pose)

    AREA_THRESHOLD = 0.02
    DIST_THRESHOLD = 0.10

    mistakes = defaultdict(lambda: defaultdict(tuple))
    for video, frames in videos.iteritems():
        for frame, labelers in frames.iteritems():
            labeler = 'openpose'
            faces = labelers[labeler]
            for face in faces:
                if bbox_area(face) < AREA_THRESHOLD:
                    continue

                mistake = True
                for other_labeler in labeler_names:
                    if labeler == other_labeler: continue
                    other_faces = labelers[other_labeler] if other_labeler in labelers else []
                    for other_face in other_faces:
                        if bbox_dist(face, other_face) < DIST_THRESHOLD:
                            mistake = False
                            break

                    if mistake and len(other_faces) > 0:
                        mistakes[video][frame] = (faces, other_faces)
                        break
                else:
                    continue
                break

    result = []
    for video, frames in list(mistakes.iteritems())[:100]:
        for frame, (faces, other_faces) in frames.iteritems():
            result.append({
                'video': video,
                'start_frame': frame,
                'objects': [bbox_to_dict(f) for f in other_faces + faces]
            })

    return simple_result(result, 'Frame')


@query("People sitting")
def people_sitting():
    def is_sitting(kp):
        def ang(v):
            return math.atan2(v[1], v[0]) / math.pi * 180

        def is_angled(v):
            v /= np.linalg.norm(v)
            v[1] = -v[1]  # correct for image coordinates
            a = ang(v)
            return a > 0 or a < -140

        return is_angled(kp[Pose.LKnee] - kp[Pose.LHip]) or is_angled(
            kp[Pose.RKnee] - kp[Pose.RHip])

    frames_qs = Frame.objects.filter(video__channel='CNN') \
        .annotate(
            pose_count=Subquery(
                Pose.objects.filter(person__frame=OuterRef('pk')).values('person__frame').annotate(c=Count('*')).values('c')),
            woman_count=Subquery(
                Face.objects.filter(person__frame=OuterRef('pk'), facegender__gender__name='female').values('person__frame').annotate(c=Count('*')).values('c'),
                models.IntegerField())) \
        .filter(pose_count__gt=0, pose_count__lt=6, woman_count__gt=0).order_by('id').select_related('video')

    frames = []
    for frame in frames_qs[:100000:10]:
        filtered = filter_poses(
            'pose',
            is_sitting, [Pose.LAnkle, Pose.LKnee, Pose.RAnkle, Pose.RKnee, Pose.RHip, Pose.LHip],
            poses=Pose.objects.filter(person__frame=frame))

        if len(filtered) > 0:
            frames.append((frame, filtered))

    return simple_result([{
        'video': frame.video.id,
        'start_frame': frame.number,
        'objects': [pose_to_dict(p) for p in poses]
    } for (frame, poses) in frames], 'Frame')


@query("Obama pictures")
def obama_pictures():
    def close(x, y):
        return abs(x - y) < 0.02

    id = 3938394
    FaceFeatures.compute_distances(id)
    sq = Face.objects.filter(
        person__tracks=OuterRef('pk'), labeler__name='mtcnn',
        facefeatures__distto__lte=1.0).values('person__tracks').annotate(c=Count('*'))
    out_tracks = []

    face_tracks = {}  #{t.id: (t, []) for t in tracks}
    for track in \
        PersonTrack.objects.filter(labeler__name='featuretrack') \
        .annotate(
            duration=Track.duration(),
            c=Subquery(sq.values('c'), models.IntegerField())) \
        .filter(duration__gt=0, c__gt=0):

        faces = list(
            Face.objects.filter(person__tracks=track,
                                labeler__name='mtcnn').select_related('person__frame'))
        face_tracks[track.id] = (track, faces)

    for track, faces in face_tracks.values():
        faces.sort(lambda a, b: a.person.frame.number - b.person.frame.number)
        valid = True
        for i in range(len(faces) - 1):
            if not (close(faces[i].bbox_x1, faces[i + 1].bbox_x1)
                    and close(faces[i].bbox_y1, faces[i + 1].bbox_y1)
                    and close(faces[i].bbox_x2, faces[i + 1].bbox_x2)
                    and close(faces[i].bbox_y2, faces[i + 1].bbox_y2)):
                valid = False
                break
        if valid:
            out_tracks.append((track, faces[0]))

    return simple_result([{
        'video': t.video_id,
        'start_frame': Frame.objects.get(video=t.video, number=t.min_frame).id,
        'end_frame': Frame.objects.get(video=t.video, number=t.max_frame).id,
        'objects': [bbox_to_dict(f)]
    } for (t, f) in out_tracks], 'FaceTrack')


@query("Segments about immigration")
def segments_about_immigration():
    tracks = TopicTrack.objects.filter(topic__name='immigration').select_related('video')
    return simple_result([{
        'video':
        track.video.id,
        'start_frame':
        Frame.objects.get(
            number=(track.max_frame + track.min_frame) / 2 / 30 * 30, video=track.video).id,
        'objects': []
    } for track in tracks], 'TopicTrack')


def panels():
    mtcnn = Labeler.objects.get(name='mtcnn')
    face_qs = Face.objects.annotate(height=BoundingBox.height_expr()).filter(
        height__gte=0.25, labeler=mtcnn, shot__in_commercial=False)
    frames = Frame.objects.annotate(c=Subquery(
        face_qs.filter(person__frame=OuterRef('pk')) \
        .values('person__frame') \
        .annotate(c=Count('*')) \
        .values('c'), models.IntegerField())) \
        .filter(c__gte=3, c__lte=3).order_by('id')

    output_frames = []
    for frame in frames[:10000:10]:
        faces = list(face_qs.filter(person__frame=frame))
        y = faces[0].bbox_y1
        valid = True
        for i in range(1, len(faces)):
            if abs(faces[i].bbox_y1 - y) > 0.05:
                valid = False
                break
        if valid:
            output_frames.append((frame, faces))

    return output_frames


@query("Panels")
def panels_():
    from query.datasets.tvnews.queries import panels
    return simple_result([{
        'video': frame.video.id,
        'start_frame': frame.number,
        'objects': [bbox_to_dict(f) for f in faces]
    } for (frame, faces) in panels()], 'Frame')


@query("Animated Rachel Maddow")
def animated_rachel_maddow():
    def pose_dist(p1, p2):
        kp1 = p1.pose_keypoints()
        kp2 = p2.pose_keypoints()

        weights = defaultdict(float, {
            Pose.LWrist: 0.4,
            Pose.RWrist: 0.4,
            Pose.Nose: 0.1,
            Pose.LElbow: 0.05,
            Pose.RElbow: 0.05
        })
        weight_vector = [weights[i] for i in range(Pose.POSE_KEYPOINTS)]

        dist = np.linalg.norm(kp2[:, :2] - kp1[:, :2], axis=1)
        weighted_dist = np.array([
            d * w for d, s1, s2, w in zip(dist, kp1[:, 2], kp2[:, 2], weight_vector)
            if s1 > 0 and s2 > 0
        ])
        return np.linalg.norm(weighted_dist)

    tracks = list(PersonTrack.objects.filter(video__path='tvnews/videos/MSNBC_20100827_060000_The_Rachel_Maddow_Show.mp4') \
        .annotate(c=Subquery(
            Face.objects.filter(person__tracks=OuterRef('pk')) \
            .filter(labeler__name='tinyfaces', facefeatures__distto__isnull=False, facefeatures__distto__lte=1.0) \
            .values('person__tracks')
            .annotate(c=Count('*'))
            .values('c'), models.IntegerField()
            )) \
        .filter(c__gt=0))

    all_dists = []
    for track in tracks:
        poses = list(Pose.objects.filter(person__tracks=track).order_by('person__frame__number'))
        dists = [pose_dist(poses[i], poses[i + 1]) for i in range(len(poses) - 1)]
        all_dists.append((track, np.mean(dists)))
    all_dists.sort(key=itemgetter(1), reverse=True)

    return simple_result([{
        'video':
        t.video.id,
        'track':
        t.id,
        'start_frame':
        Frame.objects.get(video=t.video, number=t.min_frame).id,
        'end_frame':
        Frame.objects.get(video=t.video, number=t.max_frame).id,
        'metadata': [['score', '{:.03f}'.format(score)]],
        'objects':
        [bbox_to_dict(Face.objects.filter(person__frame__number=t.min_frame, person__tracks=t)[0])]
    } for t, score in all_dists], 'PersonTrack')


@query("Audio labels")
def audio_labels():
    gender_seg = json.load(open('/app/gender_seg.json', 'rb'))
    v = Video.objects.get(
        path='tvnews/videos/MSNBCW_20160505_230000_Hardball_With_Chris_Matthews.mp4')
    return {
        'result': [{
            'type':
            'contiguous',
            'label':
            '',
            'elements': [{
                'video': v.id,
                'start_frame': int(d['start'] * v.fps),
                'end_frame': int((d['start'] + d['end']) * v.fps),
                'label': d['gender']
            } for d in sorted(gender_seg, key=itemgetter('start'))]
        }],
        'count':
        0,
        'type':
        '_'
    }
