import os
from django.db.models import Q
from django.db import transaction
from query.models import Video, Face, Labeler, Tag, VideoTag, FaceGender, Gender
from esper.prelude import Notifier
import esper.face_embeddings as face_embeddings
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import multiprocessing
from multiprocessing import Pool

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='gender-knn')
LABELED_TAG, _ = Tag.objects.get_or_create(name='gender-knn:labeled')

HANDLABELER_NAME = 'handlabeled-gender-validation'
NUM_NEIGHBORS = 7

global MODEL_NAME
MODEL_NAME = 'knn_gender_model.sav'

hand_face_genders = {
    fg['face__id']: fg['gender__id']
    for fg in FaceGender.objects.filter(
        labeler=Labeler.objects.get(name=HANDLABELER_NAME)
    ).values('face__id', 'gender__id')
}
gender_id_dict = {g.name: g.id for g in Gender.objects.all()}

training_face_ids = list(hand_face_genders.keys())
training_features = face_embeddings.features(training_face_ids)
training_ground_truth = [
    1 if hand_face_genders[fid] == gender_id_dict['M'] else 0
    for fid in training_face_ids
]

neigh = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS)
neigh.fit(training_features, training_ground_truth)
joblib.dump(neigh, MODEL_NAME)

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))
#video_ids = sorted(list([
#    video.id
#    for video in Video.objects.filter(small_dataset=True).all()
#]))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

def predict(features):
    classifier = joblib.load(MODEL_NAME)
    return classifier.predict_proba(features)

# Set up multiprocessing
n_threads = multiprocessing.cpu_count()
pool = Pool(n_threads)

# Get the videos that already have the tag
videos_tagged_already = set([
    vtag.video_id
    for vtag in VideoTag.objects.filter(tag=LABELED_TAG).all()
])

for video in tqdm(videos, total=len(video_ids)):
    faces = Face.objects.filter(frame__video_id=video.id)

    ids_to_compute = [ f.id for f in faces ]

    faces_computed_already = [
        fg.face_id
        for fg in FaceGender.objects.filter(
            face_id__in=ids_to_compute, labeler=LABELER
        )
    ]

    new_facegenders = []
    facegenders_to_update = []
    
    face_features = face_embeddings.features(ids_to_compute)
    
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[ i * k + min(i, m):(i + 1) * k + min(i + 1, m) ] for i in range(n))

    payloads = split(face_features, n_threads)
    predictions = [ pred for l in pool.map(predict, payloads) for pred in l ]
    
    for fid, scores in zip(ids_to_compute, predictions):
        gender_id = gender_id_dict['M' if scores[1] >= 0.5 else 'F']
        score = scores[1] if scores[1] >= 0.5 else scores[0]
        if fid not in faces_computed_already:
            new_facegenders.append(FaceGender(
                face_id=fid,
                gender_id=gender_id,
                probability=score,
                labeler=LABELER
            ))
        else:
            fg = FaceGender.get(face_id=fid, labeler=LABELER)
            if fg.gender_id != gender_id or fg.probability != score:
                fg.gender_id = gender_id
                fg.probability = score
                facegenders_to_update.append(fg)

    with transaction.atomic():
        FaceGender.objects.bulk_create(new_facegenders)
        for fg in facegenders_to_update:
            fg.save()

        # Tag this video as being labeled
        if video.id not in videos_tagged_already:
            VideoTag(video=video, tag=LABELED_TAG).save()

Notifier().notify('Done with gender detection')
