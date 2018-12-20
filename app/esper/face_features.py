import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, FaceFeatures
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
import json
from tqdm import tqdm

# Faces were computed at this many FPS
FACE_FPS = 2

def frames_to_detect_faces(microshot_boundaries, video):
    # Detect faces FACE_FPS times a second
    sampling_rate = int(round(video.fps) / FACE_FPS)
    frames = set(range(0, video.num_frames, sampling_rate))

    # Detect faces at every microshot boundary
    frames = frames.union(set(microshot_boundaries))

    # Also detect faces the frame before every microshot boundary
    frames = frames.union(set([boundary - 1
        for boundary in microshot_boundaries
        if boundary > 0]))

    return sorted(list(frames))

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='facenet')
LABELED_TAG, _ = Tag.objects.get_or_create(name='facenet:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

# Cluster parameters
cfg = cluster_config(num_workers=80, worker=worker_config('n1-standard-32'),
    pipelines=[st.face_embedding.FaceEmbeddingPipeline])
#with make_cluster(cfg, no_delete=True) as db_wrapper:
#    db = db_wrapper.db
if True:
    db = scannerpy.Database() 
    
    print("Loading histograms from Scanner")

    # Make sure the histograms have been computed already!
    hsv_histograms = st.histograms.compute_hsv_histograms(
        db,
        videos=[video.for_scannertools() for video in list(videos)]
    )

    for idx, hist in enumerate(hsv_histograms):
        if hist is None:
            print(videos[idx].id, 'is None')

    #hsv_histograms_loaded = [hist.load() for hist in tqdm(hsv_histograms)]

    print("Loading microshot boundaries")

    # Compute microshot boundaries
    microshot_boundaries = st.shot_detection.compute_shot_boundaries(
        db,
        videos=[video.for_scannertools() for video in list(videos)],
        histograms=hsv_histograms
    )

    bad_boundaries = []
    for idx, boundaries in enumerate(microshot_boundaries):
        if boundaries is None or boundaries is []:
            bad_boundaries.append(videos[idx].id)
    print("{} movies fail on boundary detection".format(bad_boundaries))

    print("Computing faces")

    # Compute frames FACE_FPS times a second and before and after every microshot
    #   boundary
    frames = [
        frames_to_detect_faces(list(boundaries), video)
        for boundaries, video in zip(microshot_boundaries, videos)
    ]

    print("Loading faces from Scanner")
    # Load faces
    faces = st.face_detection.detect_faces(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames
    )

    print("Computing face embeddings")
    # Compute face embeddings
    features = st.face_embedding.embed_faces(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames,
        bboxes=faces
    )

    print("Putting face embeddings in database")
    # Figure out which frames we've already computed face embeddings for
    frames_in_db_already = set([
        (f.video_id, f.number)
        for f in Frame.objects.filter(tags=LABELED_TAG).all()
    ])

    for video, framelist, facelist, featurelist in tqdm(zip(videos, frames, faces, features), total=len(videos)):
        new_features = []
        for frame, faces_in_frame, features_in_frame in zip(framelist, facelist.load(), featurelist.load()):
            if (video.id, frame) in frames_in_db_already:
                continue
            if len(faces_in_frame) == 0:
                continue
            face_objs = list(Face.objects.filter(frame__video_id=video.id).filter(frame__number=frame).all())
            for bbox, feature_vec in zip(faces_in_frame, features_in_frame):
                face_obj = None
                for obj in face_objs:
                    if (abs(obj.bbox_x1 - bbox.x1) < .000001 and
                        abs(obj.bbox_x2 - bbox.x2) < .000001 and
                        abs(obj.bbox_y1 - bbox.y1) < .000001 and
                        abs(obj.bbox_y2 - bbox.y2) < .000001 and
                        abs(obj.probability - bbox.score) < .000001):
                        face_obj = obj
                        break
                if face_obj is None:
                    print("Couldn't find face {} in {}".format(bbox, face_objs))
                new_features.append(FaceFeatures(
                    face=face_obj,
                    features=json.dumps(feature_vec.tolist()).encode(),
                    labeler=LABELER
                ))
        FaceFeatures.objects.bulk_create(new_features, batch_size=10000)
    

    for video, framelist in tqdm(zip(videos, frames), total=len(videos)):
        new_frame_tags = []
        frame_objs = Frame.objects.filter(video_id=video.id).filter(number__in=framelist)
        for frame in frame_objs:
            if (video.id, frame.number) in frames_in_db_already:
                continue
            new_frame_tags.append(
                    Frame.tags.through(frame_id=frame.pk, tag_id=LABELED_TAG.pk))
        if len(new_frame_tags) == 0:
            continue
        Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=10000)

    # Get the videos that already have the tag
    videos_tagged_already = set([
        vtag.video_id
        for vtag in VideoTag.objects.filter(tag=LABELED_TAG).all()
    ])

    # Tag this video as being labeled
    new_videotags = [
        VideoTag(video=video, tag=LABELED_TAG)
        for video in videos
        if video.id not in videos_tagged_already
    ]
    VideoTag.objects.bulk_create(new_videotags)

    print("Finished putting everything in the database")

Notifier().notify('Done with face embeddings')
