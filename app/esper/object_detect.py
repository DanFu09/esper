import scannerpy 
from scannerpy.stdlib.util import download_temp_file
import scannertools as st
import os
from django.db.models import Q
from django.db import transaction
from query.models import Video, Frame, Face, Labeler, Tag, VideoTag, ObjectLabel, Object
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
from tqdm import tqdm
import tempfile

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='ssd-mobilenet')
LABELED_TAG, _ = Tag.objects.get_or_create(name='ssd-mobilenet:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))[497:]
#video_ids = sorted(list([
#    video.id
#    for video in Video.objects.filter(small_dataset=True).all()
#]))

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

print("Getting frames to compute on")
# Get frames that we computed faces on
frames = [
    [
        f.number
        for f in Frame.objects.filter(video_id=video,
            tags__name='face_computed').order_by('number')
    ]
    for video in tqdm(video_ids)
]

# Cluster parameters
cfg = cluster_config(num_workers=200, worker=worker_config('n1-standard-32'),
    pipelines=[st.object_detection.ObjectDetectionPipeline])
#with make_cluster(cfg, no_delete=True) as db_wrapper:
#    db = db_wrapper.db
if True:
    db = scannerpy.Database() 

    print("Detecting objects")
    objects = st.object_detection.detect_objects(
        db,
        videos=[video.for_scannertools() for video in videos],
        frames=frames
    )

    print("Downloading label definitions")
    LABEL_URL = 'https://storage.googleapis.com/scanner-data/public/mscoco_label_map.pbtxt'
    label_path = download_temp_file(LABEL_URL)
    labels_to_names = {}
    with open(label_path) as f:
        cur_id = None
        for line in f.readlines():
            line = line.strip()
            if len(line.strip()) > 0:
                if line[:len('id')] == 'id':
                    cur_id = int(line.split(' ')[1])
                if line[:len('display_name')] == 'display_name':
                    display_name = line.split('"')[1]
                    labels_to_names[cur_id] = display_name
    print(label_path, labels_to_names)


    print("Putting objects into the database")
    for objectlist, framelist, video in tqdm(zip(objects, frames, videos), total=len(videos)):
        # Get the frames in this video that have already been labeled
        frames_labeled_already = set([
            face.frame_id
            for face in Face.objects.filter(frame__video_id=video.id)
        ])

        # Next get all the Frame objects out of the database
        frame_objs = Frame.objects.filter(number__in=framelist, video_id=video.id).order_by('number').all()

        new_objects = []
        new_frame_tags = []
        object_labels = {}

        for bbox_list, frame, frame_obj in zip(objectlist.load(), framelist, frame_objs):
            #if frame_obj.id in frames_labeled_already:
            #    continue
            for bbox in bbox_list:
                display_name = labels_to_names[bbox.label]
                if bbox.score < .95 or display_name == 'person':
                    continue
                if display_name in object_labels:
                    object_label = object_labels[display_name]
                else:
                    object_label, _ = ObjectLabel.objects.get_or_create(name=display_name)
                    object_labels[display_name] = object_label

                new_objects.append(Object(
                    frame=frame_obj,
                    bbox_x1=bbox.x1,
                    bbox_x2=bbox.x2,
                    bbox_y1=bbox.y1,
                    bbox_y2=bbox.y2,
                    probability=bbox.score,
                    labeler=LABELER,
                    label=object_label
                ))
            #new_frame_tags.append(Frame.tags.through(
            #    frame_id=frame_obj.pk,
            #    tag=LABELED_TAG
            #))

        with transaction.atomic():
            #Frame.tags.through.objects.bulk_create(new_frame_tags, batch_size=10000)
            Object.objects.bulk_create(new_objects, batch_size=10000)
            VideoTag.objects.get_or_create(video=video, tag=LABELED_TAG)

    print("Done putting everything into the database!")
