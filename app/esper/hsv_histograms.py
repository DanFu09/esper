import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, VideoTag, Labeler, Tag, Shot
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper
import rekall
from rekall.video_interval_collection import VideoIntervalCollection


# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='hsvhists')
LABELED_TAG, _ = Tag.objects.get_or_create(name='hsvhists:labeled')

bad_movie_ids = set([])

labeled_videos=set()
#labeled_videos = set([videotag.video_id
#        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.filter(ignore_film=False).all()])
video_ids = sorted(list(all_videos.difference(labeled_videos).difference(bad_movie_ids)))

# Load up all manually annotated shots
shots_qs = Shot.objects.filter(labeler__name__contains='manual')
shots = VideoIntervalCollection.from_django_qs(shots_qs)
shot_video_ids = sorted(list(shots.get_allintervals().keys()))

video_ids=sorted(list(set(video_ids).difference(set(shot_video_ids))))

videos = Video.objects.filter(id__in=video_ids).order_by('id')

cfg = cluster_config(num_workers=100, worker=worker_config('n1-standard-32'))
with make_cluster(cfg, no_delete=True) as db_wrapper:
    db = db_wrapper.db
#if True:
#    db_wrapper = ScannerWrapper.create()
#    db = db_wrapper.db

    hsv_histograms = st.histograms.compute_hsv_histograms(
        db,
        videos=[video.for_scannertools() for video in list(videos)],
        run_opts = {'work_packet_size': 4, 'io_packet_size': 2496,
            'checkpoint_frequency': 1, 'tasks_in_queue_per_pu': 2,
            'pipeline_instances_per_node': 2}
    )

    # Tag this video as being labeled
    new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
    VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with HSV histograms')
