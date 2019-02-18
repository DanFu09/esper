import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, VideoTag, Labeler, Tag
from esper.prelude import Notifier
from esper.kube import make_cluster, cluster_config, worker_config
from esper.scannerutil import ScannerWrapper

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='rgbhists')
LABELED_TAG, _ = Tag.objects.get_or_create(name='rgbhists:labeled')

bad_movie_ids = set([])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
video_ids = sorted(list(all_videos.difference(labeled_videos).difference(bad_movie_ids)))

videos = Video.objects.filter(id__in=video_ids).order_by('id')

cfg = cluster_config(num_workers=100, worker=worker_config('n1-standard-32'))
with make_cluster(cfg, no_delete=True) as db_wrapper:
    db = db_wrapper.db
#if True:
#    db_wrapper = ScannerWrapper.create()
#    db = db_wrapper.db

    histograms = st.histograms.compute_histograms(
        db,
        videos=[video.for_scannertools() for video in list(videos)],
        run_opts = {'work_packet_size': 25, 'io_packet_size': 1000,
            'checkpoint_frequency': 1, 'tasks_in_queue_per_pu': 2}
    )

    # Tag this video as being labeled
    new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
    VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with RGB histograms')
