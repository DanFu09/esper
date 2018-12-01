import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, VideoTag, Labeler, Tag
from esper.prelude import Notifier

video_ids = [vals[0] for vals in
        VideoTag.objects.values_list('video_id').distinct().all()]
videos = Video.objects.filter(id__in=video_ids)

db = scannerpy.Database()
hsv_histograms = st.histograms.compute_hsv_histograms(
    db,
    videos=[video.for_scannertools() for video in videos]
)

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='hsvhists')
LABELED_TAG, _ = Tag.objects.get_or_create(name='hsvhists:labeled')

# Tag this video as being labeled
new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with HSV histograms')
