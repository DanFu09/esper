import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from esper.prelude import Notifier
from query.models import Video, Shot, VideoTag, Labeler, Tag

videos = Video.objects.filter(name__contains='godfather')
db = scannerpy.Database()

hists = st.shot_detection.compute_histograms(db,
        videos=[video.for_scannertools() for video in videos],
        run_opts={'work_packet_size': 100, 'io_packet_size': 10000},
        megabatch=1)
boundaries = st.shot_detection.compute_shot_boundaries(db,
        videos=[video.for_scannertools() for video in videos],
        histograms=hists,
        megabatch=1)

LABELER, _ = Labeler.objects.get_or_create(name='shot-histogram')
LABELED_TAG, _ = Tag.objects.get_or_create(name='shot-histogram:labeled')

# Label videos
new_videotags = [VideoTag(video=video, tag=LABELED_TAG) for video in videos]
VideoTag.objects.bulk_create(new_videotags)

# Insert Shots into database
new_shots = []
for video, boundarylist in zip(videos, boundaries):
    start = 0
    for boundary in boundarylist:
        if boundary == start:
            continue
        elif boundary < start:
            start = boundary
            continue
        end = boundary - 1
        new_shots.append(Shot(
            min_frame=start, max_frame = end, labeler=LABELER, video=video))
        start = boundary
    new_shots.append(Shot(
        min_frame=start,
        max_frame = video.num_frames, 
        labeler=LABELER, 
        video=video))
Shot.objects.bulk_create(new_shots)

Notifier().notify('Done with shot calculation!')
