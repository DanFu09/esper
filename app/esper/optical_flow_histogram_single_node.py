import scannerpy
import scannertools as st
import os
from django.db.models import Q
from query.models import Video
import numpy as np
from esper.prelude import Notifier

videos = Video.objects.filter(id=1)
db = scannerpy.Database()

frames = [list(range(0, video.num_frames)) for video in videos]

flow_histograms = st.histograms.compute_flow_histograms(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    run_opts = {
        'io_packet_size': 128,
        'work_packet_size': 8
    }
)
