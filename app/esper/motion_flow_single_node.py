import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video
import numpy as np
from esper.prelude import Notifier

# Load all Star Wars and Harry Potter films
videos = Video.objects.filter(
        Q(name__contains="star wars") | Q(name__contains="harry potter"))
db = scannerpy.Database()

# Calculate at 2 fps
frames = [
    list(range(0, video.num_frames, int(round(video.fps) / 2)))
    for video in videos
]

# Calculate motion flow
flow = st.optical_flow.compute_flow(
    db,
    videos=[video.for_scannertools() for video in videos],
    frames=frames,
    cache=False
)

Notifier().notify('Done computing motion flow!')
