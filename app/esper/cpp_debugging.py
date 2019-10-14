import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, Frame
import numpy as np

videos = Video.objects.filter(name__contains="godfather")
db = scannerpy.Database()
hsv_histograms = st.histograms.compute_hsv_histograms(
    db,
    videos=[video.for_scannertools() for video in videos]
)
