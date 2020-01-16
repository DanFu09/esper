import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from query.models import Video, VideoTag, Frame, Shot
import numpy as np
from scipy.spatial import distance
from esper.prelude import load_frame
import cv2
from tqdm import tqdm
import pickle
import rekall
from rekall.video_interval_collection import VideoIntervalCollection

# Parameters for histogram outlier detection
WINDOW_SIZE = 500
POSITIVE_OUTLIER_THRESHOLD_COLOR_HIST = 4.0
NEGATIVE_OUTLIER_THRESHOLD_COLOR_HIST = 0.0
POSITIVE_OUTLIER_THRESHOLD_FLOW_HIST = 4.5
NEGATIVE_OUTLIER_THRESHOLD_FLOW_HIST = 0.0

def color_histogram_shot_labels(histogram, WINDOW_SIZE,
        POSITIVE_OUTLIER_THRESHOLD, NEGATIVE_OUTLIER_THRESHOLD, dim=3):
    histogram = list(histogram)
    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([
        np.mean([distance.chebyshev(histogram[i - 1][j], histogram[i][j]) for j in range(dim)])
        for i in range(1, len(histogram))
    ])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Do simple outlier detection to find boundaries between shots
    positive_boundaries = []
    negative_boundaries = []
    for i in range(1, n):
        window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if diffs[i] - np.mean(window) > POSITIVE_OUTLIER_THRESHOLD * np.std(window):
            positive_boundaries.append(i)
        if diffs[i] - np.mean(window) < NEGATIVE_OUTLIER_THRESHOLD * np.std(window):
            negative_boundaries.append(i)
    
    return positive_boundaries, negative_boundaries

POSITIVE_OUTLIER_THRESHOLD_FLOW_MAGNITUDE = 2.0
NEGATIVE_OUTLIER_THRESHOLD_FLOW_MAGNITUDE = 1.0
def flow_magnitude_shot_labels(flow_histogram):
    flow_histogram = list(flow_histogram)
    avg_magnitudes = [
        np.sum([i * bin_num for bin_num, i in enumerate(hist[0])]) / np.sum(hist[0])
        for hist in flow_histogram
    ]
    n = len(avg_magnitudes)

    positive_boundaries = []
    negative_boundaries = []
    for i in range(n):
        window = avg_magnitudes[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if avg_magnitudes[i] - np.mean(window) > POSITIVE_OUTLIER_THRESHOLD_FLOW_MAGNITUDE * np.std(window):
            positive_boundaries.append(i)
        if avg_magnitudes[i] - np.mean(window) < NEGATIVE_OUTLIER_THRESHOLD_FLOW_MAGNITUDE * np.std(window):
            negative_boundaries.append(i)

    return positive_boundaries, negative_boundaries

db = scannerpy.Database()

# Load up all manually annotated shots
shots_qs = Shot.objects.filter(labeler__name__contains='manual')
shots = VideoIntervalCollection.from_django_qs(shots_qs)
shot_video_ids = sorted(list(shots.get_allintervals().keys()))

#videos = list(Video.objects.filter(ignore_film=False).exclude(id__in=shot_video_ids).order_by('id').all())
videos = list(Video.objects.filter(ignore_film=False).order_by('id').all())

frames = [
    range(0, video.num_frames) for video in videos
]

print("Generating weak labels from RGB histograms")
output_directory = '/app/data/shot_detection_weak_labels/rgb_hists_high_pre'
rgb_hists = st.histograms.compute_histograms(
    db,
    videos = [video.for_scannertools() for video in videos],
    frames=frames
)
for video, rgb_hist in tqdm(zip(videos, rgb_hists), total=len(videos)):
    pos_bounds, neg_bounds = color_histogram_shot_labels(rgb_hist.load(),
            WINDOW_SIZE, POSITIVE_OUTLIER_THRESHOLD_COLOR_HIST,
            NEGATIVE_OUTLIER_THRESHOLD_COLOR_HIST) 

    with open(os.path.join(output_directory, '{}.pkl'.format(video.id)), 'wb') as f:
        pickle.dump((pos_bounds, neg_bounds), f) 

print("Generating weak labels from HSV histograms")
output_directory = '/app/data/shot_detection_weak_labels/hsv_hists_high_pre'
hsv_hists = st.histograms.compute_hsv_histograms(
    db,
    videos = [video.for_scannertools() for video in videos],
    frames=frames
)
for video, hsv_hist in tqdm(zip(videos, hsv_hists), total=len(videos)):
    pos_bounds, neg_bounds = color_histogram_shot_labels(hsv_hist.load(), 
            WINDOW_SIZE, POSITIVE_OUTLIER_THRESHOLD_COLOR_HIST,
            NEGATIVE_OUTLIER_THRESHOLD_COLOR_HIST) 

    with open(os.path.join(output_directory, '{}.pkl'.format(video.id)), 'wb') as f:
        pickle.dump((pos_bounds, neg_bounds), f) 

print("Generating weak labels from optical flow histograms")
output_directory_diffs = '/app/data/shot_detection_weak_labels/flow_hists_diffs_high_pre'
output_directory_magnitude = '/app/data/shot_detection_weak_labels/flow_hists_magnitude'
flow_hists = st.histograms.compute_flow_histograms(
    db,
    videos = [video.for_scannertools() for video in videos],
    frames=frames
)
for video, flow_hist in tqdm(zip(videos, flow_hists), total=len(videos)):
    hist = list(flow_hist.load())
    pos_bounds_diff, neg_bounds_diff = color_histogram_shot_labels(hist,
            WINDOW_SIZE, POSITIVE_OUTLIER_THRESHOLD_COLOR_HIST,
            NEGATIVE_OUTLIER_THRESHOLD_COLOR_HIST, dim=2) 
    #pos_bounds_magnitude, neg_bounds_magnitude = flow_magnitude_shot_labels(hist)

    with open(os.path.join(output_directory_diffs, '{}.pkl'.format(video.id)), 'wb') as f:
        pickle.dump((pos_bounds_diff, neg_bounds_diff), f) 
    #with open(os.path.join(output_directory_magnitude, '{}.pkl'.format(video.id)), 'wb') as f:
    #    pickle.dump((pos_bounds_magnitude, neg_bounds_magnitude), f) 
