import scannertools as st
from scannertools.prelude import *
from scipy.spatial import distance
import numpy as np
from typing import Sequence
import pickle

WINDOW_SIZE = 500
BOUNDARY_BATCH = 10000000
POSITIVE_OUTLIER = 2.5
NEGATIVE_OUTLIER = 1.0

@scannerpy.register_python_op(name='ColorHistogramShotLabels', batch=BOUNDARY_BATCH)
def color_histogram_shot_labels(config, histograms: Sequence[bytes]) -> Sequence[bytes]:
    hists = [readers.histograms(byts, config.protobufs) for byts in histograms]

    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([
        np.mean([distance.chebyshev(hists[i - 1][j], hists[i][j]) for j in range(3)])
        for i in range(1, len(hists))
    ])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)
    
    # Do simple outlier detection to find boundaries between shots
    positive_boundaries = []
    negative_boundaries = []
    for i in range(1, n):
        window = diffs[max(i - WINDOW_SIZE, 0):min(i + WINDOW_SIZE, n)]
        if diffs[i] - np.mean(window) > POSITIVE_OUTLIER * np.std(window):
            positive_boundaries.append(i)
        if diffs[i] - np.mean(window) < NEGATIVE_OUTLIER * np.std(window):
            negative_boundaries.append(i)

    return [pickle.dumps((positive_boundaries, negative_boundaries))] + ['\0' for _ in range(len(histograms) - 1)]


class ColorHistogramShotLabelsPipeline(Pipeline):
    job_suffix = 'color_histogram_shot_labels'
    base_sources = ['videos', 'histograms']
    run_opts = {
        'io_packet_size': BOUNDARY_BATCH,
        'work_packet_size': BOUNDARY_BATCH
    }

    def build_pipeline(self):
        return {
            'color_histogram_shot_labels': self._db.ops.ColorHistogramShotLabels(histograms=self._sources['histograms'].op)
        }

    def parse_output(self):
        boundaries = super().parse_output()
        return [
            pickle.loads(next(b._column.load(rows=[0]))) if b is not None else None
            for b in boundaries
        ]


compute_color_histogram_shot_labels = ColorHistogramShotLabelsPipeline.make_runner()
