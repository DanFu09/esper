import random
from esper.prelude import *
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.temporal_predicates import *
from esper.rekall import *
import matplotlib.pyplot as plt
import cv2
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import scannertools as st

import esper.shot_detection_torch.models.deepsbd_resnet as deepsbd_resnet
import esper.shot_detection_torch.models.deepsbd_alexnet as deepsbd_alexnet
import esper.shot_detection_torch.dataloaders.movies_deepsbd as movies_deepsbd_data

st.init_storage(os.environ['BUCKET'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load up five folds
with open('/app/data/shot_detection_folds.pkl', 'rb') as f:
    folds = pickle.load(f)
    
# Load DeepSBD datasets for each fold
deepsbd_datasets = []
for fold in folds:
    shots_in_fold_qs = Shot.objects.filter(
        labeler__name__contains='manual',
        video_id__in = fold
    )
    shots_in_fold = VideoIntervalCollection.from_django_qs(shots_in_fold_qs)
    
    data = movies_deepsbd_data.DeepSBDDataset(shots_in_fold, verbose=True)
    deepsbd_datasets.append(data)

# dataset to hold multiple folds
class DeepSBDTrainDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
    
    def __len__(self):
        return sum(len(d) for d in self.datasets)
    
    def __getitem__(self, idx):
        for d in self.datasets:
            if idx < len(d):
                return d[idx]
            else:
                idx -= len(d)
        
        return None
    
    def weights_for_balanced_classes(self):
        labels = [
            item[3]
            for d in self.datasets
            for item in d.items
        ]
        
        class_counts = {}
        for l in labels:
            if l not in class_counts:
                class_counts[l] = 1
            else:
                class_counts[l] += 1
        
        weights_per_class = {
            l: len(labels) / class_counts[l]
            for l in class_counts
        }
        
        return [
            weights_per_class[l]
            for l in labels
        ]

# resnet deepSBD pre-trained on Kinetics
deepsbd_resnet_model_no_clipshots = deepsbd_resnet.resnet18(
    num_classes=3,
    sample_size=128,
    sample_duration=16
)
deepsbd_resnet_model_no_clipshots.load_weights('/app/notebooks/learning/models/resnet-18-kinetics.pth')
deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.to(device).train()

# alexnet deepSBD
# deepsbd_alexnet_model_no_clipshots = deepsbd_alexnet.deepSBD()