import random
from esper.prelude import *
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.temporal_predicates import *
from esper.rekall import *
import cv2
import pickle
import multiprocessing as mp
from query.models import Video, Shot
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler

import numpy as np

from collections import OrderedDict

import esper.shot_detection_torch.models.deepsbd_resnet as deepsbd_resnet
import esper.shot_detection_torch.models.deepsbd_alexnet as deepsbd_alexnet
import esper.shot_detection_torch.dataloaders.movies_deepsbd as movies_deepsbd_data

# TRAINING_SET = 'kfolds'
# TRAINING_SET = '400_min'
TRAINING_SET = '4000_min'
# TRAINING_SET = '40000_min'
# TRAINING_SET = 'all_movies'

LOCAL_PATH = '/app/data'
FOLDS_PATH = '/app/data/shot_detection_folds.pkl'
MODEL_SAVE_PATH = '/app/notebooks/learning/models/deepsbd_resnet_train_on_4000_min_majority_vote_high_pre'
PER_ITERATION_LOGS = 'average_f1'
PER_FOLD_LOGS = 'per_fold_perf'
ITERATION_START = 1000
ITERATION_END = 60000 + 1
ITERATION_STRIDE = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Initialized constants')

# load folds from disk
with open(FOLDS_PATH, 'rb') as f:
    folds = pickle.load(f)

# resnet deepSBD pre-trained on Kinetics
deepsbd_resnet_model_no_clipshots = deepsbd_resnet.resnet18(
    num_classes=3,
    sample_size=128,
    sample_duration=16
)
deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.to(device).train()    

# Load DeepSBD datasets for each fold. This is used for testing.
deepsbd_datasets_weak_testing = []
for fold in folds:
    shots_in_fold_qs = Shot.objects.filter(
        labeler__name__contains='manual',
        video_id__in = fold
    )
    shots_in_fold = VideoIntervalCollection.from_django_qs(shots_in_fold_qs)
    
    data = movies_deepsbd_data.DeepSBDDataset(shots_in_fold, verbose=True, 
                                              preload=False, logits=True,
                                             local_path=LOCAL_PATH)
    deepsbd_datasets_weak_testing.append(data)

def prf1_array(pos_label, neg_label, gt, preds):
    tp = 0.
    fp = 0.
    tn = 0.
    fn = 0.
    
    for truth, pred in zip(gt, preds):
        if truth == pred:
            if pred == pos_label:
                tp += 1.
            else:
                tn += 1.
        else:
            if pred == pos_label:
                fp += 1.
            else:
                fn += 1.
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    return (precision, recall, f1, tp, tn, fp, fn)

def get_label(res_tensor):
    res_numpy=res_tensor.data.cpu().numpy()
    labels=[]
    for row in res_numpy:
        labels.append(np.argmax(row))
    return labels

def test_deepsbd(model, dataloader, verbose=True):
    preds = []
    labels = []
    outputs = []
    i = 0
    for clip_tensor, l, _ in (tqdm(dataloader) if verbose else dataloader):
        o = model(clip_tensor.to(device))
        l = torch.transpose(torch.stack(l).to(device), 0, 1).float()

        preds += get_label(o)
        labels += get_label(l)
        outputs += o.cpu().data.numpy().tolist()

        i += 1
    
    preds = [2 if p == 2 else 0 for p in preds]
        
    precision, recall, f1, tp, tn, fp, fn = prf1_array(2, 0, labels, preds)
    if verbose:
        print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
    
    return precision, recall, f1, tp, tn, fp, fn

f_iteration = open(os.path.join(MODEL_SAVE_PATH, PER_ITERATION_LOGS), 'a')
f_folds = open(os.path.join(MODEL_SAVE_PATH, PER_FOLD_LOGS), 'a')

for iteration in range(ITERATION_START, ITERATION_END, ITERATION_STRIDE):
    fold_data = []
    print(iteration)
    # test K folds
    for i in range(0, 5):
        # import pdb; pdb.set_trace() 
        # load 
        weights = torch.load(os.path.join(
            MODEL_SAVE_PATH,
            'fold{}_{}_iteration.pth'.format(
                i + 1 if TRAINING_SET == 'kfolds' else 1,
                iteration
            )))['state_dict']
        deepsbd_resnet_model_no_clipshots.load_state_dict(weights)
        deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.eval()
        test_dataset = deepsbd_datasets_weak_testing[i]
        dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=48,
            pin_memory=True)
        precision, recall, f1, tp, tn, fp, fn = test_deepsbd(deepsbd_resnet_model_no_clipshots, dataloader)
        fold_data.append([precision, recall, f1, tp, tn, fp, fn])
        
        f_folds.write(
                'Iteration {}\t'
                'Fold {}\t'
                'pre {pre:.4f}\t'
                'rec {rec:.4f}\t'
                'f1 {f1: .4f}\t'
                'TP {tp} '
                'TN {tn} '
                'FP {fp} '
                'FN {fn}\n'.format(
                    iteration, i+1, pre=precision, rec=recall, f1=f1, tp=tp,
                    tn=tn, fp=fp, fn=fn
                ))

    f1s = [
        fold_info[2]
        for fold_info in fold_data
    ]
    f_iteration.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        iteration, f1s[0], f1s[1], f1s[2], f1s[3], f1s[4], np.mean(f1s)
    ))

    f_folds.flush()
    f_iteration.flush()

