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

# MODEL_SAVE_PATH = '/app/notebooks/learning/models/deepsbd_resnet_train_on_4000_min_majority_vote_downsampled'
MODEL_SAVE_PATH = sys.argv[1]

if not os.path.exists(MODEL_SAVE_PATH):
    "Model save path {} does not exist".format(MODEL_SAVE_PATH)

LOCAL_PATH = '/app/data'
FOLDS_PATH = '/app/data/shot_detection_folds.pkl'
PER_ITERATION_LOGS = 'average_f1'
PER_FOLD_LOGS = 'per_fold_perf'
ITERATION_START = 1000
ITERATION_END = 30000 + 1
ITERATION_STRIDE = 1000
SAME_VAL_TEST = True
VAL_WINDOWS = '/app/data/shot_detection_weak_labels/validation_windows_same_val_test.pkl'
TEST_WINDOWS = '/app/data/shot_detection_weak_labels/test_windows_same_val_test.pkl'
Y_VAL = '/app/data/shot_detection_weak_labels/Y_val_windows_downsampled_same_val_test.npy'
Y_TEST = '/app/data/shot_detection_weak_labels/Y_test_windows_downsampled_same_val_test.npy'

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

if TRAINING_SET == "kfolds":
    # Load DeepSBD datasets for each fold. This is used for testing.
    deepsbd_datasets_weak_testing = []
    for fold in folds:
        shots_in_fold_qs = Shot.objects.filter(
            labeler__name__contains='manual',
            video_id__in = fold
        )
        shots_in_fold = VideoIntervalCollection.from_django_qs(shots_in_fold_qs)
        shots_per_fold.append(shots_in_fold)
        
        data = movies_deepsbd_data.DeepSBDDataset(shots_in_fold, verbose=True, 
                                                  preload=False, logits=True,
                                                 local_path=LOCAL_PATH)
        deepsbd_datasets_weak_testing.append(data)
else:
    if not SAME_VAL_TEST:
        # Load DeepSBD datasets for testing
        val_set = folds[2] + folds[3]
        test_set = folds[0] + folds[1] + folds[4]

        shots_in_val_qs = Shot.objects.filter(
            labeler__name__contains="manual",
            video_id__in = val_set
        )
        shots_in_val = VideoIntervalCollection.from_django_qs(shots_in_val_qs)
        data_val = movies_deepsbd_data.DeepSBDDataset(shots_in_val, verbose=True, 
                                                  preload=False, logits=True,
                                                 local_path=LOCAL_PATH, stride=8)

        shots_in_test_qs = Shot.objects.filter(
            labeler__name__contains="manual",
            video_id__in = test_set
        )
        shots_in_test = VideoIntervalCollection.from_django_qs(shots_in_test_qs)
        data_test = movies_deepsbd_data.DeepSBDDataset(shots_in_test, verbose=True, 
                                                  preload=False, logits=True,
                                                 local_path=LOCAL_PATH, stride=8)
    else:
        # Load DeepSBD datasets for validation and testing
        with open(VAL_WINDOWS, 'rb') as f:
            val_windows_by_video_id = pickle.load(f)
        with open(TEST_WINDOWS, 'rb') as f:
            test_windows_by_video_id = pickle.load(f)
        with open(Y_VAL, 'rb') as f:
            Y_val = np.load(f)
        with open(Y_TEST, 'rb') as f:
            Y_test = np.load(f)
        paths = {
            video_id: Video.objects.get(id=video_id).path
            for video_id in list(set([
                vid for vid, start, end in val_windows_by_video_id    
            ]))
        }

        # val_windows_by_video_id = [
        #     (video_id, interval.start, interval.end)
        #     for video_id in sorted(list(val_windows.get_allintervals().keys()))
        #     for interval in val_windows.get_intervallist(video_id).get_intervals()
        # ]
        # test_windows_by_video_id = [
        #     (video_id, interval.start, interval.end)
        #     for video_id in sorted(list(test_windows.get_allintervals().keys()))
        #     for interval in test_windows.get_intervallist(video_id).get_intervals()
        # ]

        def val_to_logits(val):
            """ If val is 1, positive; if val is 2, negative """
            return (0, 0, 1) if val == 1 else (1, 0, 0)

        shots = VideoIntervalCollection.from_django_qs(Shot.objects.filter(
            labeler__name__contains="manual"
        ))
        data_val = movies_deepsbd_data.DeepSBDDataset(shots, verbose=True,
                preload=False, logits=True, local_path=LOCAL_PATH, stride=16)
        data_test = movies_deepsbd_data.DeepSBDDataset(shots, verbose=True,
                preload=False, logits=True, local_path=LOCAL_PATH, stride=16)
        data_val.set_items([
            (video_id, start, end, val_to_logits(label), paths[video_id])
            for (video_id, start, end), label in zip(val_windows_by_video_id, Y_val)
        ])
        data_test.set_items([
            (video_id, start, end, val_to_logits(label), paths[video_id])
            for (video_id, start, end), label in zip(test_windows_by_video_id, Y_test)
        ])

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

if TRAINING_SET == "kfolds":
    for iteration in range(ITERATION_START, ITERATION_END, ITERATION_STRIDE):
        fold_data = []
        print(iteration)
        # test k folds
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
else:
    for iteration in range(ITERATION_START, ITERATION_END, ITERATION_STRIDE):
        dataset_info = []
        print(iteration)
        for i, dataset in enumerate([data_val, data_test]):
            weights = torch.load(os.path.join(
                MODEL_SAVE_PATH,
                'fold1_{}_iteration.pth'.format(
                    iteration
                )))['state_dict']
            deepsbd_resnet_model_no_clipshots.load_state_dict(weights)
            deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.eval()
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=48,
                pin_memory=True)
            precision, recall, f1, tp, tn, fp, fn = test_deepsbd(deepsbd_resnet_model_no_clipshots, dataloader)
            dataset_info.append(f1)

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
                        iteration, 'val' if i == 0 else 'test',
                        pre=precision, rec=recall, f1=f1, tp=tp,
                        tn=tn, fp=fp, fn=fn
                    ))

        f_iteration.write('{}\t{}\t{}\n'.format(iteration,
            dataset_info[0], dataset_info[1]))
        f_folds.flush()
        f_iteration.flush()
