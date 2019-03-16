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

from collections import OrderedDict
import scannertools as st

import esper.shot_detection_torch.models.deepsbd_resnet as deepsbd_resnet
import esper.shot_detection_torch.models.deepsbd_alexnet as deepsbd_alexnet
import esper.shot_detection_torch.dataloaders.movies_deepsbd as movies_deepsbd_data

# TRAINING_SET = 'kfolds'
# TRAINING_SET = '400_min'
# TRAINING_SET = '4000_min'
TRAINING_SET = '40000_min'
# TRAINING_SET = 'all_movies'

LOCAL_PATH = '/data'
FOLDS_PATH = '/app/data/shot_detection_folds.pkl'
MODEL_SAVE_PATH = '/app/notebooks/learning/models/deepsbd_resnet_train_on_40000_min_weak'

if LOCAL_PATH is None:
    st.init_storage(os.environ['BUCKET'])
else:
    st.init_storage()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

def test_deepsbd(model, dataloader):
    preds = []
    labels = []
    outputs = []
    i = 0
    for clip_tensor, l, _ in tqdm(dataloader):
        o = model(clip_tensor.to(device))
        l = torch.transpose(torch.stack(l).to(device), 0, 1).float()

        preds += get_label(o)
        labels += get_label(l)
        outputs += o.cpu().data.numpy().tolist()

        i += 1
    
    preds = [2 if p == 2 else 0 for p in preds]
        
    precision, recall, f1, tp, tn, fp, fn = prf1_array(2, 0, labels, preds)
    print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
    print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
    
    return preds, labels, outputs

# test K folds
for i in range(0, 5):
    # import pdb; pdb.set_trace() 
    # load 
    weights = torch.load(os.path.join(
        MODEL_SAVE_PATH,
        'fold{}_{}_iteration.pth'.format(
            i + 1 if TRAINING_SET == 'kfolds' else 1,
            (400 if TRAINING_SET == 'kfolds'
            else 2800 if TRAINING_SET == '400_min'
            else 59000 if TRAINING_SET == '4000_min'
            else 320000)
        )))['state_dict']
    deepsbd_resnet_model_no_clipshots.load_state_dict(weights)
    deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.eval()
    test_dataset = deepsbd_datasets_weak_testing[i]
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)
    test_deepsbd(deepsbd_resnet_model_no_clipshots, dataloader)

