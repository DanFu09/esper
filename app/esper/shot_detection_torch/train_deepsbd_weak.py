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
import django

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler

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
PRETRAIN_PATH = '/app/notebooks/learning/models/resnet-18-kinetics.pth'
FOLDS_PATH = '/app/data/shot_detection_folds.pkl'
WEAK_LABELS_PATH = '/app/data/shot_detection_weak_labels/majority_vote_labels_all_windows_high_pre.npy'
MODEL_SAVE_PATH = '/app/notebooks/learning/models/deepsbd_resnet_train_on_4000_min_majority_vote_high_pre'
SEGS_400_MIN_PATH = '/app/data/400_minute_train.pkl'
SEGS_4000_MIN_PATH = '/app/data/4000_minute_train.pkl'
SEGS_40000_MIN_PATH = '/app/data/40000_minute_train.pkl'
SEGS_ALL_VIDEOS_PATH = '/app/data/all_videos_train.pkl'

# only works for 400_min, 4000_min, all_movies
CONTINUE_PATH = None
# CONTINUE_PATH = '/app/notebooks/learning/models/deepsbd_resnet_train_on_40000_min_weak/fold1_270000_iteration.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('Initialized constants')

# load folds from disk
with open(FOLDS_PATH, 'rb') as f:
    folds = pickle.load(f)

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

print('Loaded test data')
    
# load weak labels
with open(WEAK_LABELS_PATH, 'rb') as f:
    weak_labels_windows = np.load(f)
    
print('Loaded weak labels from disk')
    
weak_labels_collected = collect(
    weak_labels_windows,
    lambda row: row[0][0]
)

weak_labels_col = VideoIntervalCollection({
    video_id: [
        (row[0][1] ,row[0][2], row[1])
        for row in weak_labels_collected[video_id]
    ]
    for video_id in tqdm(list(weak_labels_collected.keys()))
})

print('Finished collecting weak labels')

def weak_payload_to_logits(weak_payload):
    return (weak_payload[1], 0., weak_payload[0])

if TRAINING_SET == 'kfolds':
    deepsbd_datasets_weak_training = []
    for fold in folds:
        shots_in_fold_qs = Shot.objects.filter(
            labeler__name__contains='manual',
            video_id__in = fold
        )
        shots_in_fold = VideoIntervalCollection.from_django_qs(shots_in_fold_qs)

        data = movies_deepsbd_data.DeepSBDDataset(shots_in_fold, verbose=True,
                                                  preload=False, logits=True,
                                                 local_path=LOCAL_PATH)
        items_collected = collect(
            data.items,
            lambda item: item[0]
        )
        items_col = VideoIntervalCollection({
            video_id: [
                (item[1], item[2], (item[3], item[4]))
                for item in items_collected[video_id]
            ]
            for video_id in items_collected
        })

        new_items = weak_labels_col.join(
            items_col,
            predicate=equal(),
            working_window=1,
            merge_op = lambda weak, item: [
                (weak.start, weak.end, (weak.payload, item.payload[1]))
            ]
        )

        data.items = [
            (video_id, intrvl.start, intrvl.end, weak_payload_to_logits(intrvl.payload[0]), intrvl.payload[1])
            for video_id in sorted(list(new_items.get_allintervals().keys()))
            for intrvl in new_items.get_intervallist(video_id).get_intervals()
        ]
        deepsbd_datasets_weak_training.append(data)
elif TRAINING_SET in ['400_min', '4000_min', '40000_min', 'all_movies']:
    with open(SEGS_400_MIN_PATH if TRAINING_SET == '400_min'
            else SEGS_4000_MIN_PATH if TRAINING_SET == '4000_min'
            else SEGS_40000_MIN_PATH if TRAINING_SET == '40000_min'
            else SEGS_ALL_VIDEOS_PATH,
              'rb') as f:
        segments = VideoIntervalCollection(pickle.load(f))

    print('Creating dataset')
    data = movies_deepsbd_data.DeepSBDDataset(segments, verbose=True,
                                           preload=False, logits=True,
                                           local_path=LOCAL_PATH)
    print('Collecting')
    items_collected = collect(
        data.items,
        lambda item: item[0]
    )
    print('Recreating VIC')
    items_col = VideoIntervalCollection({
        video_id: [
            (item[1], item[2], (item[3], item[4]))
            for item in items_collected[video_id]
        ]
        for video_id in tqdm(items_collected)
    })

    print('Creating new items')
    new_items = weak_labels_col.join(
        items_col,
        predicate=equal(),
        working_window=1,
        merge_op = lambda weak, item: [
            (weak.start, weak.end, (weak.payload, item.payload[1]))
        ]
    )

    data.items = [
        (video_id, intrvl.start, intrvl.end, weak_payload_to_logits(intrvl.payload[0]), intrvl.payload[1])
        for video_id in sorted(list(new_items.get_allintervals().keys()))
        for intrvl in new_items.get_intervallist(video_id).get_intervals()
    ]
    deepsbd_datasets_weak_training = [data]


print('Finished constructing datasets')
    
# dataset to hold multiple folds for weak data
class DeepSBDWeakTrainDataset(Dataset):
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
            np.argmax(item[3])
            for d in self.datasets
            for item in d.items
        ]
        
        class_counts = [
            0
            for i in range(len(self.datasets[0].items[0]))
        ]
        for l in labels:
            class_counts[l] += 1
        
        weights_per_class = {
            i: len(labels) / l if l != 0 else 0
            for i, l in enumerate(class_counts)
        }
        
        return [
            weights_per_class[l]
            for l in labels
        ]

# helper functions for deepsbd testing
def calculate_accuracy_logits(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    _, target_preds = targets.topk(1, 1, True)
    correct = pred.eq(target_preds.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

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
    for clip_tensor, l, _ in tqdm(dataloader):
        o = model(clip_tensor.to(device))
        l = torch.transpose(torch.stack(l).to(device), 0, 1).float()

        preds += get_label(o)
        labels += get_label(l)
        outputs += o.cpu().data.numpy().tolist()
    
    preds = [2 if p == 2 else 0 for p in preds]
        
    precision, recall, f1, tp, tn, fp, fn = prf1_array(2, 0, labels, preds)
    print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
    print("TP: {}, TN: {}, FP: {}, FN: {}".format(tp, tn, fp, fn))
    
    return preds, labels, outputs

def train_iterations(iterations, training_dataloader, model, criterion, optimizer, 
                     scheduler, fold_num=1, log_file=None, save_every=100, start_iter = 0):
    i = start_iter
    training_iter = iter(training_dataloader)
    
    while i < iterations:
        data = next(training_iter, None)
        if data == None:
            training_iter = iter(training_dataloader)
            continue
        i += 1
        clip_tensor, targets, _ = data
        
        outputs = model(clip_tensor.to(device))
        targets = torch.transpose(torch.stack(targets).to(device), 0, 1).float()
        
        loss = criterion(outputs, targets)
        acc = calculate_accuracy_logits(outputs, targets)
        preds = get_label(outputs)
        preds = [2 if p == 2 else 0 for p in preds]
        target_preds = get_label(targets)
        precision, recall, f1, tp, tn, fp, fn = prf1_array(
            2, 0, target_preds, preds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('Epoch: [{0}/{1}]\t'
              'Loss_conf {loss_c:.4f}\t'
              'acc {acc:.4f}\t'
              'pre {pre:.4f}\t'
              'rec {rec:.4f}\t'
              'f1 {f1: .4f}\t'
              'TP {tp} '
              'TN {tn} '
              'FP {fp} '
              'FN {fn} '
              .format(
                  i, iterations, loss_c=loss.item(), acc=acc,
                  pre=precision, rec=recall, f1=f1,
                  tp=tp, tn=tn, fp=fp, fn=fn))
        
        if log_file is not None:
            log_file.write('Epoch: [{0}/{1}]\t'
              'Loss_conf {loss_c:.4f}\t'
              'acc {acc:.4f}\t'
              'pre {pre:.4f}\t'
              'rec {rec:.4f}\t'
              'f1 {f1: .4f}\t'
              'TP {tp} '
              'TN {tn} '
              'FP {fp} '
              'FN {fn}\n'.format(
                  i, iterations, loss_c=loss.item(), acc=acc,
                  pre=precision, rec=recall, f1=f1,
                  tp=tp, tn=tn, fp=fp, fn=fn
              ))

        if (i % save_every) == 0:
            save_file_path = os.path.join(
                MODEL_SAVE_PATH,
                'fold{}_{}_iteration.pth'.format(fold_num, i)
            )
            states = {
                'iteration': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(states, save_file_path)
    
print('Loading pretrained Kinetics data')
    
# resnet deepSBD pre-trained on Kinetics
deepsbd_resnet_model_no_clipshots = deepsbd_resnet.resnet18(
    num_classes=3,
    sample_size=128,
    sample_duration=16
)
deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.to(device).train()    

print('Training')

if TRAINING_SET == 'kfolds':
    # train K folds
    for i in range(5):
        with open(os.path.join(MODEL_SAVE_PATH, '{}.log'.format(i+1)), 'w') as log_file:
            training_datasets = DeepSBDWeakTrainDataset(
                deepsbd_datasets_weak_training[:i] + deepsbd_datasets_weak_training[i+1:])
            fold_weights = torch.DoubleTensor(training_datasets.weights_for_balanced_classes())
            fold_sampler = torch.utils.data.sampler.WeightedRandomSampler(fold_weights, len(fold_weights))

            training_dataloader = DataLoader(
                training_datasets,
                num_workers=0,
                shuffle=False,
                batch_size=16,
                sampler=fold_sampler
            )

            criterion = nn.BCEWithLogitsLoss()

            # reset model
            deepsbd_resnet_model_no_clipshots.load_weights(PRETRAIN_PATH)
            optimizer = optim.SGD(deepsbd_resnet_model_no_clipshots.parameters(), 
                                  lr=.001, momentum=.9, weight_decay=1e-3)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=60000)

            train_iterations(
                400, training_dataloader, 
                deepsbd_resnet_model_no_clipshots, 
                criterion, optimizer, scheduler, fold_num = i + 1,
                log_file = log_file
            )
elif TRAINING_SET in ['400_min', '4000_min', '40000_min', 'all_movies']:
    with open(os.path.join(MODEL_SAVE_PATH, '{}.log'.format(TRAINING_SET)), 'a') as log_file:
    #if True:
    #    log_file = None
        training_datasets = DeepSBDWeakTrainDataset(deepsbd_datasets_weak_training)
        fold_weights = torch.DoubleTensor(training_datasets.weights_for_balanced_classes())
        fold_sampler = torch.utils.data.sampler.WeightedRandomSampler(fold_weights, len(fold_weights))

        django.db.connections.close_all()
        training_dataloader = DataLoader(
            training_datasets,
            num_workers=48,
            shuffle=False,
            batch_size=16,
            pin_memory=True,
            sampler=fold_sampler
        )

        criterion = nn.BCEWithLogitsLoss()

        # reset model
        deepsbd_resnet_model_no_clipshots.load_weights(PRETRAIN_PATH)
        optimizer = optim.SGD(deepsbd_resnet_model_no_clipshots.parameters(), 
                              lr=.001, momentum=.9, weight_decay=1e-3)
        start_iter = 0
        if CONTINUE_PATH is not None:
            checkpoint = torch.load(CONTINUE_PATH)
            deepsbd_resnet_model_no_clipshots.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iteration']
        
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=60000)

        train_iterations(
            (4000 if TRAINING_SET == '400_min'
            else 60000 if TRAINING_SET == '4000_min'
            else 400000 if TRAINING_SET == '40000_min'
            else 800000),
            training_dataloader, 
            deepsbd_resnet_model_no_clipshots, 
            criterion, optimizer, scheduler,
            log_file = log_file, start_iter = start_iter, save_every=1000
        )

#print('Testing')
#            
## test K folds
#for i in range(0, 5):
#    # load 
#    weights = torch.load(os.path.join(
#        MODEL_SAVE_PATH,
#        'fold{}_{}_iteration.pth'.format(
#            i + 1 if TRAINING_SET == 'kfolds' else 1,
#            (400 if TRAINING_SET == 'kfolds'
#            else 4000 if TRAINING_SET == '400_min'
#            else 60000 if TRAINING_SET == '4000_min'
#            else 400000)
#        )))['state_dict']
#    deepsbd_resnet_model_no_clipshots.load_state_dict(weights)
#    deepsbd_resnet_model_no_clipshots = deepsbd_resnet_model_no_clipshots.eval()
#    test_dataset = deepsbd_datasets_weak_testing[i]
#    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16)
#    test_deepsbd(deepsbd_resnet_model_no_clipshots, dataloader)
