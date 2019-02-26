import rekall as rk
from rekall.video_interval_collection import VideoIntervalCollection
from rekall.temporal_predicates import *
from ..lib.spatial_transforms import Scale, ToTensor, Normalize, get_mean
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from query.models import Video
from tqdm import tqdm

class DeepSBDDataset(Dataset):
    def __init__(self, shots, window_size=16, stride=8, size=128, verbose=False):
        """Constrcutor for ShotDetectionDataset.
        
        Args:
            shots: VideoIntervalCollection of all the intervals to get frames from. If the payload is -1,
            then the interval is not an actual shot and just needs to be included in the dataset.
        """
        self.window_size = window_size
        items = set()
        frame_nums = {}
        
        shot_boundaries = shots.map(
            lambda intrvl: (intrvl.start, intrvl.start, intrvl.payload)
        ).filter(lambda intrvl: intrvl.payload != -1)
        
        clips = shots.dilate(1).coalesce().dilate(-1).map(
            lambda intrvl: (
                intrvl.start - stride - ((intrvl.start - stride) % stride),
                intrvl.end + stride - ((intrvl.end + stride) % stride),
                intrvl.payload
            )
        ).dilate(1).coalesce().dilate(-1)
        
        items_intrvls = {}
        for video_id in clips.get_allintervals():
            items_intrvls[video_id] = []
            for intrvl in clips.get_intervallist(video_id).get_intervals():
                items_intrvls[video_id] += [
                    (f, f + window_size, 0)
                    for f in range(intrvl.start, intrvl.end - stride, stride)
                ]
        items_col = VideoIntervalCollection(items_intrvls)
        
        items_w_boundaries = items_col.filter_against(
            shot_boundaries,
            predicate=during_inv()
        ).map(
            lambda intrvl: (intrvl.start, intrvl.end, 2)
        )
        
        items_w_labels = items_col.minus(
            items_w_boundaries, predicate=equal()
        ).set_union(items_w_boundaries)

        for video_id in items_w_labels.get_allintervals():
            frame_nums[video_id] = set()
            for intrvl in items_w_labels.get_intervallist(video_id).get_intervals():
                items.add((
                    video_id,
                    intrvl.start,
                    intrvl.end,
                    intrvl.payload
                ))
                for f in range(intrvl.start, intrvl.end):
                    frame_nums[video_id].add(f)

        self.items = sorted(list(items))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Scale((128, 128)),
            ToTensor(1),
            Normalize(get_mean(1), (1, 1, 1))
        ])
        
        iterator = tqdm(frame_nums) if verbose else frame_nums
        # Load frames into memory
        self.frames = {
            video_id: {
                'frame_nums': sorted(list(frame_nums[video_id])),
                'frames': [
                    self.transform(f)
                    for f in Video.objects.get(id=video_id).for_scannertools().frames(
                        sorted(list(frame_nums[video_id]))
                    )
                ]
            }
            for video_id in iterator
        }
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        """
        Indexed by video ID, then frame number
        Returns self.window_size frames before the indexed frame to self.window_size
            frames after the indexed frame
        """
        video_id, start_frame, end_frame, label = self.items[idx]
        
        start_index = self.frames[video_id]['frame_nums'].index(start_frame)
        img_tensors = self.frames[video_id]['frames'][start_index:start_index + self.window_size]
        
#         img_tensors = [
#             self.transform(f)
#             for f in Video.objects.get(id=video_id).for_scannertools().frames(
#                 list(range(frame_num - self.window_size, frame_num + self.window_size + 1))
#             )
#         ]
        
        return torch.stack(img_tensors).permute(1, 0, 2, 3), label, (video_id, start_frame, end_frame)
#         return label, (video_id, start_frame, end_frame)