import scannerpy 
import scannertools as st
import os
from django.db.models import Q
from django.db import transaction
from query.models import Video, Frame, Labeler, Tag, VideoTag
from esper.prelude import Notifier
import numpy as np
from tqdm import tqdm

# Labeler for this pipeline
LABELER, _ = Labeler.objects.get_or_create(name='hsv-saturation')
LABELED_TAG, _ = Tag.objects.get_or_create(name='hsv-saturation:labeled')

# Get all the videos that haven't been labeled with this pipeline
ids_to_exclude = set([36, 122, 205, 243, 304, 336, 455, 456, 503])

labeled_videos = set([videotag.video_id
        for videotag in VideoTag.objects.filter(tag=LABELED_TAG).all()])
all_videos = set([video.id for video in Video.objects.all()])
#video_ids = sorted(list(all_videos.difference(labeled_videos).difference(ids_to_exclude)))
video_ids=sorted(list(all_videos.difference(ids_to_exclude)))[78:]

print(video_ids, len(labeled_videos), len(video_ids))

videos = Video.objects.filter(id__in=video_ids).order_by('id').all()

print("Getting frames to compute on")
# Get frames that we computed faces on
frames = [
    [
        f.number
        for f in Frame.objects.filter(video_id=video,
            tags__name='face_computed').order_by('number')
    ]
    for video in tqdm(video_ids)
]

if True:
    db = scannerpy.Database() 

    print("Loading HSV histograms from Scanner")

    # Make sure the histograms have been computed already!
    hsv_histograms = st.histograms.compute_hsv_histograms(
        db,
        videos=[video.for_scannertools() for video in list(videos)]
    )

    print("Saving saturation values to database")
    for video, frame_list, histogram_list in tqdm(zip(videos, frames, hsv_histograms), total=len(videos)):
        new_frame_tags = []
        frames_to_update = []

        num_pixels = float(video.width * video.height)
        for i, histogram in enumerate(histogram_list.load()):
            if i in frame_list:
                saturation_channel = [ histogram[1][j] for j in range(len(histogram[1])) ]
                bin_size = 255.0 / len(saturation_channel)
                avg_saturation = np.sum([count * j * bin_size
                    for j, count in enumerate(saturation_channel)]) / num_pixels
                
                frame_obj = Frame.objects.get(video_id=video.id, number=i)
                frame_obj.saturation = avg_saturation
                #new_frame_tags.append(
                #    Frame.tags.through(frame_id=frame_obj.pk, tag_id=LABELED_TAG.pk))
                frames_to_update.append(frame_obj)
        with transaction.atomic():
            for frame in frames_to_update:
                frame.save()
            #Frame.tags.through.objects.bulk_create(new_frame_tags)

    # Get the videos that already have the tag
    videos_tagged_already = set([
        vtag.video_id
        for vtag in VideoTag.objects.filter(tag=LABELED_TAG).all()
    ])

    # Tag this video as being labeled
    new_videotags = [
        VideoTag(video=video, tag=LABELED_TAG)
        for video in videos
        if video.id not in videos_tagged_already
    ]
    VideoTag.objects.bulk_create(new_videotags)

Notifier().notify('Done with saturation computation')
