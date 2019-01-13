from operator import itemgetter, attrgetter
from query.models import Video
from esper.prelude import collect
from rekall.interval_list import Interval, IntervalList
import os
import json
import re

CAPTION_METADATA_DIR = '../data/subs/meta'

def clean_speaker(speaker):
    speaker = speaker.lower()
    speaker = re.sub(r'\([^(]+\)', '', speaker)
    return speaker

def caption_metadata_for_video(video_id):
    metadata_file = os.path.join(CAPTION_METADATA_DIR, str(video_id) + '_submeta.json')
    if os.path.exists(metadata_file):
        with open(metadata_file) as json_data:
            video_captions = json.load(json_data)
            intervals = []
            for cap in video_captions:
                start = cap['original_time'][0]
                end = cap['original_time'][1]
                aligned = False
                speaker = clean_speaker(cap['speaker']) if 'speaker' in cap else None
                if 'aligned_time' in cap:
                    start = cap['aligned_time'][0]
                    end = cap['aligned_time'][1]
                    aligned = True
                intervals.append(Interval(start, end, payload={'aligned': aligned, 'full_line': cap['line'], 'speaker': speaker, 'man_start': cap['original_time'][0], 'man_end': cap['original_time'][1]}))

            return IntervalList(intervals)
    return IntervalList([])
