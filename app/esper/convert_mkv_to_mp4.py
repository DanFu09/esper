import subprocess as sp
import os
import sys
from pprint import pprint
import pickle
import time
import datetime

files_on_cloud = sp.check_output('gsutil ls gs://esper/movies/*.mkv',
        shell=True).decode('utf-8')
paths = files_on_cloud.strip().split('\n')
item_names_on_cloud = [
        os.path.splitext(path.split('/')[-1])[0] + 
        os.path.splitext(path.split('/')[-1])[1]
        for path in paths]

items_tried = []
if os.path.isfile('convert_progress.pickle'):
    with open('convert_progress.pickle', 'rb') as f:
        items_tried = pickle.load(f)
        f.close()

time_sleep = 60
while len(items_tried) != len(item_names_on_cloud):
    items_local = sp.check_output('ls *.mkv', shell=True).decode('utf-8')
    item_names_local = items_local.strip().split('\n')

    item_to_try = None
    for item in item_names_local:
        if item in items_tried:
            continue
        else:
            item_to_try = item

    if item_to_try is None:
        cur_time = datetime.datetime.now().strftime("%Y %b %d %H:%M:%S")
        print('No local mkv files found, sleeping for {} seconds at time {}'.format(
            time_sleep, cur_time))
        time.sleep(time_sleep)
        time_sleep *= 2
        continue
    else:
        time_sleep = 60

    print('Trying {}'.format(item_to_try))
    items_tried.append(item_to_try)
    with open('convert_progress.pickle', 'wb') as f:
        pickle.dump(items_tried, f, pickle.HIGHEST_PROTOCOL)
        f.close()
   
    try:
        item_mkv = item_to_try
        item_mp4 = os.path.splitext(item_to_try)[0] + ".mp4"
        sp.check_output("ffmpeg -i {} -codec copy -c:a aac {}".format(
            item_mkv, item_mp4), shell=True)

        gs_path = "gs://esper/movies/{}".format(item_mp4)
        sp.check_output("gsutil cp {} {}".format(item_mp4, gs_path), shell=True)
        sp.check_output("rm {} {}".format(item_mkv, item_mp4), shell=True)
        print('Finished converting {}'.format(item_to_try))
    except Exception as e:
        print("Error on {}: {}".format(item_to_try, e))

