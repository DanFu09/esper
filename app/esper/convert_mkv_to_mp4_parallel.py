import subprocess as sp
import os
import sys
from multiprocessing import Pool
import pickle

mkv_files = sp.check_output('ls *.mkv', shell=True).decode('utf-8')
paths = mkv_files.strip().split('\n')

items_tried = []
if os.path.isfile('convert_progress.pickle'):
    with open('convert_progress.pickle', 'rb') as f:
        items_tried = pickle.load(f)
        f.close()

paths = [path for path in paths if path not in items_tried]

def convert_mkv(mkv_file):
    mp4_name = os.path.splitext(mkv_file)[0] + ".mp4"
    try:
        sp.check_output("ffmpeg -i {} -codec copy -c:a aac {}".format(
            mkv_file, mp4_name), shell=True, stderr=sp.STDOUT)

        return True
    except Exception as e:
        print("Error on {}: {}".format(mkv_file, e))
        return False

with Pool(os.cpu_count()) as pool:
    results = pool.map(convert_mkv, paths)

for path in paths:
    items_tried.append(path)

with open('convert_progress.pickle', 'wb') as f:
    pickle.dump(items_tried, f, pickle.HIGHEST_PROTOCOL)
    f.close()

for path, result in zip(paths, results):
    if result:
        print("Success: ", path)
    else:
        print("Failure: ", path)
