import django
django.setup()

import multiprocessing as mp
from rekall.video_interval_collection_3d import VideoIntervalCollection3D
from rekall.interval_set_3d_utils import perf_count
from tqdm import tqdm
import random
from math import ceil

def _inline_do(func, vids, profile):
    with perf_count("Running inline", enable=profile):
        return VideoIntervalCollection3D({
            v: func(v) for v in tqdm(vids)})

def _worker_init(context):
    global GLOBAL_CONTEXT
    GLOBAL_CONTEXT = context

def _apply_function(vids):
    func = GLOBAL_CONTEXT
    return {vid: func(vid) for vid in vids}

def _update_pbar(pbar, work):
    def update(result):
        pbar.update(work)
    return update

def par_do(func, vids, parallel=True, chunksize=1, profile=False, fork=False):
    """
    func takes a video_id and returns an IntervalSet3D. It cannot make
        reference to any objects outside of its scope.
    vids is a list of video_id
    chunksize denotes the batch size for each task that is sent to a worker.
    fork specifies whether to use fork to create child processes. Calling in a
        Jupyter Notebook requires this to be true, but may create problems with
        multithreaded libraries.
    Returns a VideoIntervalCollection3D
    """
    if not parallel:
        return _inline_do(func, vids, profile)

    with perf_count("Running in Parallel", enable=profile):
        if fork:
            method = "fork"
            # Existing connections will not work in forked process.
            # They need to create connections themselves.
            django.db.connections.close_all()
        else:
            method = "spawn"
        with mp.pool.Pool(initializer=_worker_init,
                initargs=(func,),
                context=mp.get_context(method)) as pool:
            with tqdm(total=len(vids)) as pbar:
                with perf_count("Dispatching tasks", enable=profile):
                    futures = []
                    random.shuffle(vids)
                    num_tasks = int(ceil(len(vids)/chunksize))
                    for task_i in range(num_tasks):
                        start = chunksize* task_i
                        end = start + chunksize
                        task = vids[start:end]
                        futures.append(pool.apply_async(
                            _apply_function, args=(task,),
                            callback=_update_pbar(pbar, chunksize),
                            error_callback = _update_pbar(pbar, chunksize)))
                videos_and_results = [f.get() for f in futures]
        output = {}
        for result_dict in videos_and_results:
            keys = result_dict.keys()
            for key in keys:
                if key in output:
                    raise RuntimeError(
                            "duplicated results found for video {0}".format(
                                key))
            output.update(result_dict)
        return VideoIntervalCollection3D(output)
