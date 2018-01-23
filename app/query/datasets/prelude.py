from scannerpy import ProtobufGenerator, Config, Database, Job, BulkJob, DeviceType, ColumnType, ScannerException
from storehouse import StorageConfig, StorageBackend
from query.base_models import ModelDelegator, Track, BoundingBox
from query.datasets.stdlib import *
from django.db import connections, connection
from django.db.models.query import QuerySet
from django.db.models import Min, Max, Count, F, OuterRef, Subquery, Sum, Avg, Func
from django.db.models.functions import Cast, Extract
from django.utils import timezone
from django_bulk_update.manager import BulkUpdateManager
from IPython.core.getipython import get_ipython
from timeit import default_timer as now
import datetime
import _strptime  # https://stackoverflow.com/a/46401422/356915
import django.db.models as models
import os
import subprocess as sp
import numpy as np
import pandas as pd
import sys
import sqlparse
import logging
import cPickle as pickle
import marshal
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import csv
import requests
import cv2
import itertools

# Import all models for current dataset
m = ModelDelegator(os.environ.get('DATASET'))
m.import_all(globals())

# Access to Scanner protobufs
cfg = Config()
proto = ProtobufGenerator(cfg)

# Logging config
log = logging.getLogger('esper')
log.setLevel(logging.DEBUG)
if not log.handlers:

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            level = record.levelname[0]
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[2:]
            if len(record.args) > 0:
                record.msg = '({})'.format(', '.join(
                    [str(x) for x in [record.msg] + list(record.args)]))
                record.args = ()
            return '{level} {time} {filename}:{lineno:03d}] {msg}'.format(
                level=level, time=time, **record.__dict__)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    log.addHandler(handler)

# Only run if we're in an IPython notebook
if get_ipython() is not None:
    # Render all DataFrames via qgrid
    import qgrid
    qgrid.set_grid_option('minVisibleRows', 1)
    qgrid.enable()

    # Matplotlib/seaborn config
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams['figure.figsize'] = (18, 8)
    plt.rc("axes.spines", top=False, right=False)
    sns.set_style('white')

    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# Setup Storehouse
ESPER_ENV = os.environ.get('ESPER_ENV')
BUCKET = os.environ.get('BUCKET')
DATA_PATH = os.environ.get('DATA_PATH')
if ESPER_ENV == 'google':
    storage_config = StorageConfig.make_gcs_config(BUCKET)
else:
    storage_config = StorageConfig.make_posix_config()
storage = StorageBackend.make_from_config(storage_config)


# http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def bbox_area(f):
    return (f.bbox_x2 - f.bbox_x1) * (f.bbox_y2 - f.bbox_y1)


def bbox_midpoint(f):
    return np.array([(f.bbox_x1 + f.bbox_x2) / 2, (f.bbox_y1 + f.bbox_y2) / 2])


def bbox_dist(f1, f2):
    return np.linalg.norm(bbox_midpoint(f1) - bbox_midpoint(f2))


def bbox_iou(f1, f2):
    x1 = max(f1.bbox_x1, f2.bbox_x1)
    x2 = min(f1.bbox_x2, f2.bbox_x2)
    y1 = max(f1.bbox_y1, f2.bbox_y1)
    y2 = min(f1.bbox_y2, f2.bbox_y2)

    if x1 > x2 or y1 > y2: return 0

    intersection = (x2 - x1) * (y2 - y1)
    return intersection / (bbox_area(f1) + bbox_area(f2) - intersection)


def unzip(l, default=([], [])):
    x = tuple(zip(*l))
    if x == ():
        return default
    else:
        return x


def group_by_frame(objs, fn_key, fn_sort, output_dict=False, include_frame=True):
    d = defaultdict(list)
    for obj in objs:
        d[fn_key(obj)].append(obj)

    for l in d.values():
        l.sort(key=fn_sort)

    if output_dict:
        return dict(d)
    else:
        l = sorted(d.iteritems(), key=itemgetter(0))
        if not include_frame:
            l = [f for _, f in l]
        return l


def ingest_if_missing(db, videos):
    needed = [video.path for video in videos if not db.has_table(video.path)]
    if len(needed) > 0:
        _, failed = db.ingest_videos([(p, p) for p in needed])
        assert (len(failed) == 0)


def shape(l):
    if type(l) is list or type(l) is tuple:
        return 'list({})'.format(shape(l[0]))
    else:
        return type(l).__name__


def par_for(f, l, process=False, workers=None, progress=True):
    Pool = ProcessPoolExecutor if process else ThreadPoolExecutor
    with Pool(max_workers=mp.cpu_count() if workers is None else workers) as executor:
        if progress:
            return list(tqdm(executor.map(f, l), total=len(l)))
        else:
            return list(executor.map(f, l))


def par_filter(f, l, **kwargs):
    return [x for x, b in zip(l, par_for(f, l, **kwargs)) if b]


def make_scanner_db(kube=False):
    if kube:
        ip = sp.check_output(
            '''
    kubectl get pods -l 'app=scanner-master' -o json | \
    jq '.items[0].spec.nodeName' -r | \
    xargs -I {} kubectl get nodes/{} -o json | \
    jq '.status.addresses[] | select(.type == "ExternalIP") | .address' -r
    ''',
            shell=True).strip()

        port = sp.check_output(
            '''
    kubectl get svc/scanner-master -o json | \
    jq '.spec.ports[0].nodePort' -r
    ''',
            shell=True).strip()

        master = '{}:{}'.format(ip, port)

        return Database(master=master, start_cluster=False)

    else:
        return Database()


class Timer:
    def __init__(self, s):
        self.s = s
        log.debug('-- START: {}'.format(s))

    def __enter__(self):
        self.start = now()

    def __exit__(self, a, b, c):
        t = int(now() - self.start)
        log.debug('-- END: {} -- {:02d}:{:02d}:{:02d}'.format(self.s, t / 3600, t / 60, t % 60))


PICKLE_CACHE_DIR = '/app/.cache'
PICKLE_METHOD = 'marshal'
NUM_CHUNKS = 8


class PickleCache:
    def __init__(self):
        if not os.path.isdir(PICKLE_CACHE_DIR):
            os.mkdir(PICKLE_CACHE_DIR)

    def _fname(self, k, i, method):
        ext = 'msl' if method == 'marshal' else 'pkl'
        return '{}/{}_{}.{}'.format(PICKLE_CACHE_DIR, k, i, ext)

    def has(self, k, i=0, method=PICKLE_METHOD):
        return os.path.isfile(self._fname(k, i, method))

    def set(self, k, v, method=PICKLE_METHOD):
        def save_chunk((i, v)):
            with open(self._fname(k, i, method), 'wb') as f:
                if method == 'marshal':
                    marshal.dump(v, f)
                else:
                    pickler = pickle.Pickler(f, pickle.HIGHEST_PROTOCOL)
                    pickler.fast = 1  # https://stackoverflow.com/a/15108940/356915
                    pickler.dump(v)

        with Timer('Saving to cache: {}'.format(k)):
            gc.disable()  # https://stackoverflow.com/a/36699998/356915
            if isinstance(v, list) and len(v) >= NUM_CHUNKS:
                n = len(v)
                chunk_size = int(math.ceil(float(n) / NUM_CHUNKS))
                par_for(
                    save_chunk,
                    [(i, v[(i * chunk_size):((i + 1) * chunk_size)]) for i in range(NUM_CHUNKS)],
                    progress=False,
                    workers=1)
            else:
                save_chunk((0, v))
            gc.enable()

    def get(self, k, fn=None, force=False, method=PICKLE_METHOD):
        if not self.has(k, 0, method) or force:
            if fn is not None:
                v = fn()
                self.set(k, v, method)
                return v
            else:
                raise Exception('Missing cache key {}'.format(k))

        def load_chunk(i):
            with open(self._fname(k, i, method), 'rb') as f:
                if method == 'marshal':
                    return marshal.load(f)
                else:
                    return pickle.load(f)

        with Timer('Loading from cache: {}'.format(k)):
            gc.disable()
            if self.has(k, 1, method):
                loaded = flatten(
                    par_for(load_chunk, range(NUM_CHUNKS), workers=NUM_CHUNKS, progress=False))
            else:
                loaded = load_chunk(0)
            gc.enable()

        return loaded


pcache = PickleCache()


class QuerySetMixin:
    def explain(self):
        # TODO(wcrichto): doesn't work for queries with strings
        cursor = connections[self.db].cursor()
        cursor.execute('EXPLAIN ANALYZE %s' % str(self.query))
        print("\n".join(([t for (t, ) in cursor.fetchall()])))

    def print_sql(self):
        q = str(self.query)
        print(sqlparse.format(q, reindent=True))

    def exists(self):
        try:
            next(self)
            return True
        except self.model.DoesNotExist:
            return False

    def save_to_csv(self, name):
        meta = self.model._meta
        with connection.cursor() as cursor:
            cursor.execute("COPY ({}) TO '{}' CSV DELIMITER ',' HEADER".format(
                str(self.query), '/app/pg/{}.csv'.format(name)))


QuerySet.__bases__ += (QuerySetMixin, )


class BulkUpdateManagerMixin:
    def batch_create(self, objs, batch_size=1000):
        for i in tqdm(range(0, len(objs), batch_size)):
            self.bulk_create(objs[i:(i + batch_size)])

    def bulk_create_copy(self, objects):
        meta = self.model._meta
        keys = [
            f.attname for f in meta._get_fields(reverse=False)
            if not isinstance(f, models.ManyToManyField)
        ]
        fname = '/app/rows.csv'
        log.debug('Creating CSV')
        with open(fname, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(keys)
            max_id = self.all().aggregate(Max('id'))['id__max']
            id = max_id + 1 if max_id is not None else 0
            for obj in tqdm(objects):
                obj['id'] = id
                writer.writerow([obj[k] for k in keys])
                id += 1

        table = meta.db_table
        log.debug('Writing to database')
        with connection.cursor() as cursor:
            cursor.execute("COPY {} ({}) FROM '{}' DELIMITER ',' CSV HEADER".format(
                table, ', '.join(keys), fname))
            cursor.execute("SELECT setval('{}_id_seq', {}, false)".format(table, id))

        os.remove(fname)
        log.debug('Done!')


BulkUpdateManager.__bases__ += (BulkUpdateManagerMixin, )


def model_repr(model):
    def field_repr(field):
        return '{}: {}'.format(field.name, getattr(model, field.name))

    return '{}({})'.format(
        model.__class__.__name__,
        ', '.join([
            field_repr(field) for field in model._meta.get_fields(include_hidden=False)
            if not field.is_relation
        ]))


models.Model.__repr__ = model_repr


def make_montage(video,
                 frames,
                 output_path=None,
                 bboxes=None,
                 width=1600,
                 num_cols=8,
                 workers=16,
                 target_height=None,
                 progress=False):
    target_width = width / num_cols

    def load_frame((video, n, bboxes)):
        while True:
            try:
                r = requests.get(
                    'http://frameserver:7500/fetch', params={
                        'path': video.path,
                        'frame': n,
                    })
                break
            except requests.ConnectionError:
                pass
        img = cv2.imdecode(np.fromstring(r.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise Exception("Bad frame {} for {}".format(n, video.path))
        for bbox in bboxes:
            img = cv2.rectangle(
                img, (int(bbox['bbox_x1'] * img.shape[1]), int(bbox['bbox_y1'] * img.shape[0])),
                (int(bbox['bbox_x2'] * img.shape[1]), int(bbox['bbox_y2'] * img.shape[0])),
                (0, 0, 255), 8)
        th = int(img.shape[0] *
                 (target_width / float(img.shape[1]))) if target_height is None else target_height
        return cv2.resize(img, (target_width, th))

    bboxes = bboxes or [[] for _ in range(len(frames))]
    videos = video if isinstance(video, list) else [video for _ in range(len(frames))]
    imgs = par_for(load_frame, zip(videos, frames, bboxes), progress=progress, workers=workers)
    target_height = imgs[0].shape[0]
    num_rows = int(math.ceil(float(len(imgs)) / num_cols))

    montage = np.zeros((num_rows * target_height, width, 3), dtype=np.uint8)
    for row in range(num_rows):
        for col in range(num_cols):
            i = row * num_cols + col
            if i >= len(imgs):
                break
            img = imgs[i]
            montage[row * target_height:(row + 1) * target_height, col * target_width:(
                col + 1) * target_width, :] = img
        else:
            continue
        break

    if output_path is not None:
        cv2.imwrite(output_path, montage)
    else:
        return montage


def _get_frame((videos, fps, start, i, kwargs)):
    return make_montage(videos, [int(math.ceil(v.fps)) / fps * i + start for v in videos], **kwargs)


def make_montage_video(videos, start, end, output_path, **kwargs):
    def gcd(a, b):
        return gcd(b, a % b) if b else a

    fps = reduce(gcd, [int(math.ceil(v.fps)) for v in videos])

    first = _get_frame((videos, fps, start, 0, kwargs))
    vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps,
                          (first.shape[1], first.shape[0]))

    frames = par_for(
        _get_frame, [(videos, fps, start, i, kwargs) for i in range(end - start)],
        workers=8,
        process=True)
    for frame in tqdm(frames):
        vid.write(frame)

    vid.release()


def gather(l, idx):
    return [l[i] for i in idx]


def gather2(l, idx):
    return [l[i][j] for i, j in idx]


# https://mathieularose.com/how-not-to-flatten-a-list-of-lists-in-python/
def flatten(l):
    return list(itertools.chain.from_iterable(l))


def collect(l, kfn):
    d = defaultdict(list)
    for x in l:
        d[kfn(x)].append(x)
    return dict(d)


class Break(Exception):
    pass
