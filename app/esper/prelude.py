from scannerpy import ProtobufGenerator, Config, Database, Job, DeviceType, ColumnType, ScannerException
from storehouse import StorageConfig, StorageBackend
from query.base_models import Track, BoundingBox
from django.db import connections, connection
from django.db.models.query import QuerySet
from django.db.models import Min, Max, Count, F, Q, OuterRef, Subquery, Sum, Avg, Func
from django.db.models.functions import Cast, Extract
from django.utils import timezone
from django_bulk_update.manager import BulkUpdateManager
from IPython.core.getipython import get_ipython
from timeit import default_timer as now
from functools import reduce
from typing import Dict
from pprint import pprint

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
import pickle as pickle
import marshal
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import requests
import cv2
import itertools
import shutil
import tempfile
import random
import socket
from contextlib import contextmanager
from collections import defaultdict

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
    import beakerx

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
        choice = input().lower()
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


def bbox_area2(f):
    return (f['bbox_x2'] - f['bbox_x1']) * (f['bbox_y2'] - f['bbox_y1'])


def bbox_iou2(f1, f2):
    x1 = max(f1['bbox_x1'], f2['bbox_x1'])
    x2 = min(f1['bbox_x2'], f2['bbox_x2'])
    y1 = max(f1['bbox_y1'], f2['bbox_y1'])
    y2 = min(f1['bbox_y2'], f2['bbox_y2'])

    if x1 > x2 or y1 > y2: return 0

    intersection = (x2 - x1) * (y2 - y1)
    return intersection / (bbox_area2(f1) + bbox_area2(f2) - intersection)


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

    for l in list(d.values()):
        l.sort(key=fn_sort)

    if output_dict:
        return dict(d)
    else:
        l = sorted(iter(list(d.items())), key=itemgetter(0))
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


class ScannerWrapper:
    def __init__(self, kube=False, multiworker=False):
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

            self.db = Database(master=master, start_cluster=False)

        else:
            workers = ['localhost:{}'.format(5002 + i)
                       for i in range(mp.cpu_count() // 8)] if multiworker else None
            # import scannerpy.libscanner as bindings
            # import scanner.metadata_pb2 as metadata_types
            # params = metadata_types.MachineParameters()
            # params.ParseFromString(bindings.default_machine_params())
            # params.num_load_workers = 2
            # params.num_save_workers = 2
            self.db = Database(
                #machine_params=params.SerializeToString(),
                workers=workers)

    def sql_config(self):
        return self.db.protobufs.SQLConfig(
            adapter='postgres',
            hostaddr=socket.gethostbyname('db'),
            port=5432,
            dbname='esper',
            user=os.environ['DJANGO_DB_USER'],
            password=os.environ['DJANGO_DB_PASSWORD'])

    def sql_sink(self, cls, insert=True):
        from query.models import ScannerJob
        return lambda input: self.db.sinks.SQL(
            config=self.sql_config(),
            input=input,
            table=cls._meta.db_table,
            job_table=ScannerJob._meta.db_table,
            insert=insert)

    # Remove videos that don't have a table or have already been processed by the pipeline
    def filter_videos(self, videos, pipeline):
        suffix = pipeline.job_suffix
        assert suffix is not None

        processed = set([
            t['name'] for t in ScannerJob.objects.filter(name__contains=suffix).values('name')])

        return [
            v for v in videos if self._db.has_table(v.path) and not '{}_{}'.format(v.path, suffix) in processed
        ]



class Timer:
    def __init__(self, s, run=True):
        self._s = s
        self._run = run
        if run:
            log.debug('-- START: {} --'.format(s))

    def __enter__(self):
        self.start = now()

    def __exit__(self, a, b, c):
        t = int(now() - self.start)
        if self._run:
            log.debug('-- END: {} -- {:02d}:{:02d}:{:02d}'.format(self._s, int(t / 3600), int(
                t / 60),
                                                                  int(t) % 60))


CACHE_DIR = '/app/.cache'
DEFAULT_CACHE_METHOD = 'pickle'
NUM_CHUNKS = 8


class PyCache:
    def __init__(self):
        if not os.path.isdir(CACHE_DIR):
            os.mkdir(CACHE_DIR)

    def _fname(self, k, i, method):
        exts = {'pickle': 'pkl', 'marshal': 'msl', 'numpy': 'bin'}
        return '{}/{}_{}.{}'.format(CACHE_DIR, k, i, exts[method])

    def has(self, k, i=0, method=DEFAULT_CACHE_METHOD):
        return os.path.isfile(self._fname(k, i, method))

    def set(self, k, v, method=DEFAULT_CACHE_METHOD):
        def save_chunk(args):
            (i, v) = args
            with open(self._fname(k, i, method), 'wb') as f:
                if method == 'marshal':
                    marshal.dump(v, f)
                elif method == 'numpy':
                    for arr in v:
                        f.write(arr.tobytes())
                elif method == 'pickle':
                    pickler = pickle.Pickler(f, pickle.HIGHEST_PROTOCOL)
                    pickler.fast = 1  # https://stackoverflow.com/a/15108940/356915
                    pickler.dump(v)
                else:
                    raise Exception("Invalid cache method {}".format(method))

        with Timer('Saving to cache: {}'.format(k)):
            gc.disable()  # https://stackoverflow.com/a/36699998/356915
            if (isinstance(v, list) or isinstance(v, tuple)) and len(v) >= NUM_CHUNKS:
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

    def get(self, k, fn=None, force=False, method=DEFAULT_CACHE_METHOD, **kwargs):
        if not (all([self.has(k2, 0, m2) for k2, m2 in zip(k, method)])
                if isinstance(k, tuple) else self.has(k, 0, method)) or force:
            if fn is not None:
                v = fn()
                if isinstance(k, tuple):
                    for (k2, v2, m2) in zip(k, v, method):
                        self.set(k2, v2, m2)
                else:
                    self.set(k, v, method)
                return v
            else:
                raise Exception('Missing cache key {}'.format(k))

        if isinstance(k, tuple):
            return tuple([self.get(k2, method=m2, **kwargs) for k2, m2 in zip(k, method)])

        else:

            def load_chunk(i):
                with open(self._fname(k, i, method), 'rb') as f:
                    if method == 'marshal':
                        return marshal.load(f)
                    elif method == 'numpy':
                        dtype = kwargs['dtype']
                        size = np.dtype(dtype).itemsize * kwargs['length']
                        byte_str = f.read()
                        assert len(byte_str) % size == 0
                        return [
                            np.frombuffer(byte_str[i:i + size], dtype=dtype)
                            for i in range(0, len(byte_str), size)
                        ]
                    elif method == 'pickle':
                        return pickle.load(f, encoding='latin1')
                    else:
                        raise Exception("Invalid cache method {}".format(method))

            with Timer('Loading from cache: {}'.format(k)):
                gc.disable()
                if self.has(k, 1, method):
                    loaded = flatten(
                        par_for(
                            load_chunk, list(range(NUM_CHUNKS)), workers=NUM_CHUNKS,
                            progress=False))
                else:
                    loaded = load_chunk(0)
                gc.enable()

            return loaded


pcache = PyCache()


class QuerySetMixin(object):
    def explain(self):
        # TODO(wcrichto): doesn't work for queries with strings
        cursor = connections[self.db].cursor()
        cursor.execute('EXPLAIN ANALYZE %s' % str(self.query))
        print(("\n".join(([t for (t, ) in cursor.fetchall()]))))

    def print_sql(self):
        q = str(self.query)
        print((sqlparse.format(q, reindent=True)))

    def exists(self):
        try:
            next(self)
            return True
        except self.model.DoesNotExist:
            return False

    def values_with(self, *fields):
        return self.values(*([f.name for f in self.model._meta.get_fields()] + list(fields)))

    def save_to_csv(self, name):
        meta = self.model._meta
        with connection.cursor() as cursor:
            cursor.execute("COPY ({}) TO '{}' CSV DELIMITER ',' HEADER".format(
                str(self.query), '/app/pg/{}.csv'.format(name)))

def qs_child_count(qs, fkey_path):
    return Subquery(
        qs.filter(**{fkey_path:OuterRef('pk')}) \
        .values(fkey_path) \
        .annotate(c=Count('*')) \
        .values('c'),
        models.IntegerField())


for key in QuerySetMixin.__dict__:
    if key[:2] == '__':
        continue

    setattr(QuerySet, key, QuerySetMixin.__dict__[key])


def print_sql(self):
    q = str(self.query)
    print((sqlparse.format(q, reindent=True)))


setattr(QuerySet, 'print_sql', print_sql)


def bulk_create_copy(self, objects, table=None):
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
            if table is None:
                obj['id'] = id
                id += 1
            writer.writerow([obj[k] for k in keys])

    log.debug('Writing to database')
    with connection.cursor() as cursor:
        cursor.execute("COPY {} ({}) FROM '{}' DELIMITER ',' CSV HEADER".format(
            table or meta.db_table, ', '.join(keys), fname))
        if table is None:
            cursor.execute("SELECT setval('{}_id_seq', {}, false)".format(table, id))

    os.remove(fname)
    log.debug('Done!')


class BulkUpdateManagerMixin:
    def batch_create(self, objs, batch_size=1000):
        for i in tqdm(list(range(0, len(objs), batch_size))):
            self.bulk_create(objs[i:(i + batch_size)])

    def bulk_create_copy(self, objects):
        return bulk_create_copy(self, objects)


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


def crop(img, bbox):
    [h, w] = img.shape[:2]
    return img[int(bbox.bbox_y1 * h):int(bbox.bbox_y2 * h),
               int(bbox.bbox_x1 * w):int(bbox.bbox_x2 * w)]


def resize(img, w, h):
    th = int(img.shape[0] * (tw / float(img.shape[1]))) if h is None else h
    tw = int(img.shape[1] * (th / float(img.shape[0]))) if w is None else w
    return cv2.resize(img, (tw, th))


def load_frame(video, frame, bboxes):
    while True:
        try:
            r = requests.get(
                'http://frameserver:7500/fetch', params={
                    'path': video.path,
                    'frame': frame,
                })
            break
        except requests.ConnectionError:
            pass
    img = cv2.imdecode(np.fromstring(r.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception("Bad frame {} for {}".format(frame, video.path))
    for bbox in bboxes:
        img = cv2.rectangle(
            img, (int(bbox['bbox_x1'] * img.shape[1]), int(bbox['bbox_y1'] * img.shape[0])),
            (int(bbox['bbox_x2'] * img.shape[1]), int(bbox['bbox_y2'] * img.shape[0])), (0, 0, 255),
            8)

    return img


def face_to_dict(face):
    return {
        'bbox_x1': face.bbox_x1,
        'bbox_x2': face.bbox_x2,
        'bbox_y1': face.bbox_y1,
        'bbox_y2': face.bbox_y2,
    }


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

    bboxes = bboxes or [[] for _ in range(len(frames))]
    videos = video if isinstance(video, list) else [video for _ in range(len(frames))]
    imgs = par_for(
        lambda t: resize(load_frame(*t), target_width, target_height),
        list(zip(videos, frames, bboxes)),
        progress=progress,
        workers=workers)
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


def _get_frame(args):
    (videos, fps, start, i, kwargs) = args
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


# For breaking out of nested loops
class Break(Exception):
    pass


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # cm = cm.T
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm.T
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap or plt.cm.Blues)
    plt.title(('Normalized ' if normalize else '') + title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def readlines(path):
    with open(path, 'r') as f:
        return [s.strip() for s in f.readlines()]


class WithMany:
    def __init__(self, *args):
        self._args = args

    def __enter__(self):
        return tuple([obj.__enter__() for obj in self._args])

    def __exit__(self, *args, **kwargs):
        for obj in self._args:
            obj.__exit__(*args, **kwargs)


@contextmanager
def named_temp_dir(delete=True, **kwargs):
    dir = tempfile.mkdtemp(**kwargs)
    try:
        yield dir
    finally:
        if delete:
            shutil.rmtree(dir)


# https://gist.github.com/howardhamilton/537e13179489d6896dd3
@contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def imshow(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.axis('off')


def caption_search(phrases):
    from query.models import Video
    videos = {v.item_name(): v.id for v in Video.objects.all()}
    r = requests.post('http://localhost:8111/subsearch', json={'phrases': phrases})

    def item_name(path):
        return '.'.join(os.path.basename(path).split('.')[:-2])

    return [{videos[item_name(k)]: v for k, v in d.items()} for d in r.json()]


def caption_count(phrases):
    r = requests.post('http://localhost:8111/subcount', json={'phrases': phrases})
    return r.json()


def mutual_info(p):
    r = requests.post('http://localhost:8111/mutualinfo', json={'phrases': [p]})
    return r.json()


def find_segments(lexicon):
    r = requests.post('http://localhost:8111/findsegments', json={'phrases': lexicon})
    return r.json()


def face_knn(target=None, targets=None, ids=None, k=None,
             min_threshold=-1.0, max_threshold=1000.0,
             not_ids=[], not_id_penalty=0.2):
    r = requests.post(
        'http://localhost:8111/facesearch',
        json={
            'exemplar': target.tolist() if target is not None else [],
            'exemplars': [x.tolist() for x in targets] if targets is not None else [],
            'ids': ids if ids is not None else [],
            'k': k if k is not None else -1,
            'min_threshold': min_threshold,
            'max_threshold': max_threshold,
            'non_targets': not_ids,
            'non_target_penalty': not_id_penalty
        })
    return r.json()


def face_svm(pos_ids, neg_ids=[], neg_samples=1000, pos_samples=500,
             min_threshold=-2.0, max_threshold=2.0):
    r = requests.post(
        'http://localhost:8111/facesearch_svm',
        json={
            'pos_ids': pos_ids,
            'neg_ids': neg_ids,
            'pos_samples': pos_samples,
            'neg_samples': neg_samples,
            'min_threshold': min_threshold,
            'max_threshold': max_threshold
        })
    return r.json()


def face_kmeans(ids, k=25):
    r = requests.post(
        'http://localhost:8111/face_kmeans',
        json={
            'k': k,
            'ids': ids,
        })
    return r.json()


def face_features(self, ids):
    r = requests.post('http://localhost:8111/facefeatures', json={'ids': ids})
    return [np.array(a) for a in r.json()]


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def esper_widget(result, **kwargs):
    from esper.stdlib import result_with_metadata, esper_js_globals
    import esper_jupyter
    if not 'select_mode' in kwargs:
        kwargs['select_mode'] = 1
    return esper_jupyter.EsperWidget(
        result=result_with_metadata(result),
        jsglobals=esper_js_globals(),
        settings=kwargs)
