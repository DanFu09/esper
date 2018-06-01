from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from timeit import default_timer as now
import django.db.models as models
from concurrent.futures import ThreadPoolExecutor
from django.views.decorators.csrf import csrf_exempt
import sys
import json
import os
import tempfile
import subprocess as sp
import shlex
import math
import pysrt
import requests
import enum
from storehouse import StorageConfig, StorageBackend
import traceback
from pprint import pprint
from esper.stdlib import *
from esper.prelude import *

ESPER_ENV = os.environ.get('ESPER_ENV')
BUCKET = os.environ.get('BUCKET')
DATA_PATH = os.environ.get('DATA_PATH')

if ESPER_ENV == 'google':
    storage_config = StorageConfig.make_gcs_config(BUCKET)
else:
    storage_config = StorageConfig.make_posix_config()
storage = StorageBackend.make_from_config(storage_config)


# Prints and flushes (necessary for gunicorn logs)
def _print(*args):
    pprint((*args))
    sys.stdout.flush()

# Renders home page
def index(request):
    return render(request, 'index.html', {'globals': json.dumps(esper_js_globals())})


# Run search routine
def search(request):
    params = json.loads(request.body.decode('utf-8'))

    def make_error(err):
        return JsonResponse({'error': err})

    try:
        fprint('Executing')
        ############### vvv DANGER -- REMOTE CODE EXECUTION vvv ###############
        _globals = {}
        _locals = {}
        for k in globals():
            _globals[k] = globals()[k]
        for k in locals():
            _locals[k] = locals()[k]
        exec ((params['code']), _globals, _locals)
        result = _locals['FN']()
        ############### ^^^ DANGER -- REMOTE CODE EXECUTION ^^^ ###############

        if not isinstance(result, dict):
            return make_error(
                'Result must be a dict {{result, count, type}}, received type {}'.format(
                    type(result)))

        if not isinstance(result['result'], list):
            return make_error('Result must be a frame list')

        return JsonResponse({'success': result_with_metadata(result)})

    except Exception:
        return make_error(traceback.format_exc())


# Get distinct values in schema
def schema(request):
    params = json.loads(request.body.decode('utf-8'))
    m = ModelDelegator(params['dataset'])

    cls = getattr(m, params['cls_name'])
    result = [
        r[params['field']]
        for r in cls.objects.values(params['field']).distinct().order_by(params['field'])[:100]
    ]
    try:
        json.dumps(result)
    except TypeError as e:
        return JsonResponse({'error': str(e)})

    return JsonResponse({'result': result})


# Convert captions in SRT format to WebVTT (for displaying in web UI)
def srt_to_vtt(s):
    subs = pysrt.from_string(s)
    subs.shift(seconds=-5)  # Seems like TV news captions are delayed by a few seconds

    entry_fmt = '{position}\n{start} --> {end}\n{text}'

    def fmt_time(t):
        return '{:02d}:{:02d}:{:02d}.{:03d}'.format(t.hours, t.minutes, t.seconds, t.milliseconds)

    entries = [
        entry_fmt.format(
            position=i, start=fmt_time(sub.start), end=fmt_time(sub.end), text=sub.text)
        for i, sub in enumerate(subs)
    ]

    return '\n\n'.join(['WEBVTT'] + entries)


# Get subtitles for video
def subtitles(request):
    video_id = request.GET.get('video')
    video = Video.objects.get(id=video_id)

    srt = video.srt_extension
    sub_path = '/app/data/subs/{}.{}.srt'.format(video.item_name(), srt)

    s = open(sub_path, 'rb').read().decode('utf-8')
    vtt = srt_to_vtt(s)

    return HttpResponse(vtt, content_type="text/vtt")


def save_frame_labels(m, groups):
    face_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-face')
    gender_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-gender')
    identity_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-identity')
    labeled_tag, _ = m.Tag.objects.get_or_create(name='handlabeled-face:labeled')
    verified_tag, _ = m.Tag.objects.get_or_create(name='tmp-verified')

    all_frames = [(elt['video'], elt['min_frame'], elt['objects']) for group in groups
                  for elt in group['elements']]

    frame_ids = [
        m.Frame.objects.get(video_id=vid, number=frame_num).id
        for (vid, frame_num, faces) in all_frames
    ]
    m.Frame.tags.through.objects.filter(
        tvnews_frame__id__in=frame_ids, tvnews_tag=labeled_tag).delete()
    m.Frame.tags.through.objects.filter(
        tvnews_frame__id__in=frame_ids, tvnews_tag=verified_tag).delete()
    m.Face.objects.filter(person__frame__id__in=frame_ids, labeler=face_labeler).delete()

    all_tags = []
    all_people = []
    all_faces = []
    all_genders = []
    all_identities = []
    for (vid, frame_num, faces) in all_frames:
        frame = m.Frame.objects.get(video_id=vid, number=frame_num)
        all_tags.append(
            m.Frame.tags.through(tvnews_frame_id=frame.id, tvnews_tag_id=labeled_tag.id))
        all_tags.append(
            m.Frame.tags.through(tvnews_frame_id=frame.id, tvnews_tag_id=verified_tag.id))
        for face in faces:
            all_people.append(m.Person(frame=frame))

            face_params = {
                'bbox_score': 1.0,
                'labeler': face_labeler,
                'person_id': None,
                'shot_id': None,
                'background': face['background']
            }
            for k in ['bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2']:
                face_params[k] = face[k]
            all_faces.append(m.Face(**face_params))

            all_genders.append(
                m.FaceGender(face_id=None, gender_id=face['gender_id'], labeler=gender_labeler))

            if 'identity_id' in face:
                all_identities.append(
                    m.FaceIdentity(
                        face_id=None, identity_id=face['identity_id'], labeler=identity_labeler))
            else:
                all_identities.append(None)

    m.Frame.tags.through.objects.bulk_create(all_tags)

    m.Person.objects.bulk_create(all_people)

    for (p, f) in zip(all_people, all_faces):
        f.person_id = p.id
    m.Face.objects.bulk_create(all_faces)

    for (f, g, i) in zip(all_faces, all_genders, all_identities):
        g.face_id = f.id
        if i is not None:
            i.face_id = f.id
    m.FaceGender.objects.bulk_create(all_genders)
    m.FaceIdentity.objects.bulk_create([i for i in all_identities if i is not None])


def save_speaker_labels(m, groups):
    audio_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-audio')
    segment_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-audio:labeled')
    speakers = []
    segments = []
    for group in groups:
        elements = group['elements']
        frame_nums = []
        for e in elements:
            speakers.append(
                m.Speaker(
                    labeler=audio_labeler,
                    video_id=e['video'],
                    min_frame=e['min_frame'],
                    max_frame=e['max_frame'],
                    gender_id=e['gender_id']))
            frame_nums.append(e['min_frame'])
            frame_nums.append(e['max_frame'])

        frame_nums.sort()
        segments.append(
            m.Segment(
                labeler=segment_labeler,
                video_id=elements[0]['video'],
                min_frame=frame_nums[0],
                max_frame=frame_nums[-1]))

    m.Speaker.objects.bulk_create(speakers)
    m.Segment.objects.bulk_create(segments)


def save_single_identity_labels(m, groups):
    identity_labeler, _ = m.Labeler.objects.get_or_create(name='handlabeled-identity')

    last_obj = groups[-1]['elements'][0]['objects'][0]
    if not 'identity_id' in last_obj:
        raise Exception("Missing identity on last object")

    identity = m.Thing.objects.get(id=last_obj['identity_id'])

    face_ids = [o['id'] for g in groups for e in g['elements'] for o in e['objects']]
    identities = [
        m.FaceIdentity(face_id=face_id, identity=identity, labeler=identity_labeler)
        for face_id in face_ids if face_id != -1
    ]

    m.FaceIdentity.objects.filter(face_id__in=face_ids, labeler=identity_labeler).delete()
    m.FaceIdentity.objects.bulk_create(identities)


class LabelMode(enum.IntEnum):
    DEFAULT = 0
    SINGLE_IDENTITY = 1


# Register frames as labeled
def labeled(request):
    try:
        params = json.loads(request.body.decode('utf-8'))
        m = ModelDelegator(params['dataset'])

        label_mode = int(params['label_mode'])
        groups = params['groups']
        ty = groups[0]['type']
        if label_mode == LabelMode.DEFAULT:
            if ty == 'flat':
                save_frame_labels(m, groups)
            else:
                save_speaker_labels(m, groups)

        elif label_mode == LabelMode.SINGLE_IDENTITY:
            save_single_identity_labels(m, groups)

        else:
            raise Exception('Invalid label mode: {}'.format(label_mode))

        return JsonResponse({'success': True})
    except Exception:
        return JsonResponse({'success': False, 'error': traceback.format_exc()})
