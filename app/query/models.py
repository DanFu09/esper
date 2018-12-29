from django.db import models
from . import base_models as base
import math
import numpy as np
import tempfile
import subprocess as sp

class Identity(models.Model):
    name = base.CharField(unique=True)

class Genre(models.Model):
    name = base.CharField(unique=True)

class Director(models.Model):
    name = base.CharField(unique=True)

class Actor(models.Model):
    name = base.CharField(unique=True)

class Character(models.Model):
    name = base.CharField(unique=True)

class Producer(models.Model):
    name = base.CharField(unique=True)

class Writer(models.Model):
    name = base.CharField(unique=True)

class Cinematographer(models.Model):
    name = base.CharField(unique=True)

class Editor(models.Model):
    name = base.CharField(unique=True)

class Country(models.Model):
    name = base.CharField(unique=True)

class Language(models.Model):
    name = base.CharField(unique=True)

class Video(base.Video):
    name = base.CharField()
    year = models.IntegerField()
    title = base.CharField()
    release_month = base.CharField(null=True)
    release_day = models.IntegerField(null=True)
    genres = models.ManyToManyField(Genre)    
    directors = models.ManyToManyField(Director)
    producers = models.ManyToManyField(Producer)
    writers = models.ManyToManyField(Writer)
    cinematographers = models.ManyToManyField(Cinematographer)
    editors = models.ManyToManyField(Editor)

    box_office = models.IntegerField(null=True)
    budget = models.IntegerField(null=True)
    budget_currency = base.CharField(null=True)
    
    countries = models.ManyToManyField(Country)    
    languages = models.ManyToManyField(Language)

    rating = models.FloatField(null=True)
    ratingCount = models.FloatField(null=True)

    small_dataset = models.BooleanField(default=False)
    decode_errors = models.BooleanField(default=False)

    def get_stride(self):
        return int(math.ceil(self.fps) / 2)

    def item_name(self):
        return '.'.join(self.path.split('/')[-1].split('.')[:-1])

    def url(self, duration='1d'):
        fetch_cmd = 'PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHONPATH gsutil signurl -d {} /app/service-key.json gs://esper/{} ' \
                    .format(duration, self.path)
        url = sp.check_output(fetch_cmd, shell=True).decode('utf-8').split('\n')[1].split('\t')[-1]
        return url

class CharacterActor(models.Model):
    video = models.ForeignKey(Video)
    character = models.ForeignKey(Character)
    actor = models.ForeignKey(Actor)
    primary = models.BooleanField(default=False)

class Tag(models.Model):
    name = base.CharField()


class VideoTag(models.Model):
    video = models.ForeignKey(Video)
    tag = models.ForeignKey(Tag)


class ShotScale(models.Model):
    name = base.CharField()


class Frame(base.Frame):
    tags = models.ManyToManyField(Tag)
    shot_boundary = models.BooleanField(default=False)
    brightness = models.FloatField(null=True)
    contrast = models.FloatField(null=True)
    sharpness = models.FloatField(null=True)
    saturation = models.FloatField(null=True)
    shot_scale = models.ForeignKey(ShotScale, default=1)


class Labeler(base.Labeler):
    data_path = base.CharField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True, null=True, blank=True)


Labeled = base.Labeled(Labeler)

class Gender(models.Model):
    name = base.CharField()

Track = base.Track(Labeler)


class Shot(Track):
    pass

class Pose(Labeled, base.Pose, models.Model):
    frame = models.ForeignKey(Frame)

class PoseMeta(Labeled, models.Model):
    frame = models.ForeignKey(Frame)

class Face(Labeled, base.BoundingBox, models.Model):
    frame = models.ForeignKey(Frame)
    shot = models.ForeignKey(Shot, null=True)
    background = models.BooleanField(default=False)
    blurriness = models.FloatField(null=True)
    probability = models.FloatField(default=1.)

    class Meta:
        unique_together = ('labeler', 'frame', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2')


class FaceGender(Labeled, models.Model):
    face = models.ForeignKey(Face)
    gender = models.ForeignKey(Gender)
    probability = models.FloatField(default=1.)

    class Meta:
        unique_together = ('labeler', 'face')


class FaceIdentity(Labeled, models.Model):
    face = models.ForeignKey(Face)
    identity = models.ForeignKey(Identity)
    probability = models.FloatField(default=1.)

    class Meta:
        unique_together = ('labeler', 'face')


class FaceCharacterActor(Labeled, models.Model):
    face = models.ForeignKey(Face)
    characteractor = models.ForeignKey(CharacterActor)
    probability = models.FloatField(default=1.)

    class Meta:
        unique_together = ('labeler', 'face')


class FaceFeatures(Labeled, base.Features, models.Model):
    face = models.ForeignKey(Face)

    class Meta:
        unique_together = ('labeler', 'face')


class ScannerJob(models.Model):
    name = base.CharField()


class Object(base.BoundingBox, models.Model):
    frame = models.ForeignKey(Frame)
    label = models.IntegerField()
    probability = models.FloatField()

class FaceLandmarks(Labeled, base.FaceLandmarks, models.Model):
    face = models.ForeignKey(Face)

