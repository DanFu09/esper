from django.db import models
import numpy as np
from scannerpy import ProtobufGenerator, Config

cfg = Config()
proto = ProtobufGenerator(cfg)
MAX_STR_LEN = 256


class ProtoField(models.BinaryField):
    def __init__(self, proto, *args, **kwargs):
        self._proto = proto
        super(ProtoField, self).__init__(*args, **kwargs)

    def get_prep_value(self, value):
        return value.SerializeToString()

    def from_db_value(self, value, expression, connection, context):
        return self.to_python(value)

    def to_python(self, value):
        v = self._proto()
        v.ParseFromString(value)
        return v

    def deconstruct(self):
        name, path, args, kwargs = super(ProtoField, self).deconstruct()
        return name, path, [self._proto] + args, kwargs


class Video(models.Model):
    path = models.CharField(max_length=MAX_STR_LEN)
    num_frames = models.IntegerField()
    fps = models.FloatField()
    width = models.IntegerField()
    height = models.IntegerField()

    def audio_path(self):
        return 'assets/audio/{}.aac'.format(self.id)

    def detected_labelset(self):
        return LabelSet.objects.filter(video=self, name="detected").get()

    def handlabeled_labelset(self):
        return LabelSet.objects.filter(video=self, name="handlabeled").get()


class LabelSet(models.Model):
    video = models.ForeignKey(Video)
    name = models.CharField(max_length=MAX_STR_LEN)


class Detector(models.Model):
    name = models.CharField(max_length=MAX_STR_LEN)


class DetectorsUsed(models.Model):
    video = models.ForeignKey(LabelSet)
    detector = models.ForeignKey(Detector)


class Identity(models.Model):
    name = models.CharField(max_length=MAX_STR_LEN)
    classifier = models.BinaryField()
    cohesion = models.FloatField()
    labelset = models.ForeignKey(LabelSet)


class Track(models.Model):
    first_frame = models.IntegerField()
    last_frame = models.IntegerField()
    is_male = models.NullBooleanField()
    labelset = models.ForeignKey(LabelSet)


class Face(models.Model):
    frame = models.IntegerField()
    identity = models.ForeignKey(Identity, null=True, on_delete=models.SET_NULL)
    track = models.ForeignKey(Track, null=True, on_delete=models.SET_NULL)
    bbox = ProtoField(proto.BoundingBox)
    features = models.TextField() # So we can use json.dumps to store a list.
    gender = models.CharField(max_length=2, default='0')   # M, F or U.
    labelset = models.ForeignKey(LabelSet)
