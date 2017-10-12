from django.db import models, connection
from django.db.models.base import ModelBase
from django_bulk_update.manager import BulkUpdateManager
from scannerpy import ProtobufGenerator, Config
import sys
import numpy as np
import json
import warnings
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
        try:
            v.ParseFromString(value)
            return v
        except TypeError:
            return None

    def deconstruct(self):
        name, path, args, kwargs = super(ProtoField, self).deconstruct()
        return name, path, [self._proto] + args, kwargs


def CharField(*args, **kwargs):
    return models.CharField(*args, max_length=MAX_STR_LEN, **kwargs)


current_dataset = None
datasets = {}


class Dataset(object):
    def __init__(self, name):
        self.name = name
        if name not in datasets:
            datasets[name] = self

    def __enter__(self):
        global current_dataset
        current_dataset = datasets[self.name]

    def __exit__(self, type, val, traceback):
        global current_dataset
        current_dataset = None


def ForeignKey(model, name, **kwargs):
    return models.ForeignKey(model, related_query_name=name.lower(), **kwargs)


class BaseMeta(ModelBase):
    def __new__(cls, name, bases, attrs):
        global current_dataset
        base = bases[0]
        is_base_class = base is models.Model

        if is_base_class:
            if not 'Meta' in attrs:
                class Meta:
                    abstract = True
                attrs['Meta'] = Meta

            attrs['objects'] = BulkUpdateManager()
            child_name = name

        else:
            if base.__name__ == 'Frame':
                attrs['video'] = ForeignKey(current_dataset.Video, name)
                class Meta:
                    unique_together = ('video', 'number')
                attrs['Meta'] = Meta

            child_name = '{}_{}'.format(current_dataset.name, name)

        new_cls = super(BaseMeta, cls).__new__(cls, child_name, bases, attrs)

        if not is_base_class:
            setattr(current_dataset, name, new_cls)
            if base.__name__ == 'Concept':
                register_concept(name)

        return new_cls


def register_concept(name):
    global current_datset

    class Meta:
        unique_together = ('concept', 'frame', 'labeler')

    cls_name = '{}Instance'.format(name)
    inst = type(cls_name, (Instance, ), {
        '__module__': current_dataset.Video.__module__,
        'concept': ForeignKey(getattr(current_dataset, name), cls_name),
        'frame': ForeignKey(current_dataset.Frame, cls_name),
        'labeler': ForeignKey(current_dataset.Labeler, cls_name),
        'Meta': Meta
    })

    class Meta:
        unique_together = ('labeler', 'instance')
    instance_cls_name = cls_name
    cls_name = '{}Features'.format(name)
    type(cls_name, (Features, ), {
        '__module__': current_dataset.Video.__module__,
        'labeler': ForeignKey(current_dataset.Labeler, cls_name),
        'instance': ForeignKey(inst, cls_name),
        'Meta': Meta,
        '_datasetName' : current_dataset.name,
        '_conceptName' : cls_name,
        '_instanceName' : instance_cls_name
    })


class Video(models.Model):
    __metaclass__ = BaseMeta
    path = CharField(db_index=True)
    num_frames = models.IntegerField()
    fps = models.FloatField()
    width = models.IntegerField()
    height = models.IntegerField()


class Frame(models.Model):
    __metaclass__ = BaseMeta
    number = models.IntegerField(db_index=True)

    def save(self, *args, **kwargs):
        self.validate_unique()
        super(Frame, self).save(*args, **kwargs)

class Labeler(models.Model):
    __metaclass__ = BaseMeta
    name = CharField(db_index=True)


class Concept(models.Model):
    __metaclass__ = BaseMeta


class Model(models.Model):
    __metaclass__ = BaseMeta

class Instance(models.Model):
    __metaclass__ = BaseMeta

    bbox = ProtoField(proto.BoundingBox)

    def save(self, *args, **kwargs):
        self.validate_unique()
        super(Instance, self).save(*args, **kwargs)

class Features(models.Model):
    __metaclass__ = BaseMeta

    features = models.BinaryField()

    def save(self, *args, **kwargs):
        self.validate_unique()
        super(Features, self).save(*args, **kwargs)

    @classmethod
    def getTempFeatureModel(cls, instance_ids):
        with connection.cursor() as cursor:
            with Dataset(cls._datasetName):
                col_def = ''.join([', `distto_{}` double precision NULL'.format(inst) for inst in instance_ids])
                cursor.execute("CREATE TEMPORARY TABLE {} (`id` integer NOT NULL PRIMARY KEY, `features` longblob NOT NULL, `instance_id` int(11) NOT NULL, `labeler_id` int(11) NOT NULL {})".format(cls._meta.db_table+"temp", col_def))
                current_dataset = datasets[cls._datasetName]
                cls_name = cls._conceptName+"Temp" 
                model_params = {'__module__' : cls.__module__,
                        #'features': models.BinaryField(),
                        'labeler': ForeignKey(current_dataset.Labeler, cls_name),
                        'instance': ForeignKey(getattr(current_dataset,cls._instanceName), cls_name)}
                for instance_id in instance_ids:
                    model_params['distto_{}'.format(instance_id)] = models.FloatField(null=True)
                with warnings.catch_warnings():
                    # we ignore the warning indicating we are reloading a model
                    # because it is a temporary table
                    warnings.simplefilter("ignore")
                    tempmodel = type(cls_name, (Features, ), model_params)
                testfeatures = {}
                for i in cls.objects.filter(instance_id__in=instance_ids).all():
                    testfeatures[i.instance_id] = np.array(json.loads(i.features))
                it = cls.objects.all()
                batch_size = 1000
                batch = []
                feature_batch = []
                for feat in it:
                    newfeat = tempmodel()
                    newfeat.id = feat.id
                    newfeat.features = feat.features
                    newfeat.labeler_id = feat.labeler_id
                    newfeat.instance_id = feat.instance_id
                    #TODO better distance computation
                    for i in instance_ids:
                        setattr(newfeat, 'distto_{}'.format(i), np.sum(np.square(np.array(json.loads(feat.features))-testfeatures[i])))
                    batch.append(newfeat)
                    if len(batch) == batch_size:
                        tempmodel.objects.bulk_create(batch)
                        batch = []
                tempmodel.objects.bulk_create(batch)
                return tempmodel


                    

    @classmethod
    def dropTempFeatureModel(cls):
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS {};".format(cls._meta.db_table+"temp"))

            

class ModelDelegator:
    def __init__(self, name):
        self._dataset = datasets[name]

    def datasets(self):
        return datasets.keys()

    def __getattr__(self, k):
        return getattr(self._dataset, k)
