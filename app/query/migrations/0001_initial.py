# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-04-13 13:10
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion
import query.base_models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='default_Face',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bbox_x1', models.FloatField()),
                ('bbox_x2', models.FloatField()),
                ('bbox_y1', models.FloatField()),
                ('bbox_y2', models.FloatField()),
                ('bbox_score', models.FloatField()),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model, query.base_models.BoundingBox),
        ),
        migrations.CreateModel(
            name='default_FaceFeatures',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('features', models.BinaryField()),
                ('distto', models.FloatField(null=True)),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facefeatures', to='query.default_Face')),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model, query.base_models.Features),
        ),
        migrations.CreateModel(
            name='default_FaceGender',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.default_Face')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='default_Frame',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField(db_index=True)),
            ],
        ),
        migrations.CreateModel(
            name='default_Gender',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='default_Labeler',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='default_Person',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='person', to='query.default_Frame')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='default_PersonTrack',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='persontrack', to='query.default_Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='default_Pose',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('keypoints', models.BinaryField()),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='pose', to='query.default_Labeler')),
                ('person', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='pose', to='query.default_Person')),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model, query.base_models.Pose),
        ),
        migrations.CreateModel(
            name='default_Video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(db_index=True, max_length=256)),
                ('num_frames', models.IntegerField()),
                ('fps', models.FloatField()),
                ('width', models.IntegerField()),
                ('height', models.IntegerField()),
                ('has_captions', models.BooleanField(default=False)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Channel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Commercial',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Face',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bbox_x1', models.FloatField()),
                ('bbox_x2', models.FloatField()),
                ('bbox_y1', models.FloatField()),
                ('bbox_y2', models.FloatField()),
                ('bbox_score', models.FloatField()),
                ('background', models.BooleanField(default=False)),
                ('is_host', models.BooleanField(default=False)),
            ],
            bases=(models.Model, query.base_models.BoundingBox),
        ),
        migrations.CreateModel(
            name='tvnews_FaceFeatures',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('features', models.BinaryField()),
                ('distto', models.FloatField(null=True)),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facefeatures', to='query.tvnews_Face')),
            ],
            bases=(models.Model, query.base_models.Features),
        ),
        migrations.CreateModel(
            name='tvnews_FaceGender',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.tvnews_Face')),
            ],
        ),
        migrations.CreateModel(
            name='tvnews_FaceIdentity',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='faceidentity', to='query.tvnews_Face')),
            ],
        ),
        migrations.CreateModel(
            name='tvnews_Frame',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('number', models.IntegerField(db_index=True)),
            ],
        ),
        migrations.CreateModel(
            name='tvnews_Gender',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Identity',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256, null=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Labeler',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Person',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='person', to='query.tvnews_Frame')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Pose',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('keypoints', models.BinaryField()),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='pose', to='query.tvnews_Labeler')),
                ('person', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='pose', to='query.tvnews_Person')),
            ],
            bases=(models.Model, query.base_models.Pose),
        ),
        migrations.CreateModel(
            name='tvnews_Segment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
                ('polarity', models.FloatField(null=True)),
                ('subjectivity', models.FloatField(null=True)),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='segment', to='query.tvnews_Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Shot',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
                ('in_commercial', models.BooleanField(default=False)),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='shot', to='query.tvnews_Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Show',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Speaker',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('min_frame', models.IntegerField()),
                ('max_frame', models.IntegerField()),
                ('gender', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='speaker', to='query.tvnews_Gender')),
                ('identity', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_query_name='speaker', to='query.tvnews_Identity')),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='speaker', to='query.tvnews_Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Tag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Thing',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
        ),
        migrations.CreateModel(
            name='tvnews_ThingType',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_Video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('path', models.CharField(db_index=True, max_length=256)),
                ('num_frames', models.IntegerField()),
                ('fps', models.FloatField()),
                ('width', models.IntegerField()),
                ('height', models.IntegerField()),
                ('has_captions', models.BooleanField(default=False)),
                ('time', models.DateTimeField()),
                ('commercials_labeled', models.BooleanField(default=False)),
                ('srt_extension', models.CharField(max_length=256)),
                ('channel', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='video', to='query.tvnews_Channel')),
                ('show', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='video', to='query.tvnews_Show')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='tvnews_VideoTag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='videotag', to='query.tvnews_Tag')),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='videotag', to='query.tvnews_Video')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='tvnews_thing',
            name='type',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='thing', to='query.tvnews_ThingType'),
        ),
        migrations.AddField(
            model_name='tvnews_speaker',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='speaker', to='query.tvnews_Video'),
        ),
        migrations.AddField(
            model_name='tvnews_shot',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='shot', to='query.tvnews_Video'),
        ),
        migrations.AddField(
            model_name='tvnews_segment',
            name='things',
            field=models.ManyToManyField(related_query_name='segment', to='query.tvnews_Thing'),
        ),
        migrations.AddField(
            model_name='tvnews_segment',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='segment', to='query.tvnews_Video'),
        ),
        migrations.AddField(
            model_name='tvnews_identity',
            name='thing',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_query_name='identity', to='query.tvnews_Thing'),
        ),
        migrations.AddField(
            model_name='tvnews_frame',
            name='tags',
            field=models.ManyToManyField(related_query_name='frame', to='query.tvnews_Tag'),
        ),
        migrations.AddField(
            model_name='tvnews_frame',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='frame', to='query.tvnews_Video'),
        ),
        migrations.AddField(
            model_name='tvnews_faceidentity',
            name='identity',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='faceidentity', to='query.tvnews_Thing'),
        ),
        migrations.AddField(
            model_name='tvnews_faceidentity',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='faceidentity', to='query.tvnews_Labeler'),
        ),
        migrations.AddField(
            model_name='tvnews_facegender',
            name='gender',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.tvnews_Gender'),
        ),
        migrations.AddField(
            model_name='tvnews_facegender',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.tvnews_Labeler'),
        ),
        migrations.AddField(
            model_name='tvnews_facefeatures',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facefeatures', to='query.tvnews_Labeler'),
        ),
        migrations.AddField(
            model_name='tvnews_face',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='face', to='query.tvnews_Labeler'),
        ),
        migrations.AddField(
            model_name='tvnews_face',
            name='person',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='face', to='query.tvnews_Person'),
        ),
        migrations.AddField(
            model_name='tvnews_face',
            name='shot',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_query_name='face', to='query.tvnews_Shot'),
        ),
        migrations.AddField(
            model_name='tvnews_commercial',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='commercial', to='query.tvnews_Labeler'),
        ),
        migrations.AddField(
            model_name='tvnews_commercial',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='commercial', to='query.tvnews_Video'),
        ),
        migrations.AddField(
            model_name='default_persontrack',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='persontrack', to='query.default_Video'),
        ),
        migrations.AddField(
            model_name='default_person',
            name='tracks',
            field=models.ManyToManyField(related_query_name='person', to='query.default_PersonTrack'),
        ),
        migrations.AddField(
            model_name='default_frame',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='frame', to='query.default_Video'),
        ),
        migrations.AddField(
            model_name='default_facegender',
            name='gender',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.default_Gender'),
        ),
        migrations.AddField(
            model_name='default_facegender',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facegender', to='query.default_Labeler'),
        ),
        migrations.AddField(
            model_name='default_facefeatures',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='facefeatures', to='query.default_Labeler'),
        ),
        migrations.AddField(
            model_name='default_face',
            name='labeler',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='face', to='query.default_Labeler'),
        ),
        migrations.AddField(
            model_name='default_face',
            name='person',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_query_name='face', to='query.default_Person'),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_thing',
            unique_together=set([('name', 'type')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_pose',
            unique_together=set([('labeler', 'person')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_frame',
            unique_together=set([('video', 'number')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_faceidentity',
            unique_together=set([('labeler', 'face')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_facegender',
            unique_together=set([('labeler', 'face')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_facefeatures',
            unique_together=set([('labeler', 'face')]),
        ),
        migrations.AlterUniqueTogether(
            name='tvnews_face',
            unique_together=set([('labeler', 'person')]),
        ),
        migrations.AlterUniqueTogether(
            name='default_frame',
            unique_together=set([('video', 'number')]),
        ),
    ]
