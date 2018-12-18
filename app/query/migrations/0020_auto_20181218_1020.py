# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-12-18 10:20
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0019_character_characteractor'),
    ]

    operations = [
        migrations.CreateModel(
            name='FaceCharacterActor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('probability', models.FloatField(default=1.0)),
                ('characteractor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.CharacterActor')),
                ('face', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Face')),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Labeler')),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='facecharacteractor',
            unique_together=set([('labeler', 'face')]),
        ),
    ]