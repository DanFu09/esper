# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2019-01-12 19:19
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0028_shot_cinematic'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='ignore_film',
            field=models.BooleanField(default=False),
        ),
    ]
