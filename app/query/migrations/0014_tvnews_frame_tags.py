# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-12-26 17:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0013_tvnews_shot'),
    ]

    operations = [
        migrations.AddField(
            model_name='tvnews_frame',
            name='tags',
            field=models.ManyToManyField(related_query_name='frame', to='query.tvnews_Tag'),
        ),
    ]