# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-12-24 09:30
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0020_auto_20181218_1020'),
    ]

    operations = [
        migrations.CreateModel(
            name='PoseMeta',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Frame')),
                ('labeler', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='query.Labeler')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
