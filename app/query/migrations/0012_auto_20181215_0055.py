# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-12-15 00:55
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0011_auto_20181215_0053'),
    ]

    operations = [
        migrations.CreateModel(
            name='Genre',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
        ),
        migrations.AddField(
            model_name='video',
            name='genres',
            field=models.ManyToManyField(to='query.Genre'),
        ),
    ]
