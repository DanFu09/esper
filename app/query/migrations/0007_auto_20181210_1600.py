# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2018-12-10 16:00
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('query', '0006_facelandmarks'),
    ]

    operations = [
        migrations.CreateModel(
            name='ShotScale',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
            ],
        ),
        migrations.AddField(
            model_name='frame',
            name='shot_scale',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='query.ShotScale'),
        ),
    ]
