# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-07-08 08:20
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('keras_endpoint', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='KerasModelWeights',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('weight_file', models.CharField(max_length=256, unique=True)),
                ('name', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='keras_endpoint.KerasModel')),
            ],
        ),
    ]
