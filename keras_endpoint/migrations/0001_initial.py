# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-07-07 13:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='KerasModel',
            fields=[
                ('name', models.CharField(max_length=256, primary_key=True, serialize=False)),
                ('date_created', models.DateTimeField(auto_now_add=True)),
                ('definition', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
