from models import *
from rest_framework import serializers, viewsets, routers

class KerasModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = KerasModel
        fields = ('name','definition')

class KerasModelWeightsSerializer(serializers.ModelSerializer):
    class Meta:
        model = KerasModelWeights
        fields = ('model','weight_name')

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ('name',)

class PredictTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictTask
        fields = ('name','weight','completed')
        depth = 2

class TrainTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainTask
        fields = ('name','weight','completed','config','dataset')
        depth = 2
