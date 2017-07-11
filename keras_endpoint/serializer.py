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
