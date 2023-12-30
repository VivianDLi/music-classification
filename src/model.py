import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def create_model():
        pass

    def train(data):
        pass

    def evaluate(data):
        pass


class QuantizedModel:
    def quantize():
        pass


class PrunedModel:
    def prune():
        pass


class FullModel(Model):
    def create_model():
        return tf.keras.Sequential([])


class SmallerModel(Model):
    def create_model():
        return tf.keras.Sequential([])


class QuantizedModel(FullModel, QuantizedModel):
    pass


class PrunedModel(FullModel, PrunedModel):
    pass


class QuantizedPrunedModel(QuantizedModel, PrunedModel, FullModel):
    pass


class QuantizedSmallerModel(SmallerModel, QuantizedModel):
    pass


class PrunedSmallerModel(SmallerModel, PrunedModel):
    pass


class QuantizedPrunedSmallerModel(SmallerModel, QuantizedModel, PrunedModel):
    pass
