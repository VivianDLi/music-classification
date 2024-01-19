from dataset import Mazurkas, RondoDB, Covers80
from models import (
    FullModel,
    SmallModel,
    QuantizedFullModel,
    QuantizedSmallModel,
    PrunedFullModel,
    PrunedSmallModel,
    QuantizedPrunedFullModel,
    QuantizedPrunedSmallModel,
)

import time
from datetime import datetime
import tensorflow as tf

import logging

logging.basicConfig(
    filename=f"./logs/timing_logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log",
    encoding="utf-8",
    level=logging.INFO,
)

if __name__ == "__main__":
    # testing for GPU
    logging.info(tf.config.list_physical_devices("GPU"))

    mazurkas = Mazurkas()
    covers80 = Covers80()
    rondodb = RondoDB()

    full_model = FullModel(rondodb.num_classes, True)
    small_model = SmallModel(rondodb.num_classes, True)
    quantized_full_model = QuantizedFullModel(rondodb.num_classes, True)
    quantized_small_model = QuantizedSmallModel(rondodb.num_classes, True)
    pruned_full_model = PrunedFullModel(rondodb.num_classes, True)
    pruned_small_model = PrunedSmallModel(rondodb.num_classes, True)
    q_p_full_model = QuantizedPrunedFullModel(rondodb.num_classes, True)
    q_p_small_model = QuantizedPrunedSmallModel(rondodb.num_classes, True)

    # example model train
    full_model.train(rondodb)
    full_model.convert()
