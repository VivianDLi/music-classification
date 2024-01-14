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
import tensorflow as tf

if __name__ == "__main__":
    # testing for GPU
    print(tf.config.list_physical_devices("GPU"))

    mazurkas = Mazurkas()
    covers80 = Covers80()
    rondodb = RondoDB()

    start_time = time.time()
    eval_times = dict()
    # full model
    full_model = FullModel(rondodb.num_classes)
    full_model.train(rondodb)
    train_time = time.time()
    eval_times["train"] = train_time - start_time
    full_model.model.summary()
    full_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - train_time
    print("Evaluation times (Full Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # small model
    small_model = SmallModel(rondodb.num_classes)
    small_model.train(rondodb)
    train_time = time.time()
    eval_times["train"] = train_time - start_time
    small_model.model.summary()
    small_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - train_time
    print("Evaluation times (Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # quantized full model
    quantized_full_model = QuantizedFullModel(rondodb.num_classes)
    quantized_full_model.model.summary()
    quantized_full_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - start_time
    print("Evaluation times (Quantized Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # quantized small model
    quantized_small_model = QuantizedSmallModel(rondodb.num_classes)
    quantized_small_model.model.summary()
    quantized_small_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - start_time
    print("Evaluation times (Quantized Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # pruned full model
    pruned_full_model = PrunedFullModel(rondodb.num_classes)
    pruned_full_model.prune(rondodb)
    prune_time = time.time()
    eval_times["prune"] = prune_time - start_time
    pruned_full_model.model.summary()
    pruned_full_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - prune_time
    print("Evaluation times (Pruned Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # pruned small model
    pruned_small_model = PrunedSmallModel(rondodb.num_classes)
    pruned_small_model.prune(rondodb)
    prune_time = time.time()
    eval_times["prune"] = prune_time - start_time
    pruned_small_model.model.summary()
    pruned_small_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - prune_time
    print("Evaluation times (Pruned Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # quantized pruned full model
    q_p_full_model = QuantizedPrunedFullModel(rondodb.num_classes)
    q_p_full_model.prune(rondodb)
    prune_time = time.time()
    eval_times["prune"] = prune_time - start_time
    q_p_full_model.model.summary()
    q_p_full_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - prune_time
    print("Evaluation times (qp Full Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    start_time = time.time()
    eval_times = dict()
    # quantized pruned small model
    q_p_small_model = QuantizedPrunedSmallModel(rondodb.num_classes)
    q_p_small_model.prune(rondodb)
    prune_time = time.time()
    eval_times["prune"] = prune_time - start_time
    q_p_small_model.model.summary()
    q_p_small_model.convert(rondodb)
    convert_time = time.time()
    eval_times["convert"] = convert_time - prune_time
    print("Evaluation times (qp Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")

    convert_time = time.time()
    full_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    full_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Full Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
      
    convert_time = time.time()    
    small_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    small_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
    
    convert_time = time.time()    
    quantized_full_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    quantized_full_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Quantized Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
    
    convert_time = time.time()    
    quantized_small_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    quantized_small_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Quantized Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
    
    convert_time = time.time()    
    pruned_full_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    pruned_full_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Pruned Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
    
    convert_time = time.time()    
    pruned_small_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    pruned_small_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (Pruned Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
        
    convert_time = time.time()
    q_p_full_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    q_p_full_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (qp Full Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")
        
    convert_time = time.time()
    q_p_small_model.evaluate(covers80)
    covers_eval_time = time.time()
    eval_times["covers80"] = covers_eval_time - convert_time
    q_p_small_model.evaluate(mazurkas)
    mazurkas_eval_time = time.time()
    eval_times["mazurkas"] = mazurkas_eval_time - covers_eval_time
    print("Evaluation times (qp Small Model):")
    for k in eval_times:
        print(f"{k}: {eval_times[k]}s")