from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from keras.callbacks import BackupAndRestore, CSVLogger, TensorBoard
from keras.models import load_model

from dataset import Dataset
import keras_models as km
from constants import NUM_EPOCHS, NUM_PRUNING_EPOCHS


class Model:
    def __init__(self, name, model: tf.keras.Model, from_file: bool = False):
        self.name = name
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

        if from_file:
            self.model.save(f"./models/{self.name}.keras")
            km.get_representation_model(self.model).save(
                f"./models/{self.name}-representation.keras"
            )
            print("Model saved.")

    def train(
        self,
        dataset: Dataset,
        checkpoint_used=False,
    ):
        """Executes a training loop for the model, logging results and saving the trained model when done. This training loop uses SparseCategoricalCrossEntropy loss and the Adam optimizer with a 0.001 learning rate. Override this method to implement pruning.

        Args:
            dataset (Dataset): training dataset to use.
            checkpoint_used (bool, optional): if the training is resuming from a checkpoint. Defaults to False.
        """
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )

        checkpoint_path = f"./logs/checkpoints/{self.name}"
        logging_path = f"./logs/training/{self.name}.log"
        tensorboard_path = f"./logs/tensorboard/{self.name}"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = BackupAndRestore(checkpoint_path)
        logging_callback = CSVLogger(logging_path, append=checkpoint_used)
        tensorboard_callback = TensorBoard(tensorboard_path)

        self.model.fit(
            dataset.get_train_dataset(),
            callbacks=[
                checkpoint_callback,
                logging_callback,
                tensorboard_callback,
            ],
            epochs=NUM_EPOCHS,
        )
        print("Training Finished.")

        self.model.save(f"./models/{self.name}.keras")
        km.get_representation_model(self.model).save(
            f"./models/{self.name}-representation.keras"
        )
        print("Model saved.")

    def convert(self, dataset: Dataset = None):
        """Converts the current model to TensorflowLite and saves it (.tflite). Requires the model to be saved beforehand as a .keras file in the models folder. Override this method to implement quantization.

        Args:
            dataset (Dataset, optional): provides a representative dataset to use for conversion if necessary. Defaults to None.
        """
        model_path = f"./models/{self.name}-representation.keras"
        tflite_model_path = f"./models/{self.name}.tflite"
        if Path(model_path).exists():
            model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        print("Conversion finished.")

        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print("Model saved.")

    def evaluate(self, dataset: Dataset):
        """
        Given a query q and references {rn}, extract CQT descriptors, and get network representations for each.
        Then, for all possible query/reference pairs in the test dataset, compute the cosine similarities between their network representations.

        Use later to compute a ranking list for each query to evaluate the mean average precision, precision @ 10, and mean rank of first correctly identified cover for evaluation.

        Returns:
            results (dict(str, dict(str, float))): dictionary of query songs to another dictionary of reference songs to cosine similarity distance.
        """
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(
            model_path=f"./models/{self.name}.tflite"
        )
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        results = {}

        reference_ds, query_ds = dataset.get_test_datasets()
        for query_cqt, query_label in query_ds:
            for reference_cqt, reference_label in reference_ds:
                interpreter.resize_tensor_input(input_index, query_cqt.shape)
                interpreter.allocate_tensors()
                if "quantized" in self.name:
                    scale, offset = interpreter.get_input_details()[0][
                        "quantization"
                    ]
                    query_cqt = tf.cast(query_cqt / scale - offset, tf.int8)
                    reference_cqt = tf.cast(
                        reference_cqt / scale - offset, tf.int8
                    )
                interpreter.set_tensor(input_index, query_cqt)
                interpreter.invoke()
                query_data = interpreter.get_tensor(output_index)

                interpreter.resize_tensor_input(
                    input_index, reference_cqt.shape
                )
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_index, reference_cqt)
                interpreter.invoke()
                reference_data = interpreter.get_tensor(output_index)

                distance = tf.tensordot(query_data, reference_data, 0) / (
                    tf.norm(query_data) * tf.norm(reference_data)
                )
                
                string_query_label = query_label.numpy()[0].decode('utf-8')
                string_reference_label = reference_label.numpy()[0].decode('utf-8')
                if string_query_label not in results:
                    results[string_query_label] = {}
                if string_reference_label not in results[string_query_label]:
                    results[string_query_label][string_reference_label] = distance.numpy()
        with open(f"./results/{self.name}.pkl", "wb") as f:
            pickle.dump(results, f)
        return results

    def get_binary(self):
        """
        Honestly, just a stub as a reminder on how to extract a hex dump from a model file (in linux).

        First, install tflite_tools:
            !git clone https://github.com/eliberis/tflite-tools.git tflite_tools

        Then, see information about the model's size and structure to see if it fits on device:
            !python tflite_tools/tflite_tools.py -i cnn_model.tflite --calc-macs --calc-size

        Then, export the model to a .txt (hex dump):
            !apt-get -qq install xxd
            !echo "Exporting model. Model size (in bytes):"
            !stat --printf="%s" lstm_model.tflite
            !xxd -i lstm_model.tflite > lstm_model.txt # xxd is just used to create a hex dump from model file
        """
        return NotImplemented


class QuantizedModel(Model):
    def convert(self, dataset: Dataset):
        """Converts the current model to TFLite, performs 8-bit integer quantization, and saves it (.tflite). Requires the model to be saved beforehand as a .keras file in the models folder.

        Args:
            dataset (Dataset): a representative dataset (the validation dataset) to use for quantization.
        """

        def _representative_dataset_gen():
            for sample, _ in dataset.get_val_dataset():
                yield [np.expand_dims(sample, axis=0)]

        model_path = f"./models/{self.name}-representation.keras"
        tflite_model_path = f"./models/{self.name}.tflite"
        if Path(model_path).exists():
            model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        print("Conversion finished.")

        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print("Model saved.")


class PrunedModel(Model):
    def prune(self, dataset: Dataset, checkpoint_used=False):
        """Executes a training loop for the model, logging results and saving the trained model when done. This training loop uses SparseCategoricalCrossEntropy loss and the Adam optimizer with a 0.001 learning rate. Pruning is implemented as fine-tuning with a polynomial decay schedule from 0.2 to 0.8 sparsity on convolutional layers 4-10.

        Args:
            dataset (Dataset): training dataset to use.
            checkpoint_used (bool, optional): if the training is resuming from a checkpoint. Defaults to False.
        """

        def _apply_pruning_to_layers(layer):
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.2,
                    final_sparsity=0.8,
                    begin_step=0,
                    end_step=-1,
                )
            }
            if "conv" in layer.name and layer.name.split("_")[1] in [
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ]:
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer, **pruning_params
                )
            return layer

        model_for_pruning = tf.keras.models.clone_model(
            self.model, clone_function=_apply_pruning_to_layers
        )

        checkpoint_path = f"./logs/checkpoints/{self.name}"
        logging_path = f"./logs/training/{self.name}.log"
        tensorboard_path = f"./logs/tensorboard/{self.name}"
        pruning_path = f"./logs/pruning/{self.name}"
        Path(pruning_path).mkdir(parents=True, exist_ok=True)

        checkpoint_callback = BackupAndRestore(checkpoint_path)
        logging_callback = CSVLogger(logging_path, append=True)
        tensorboard_callback = TensorBoard(tensorboard_path)

        model_for_pruning.fit(
            dataset.get_train_dataset(),
            callbacks=[
                checkpoint_callback,
                logging_callback,
                tensorboard_callback,
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=pruning_path),
            ],
            epochs=NUM_PRUNING_EPOCHS,
            validation_data=dataset.get_val_dataset(),
            validation_batch_size=1,
        )
        self.model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        print("Pruning Finished.")

        self.model.save(f"./models/{self.name}.keras")
        km.get_representation_model(self.model).save(
            f"./models/{self.name}-representation.keras"
        )
        print("Model saved.")


class FullModel(Model):
    def __init__(self, n_classes):
        super().__init__("full_model", km.get_full_model(n_classes))


class SmallModel(Model):
    def __init__(self, n_classes):
        super().__init__("small_model", km.get_small_model(n_classes))


class QuantizedFullModel(QuantizedModel):
    def __init__(self, n_classes, from_file: bool):
        if from_file:
            model = load_model(f"./models/full_model.keras")
        else:
            model = km.get_full_model(n_classes)
        super().__init__("quantized_full_model", model)


class PrunedFullModel(PrunedModel):
    def __init__(self, n_classes, from_file):
        if from_file:
            model = load_model(f"./models/full_model.keras")
        else:
            model = km.get_full_model(n_classes)
        super().__init__("pruned_full_model", model)


class QuantizedPrunedFullModel(QuantizedModel, PrunedModel):
    def __init__(self, n_classes, from_file: bool):
        if from_file:
            model = load_model(f"./models/full_model.keras")
        else:
            model = km.get_full_model(n_classes)
        super().__init__("quantized_pruned_full_model", model)


class QuantizedSmallModel(QuantizedModel):
    def __init__(self, n_classes, from_file: bool):
        if from_file:
            model = load_model(f"./models/small_model.keras")
        else:
            model = km.get_small_model(n_classes)
        super().__init__("quantized_small_model", model)


class PrunedSmallModel(PrunedModel):
    def __init__(self, n_classes, from_file: bool):
        if from_file:
            model = load_model(f"./models/small_model.keras")
        else:
            model = km.get_small_model(n_classes)
        super().__init__("pruned_small_model", model)


class QuantizedPrunedSmallModel(QuantizedModel, PrunedModel):
    def __init__(self, n_classes, from_file: bool):
        if from_file:
            model = load_model(f"./models/small_model.keras")
        else:
            model = km.get_small_model(n_classes)
        super().__init__("quantized_pruned_small_model", model)
