import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    Dense,
    MaxPool2D,
    GlobalMaxPool2D,
    BatchNormalization,
    ReLU,
)
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.callbacks import BackupAndRestore, CSVLogger, TensorBoard
from keras.models import load_model

import numpy as np

from abc import ABC
from pathlib import Path
from dataset import Dataset
from constants import NUM_EPOCHS, NUM_PRUNING_EPOCHS


class FullModel(tf.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = Conv2D(32, (12, 3), (1, 1), name="conv_1")
        self.conv2 = Conv2D(64, (13, 3), (1, 2), name="conv_2")
        self.conv3 = Conv2D(64, (13, 3), (1, 1), name="conv_3")
        self.conv4 = Conv2D(64, (3, 3), (1, 2), name="conv_4")
        self.conv5 = Conv2D(128, (3, 3), (1, 1), name="conv_5")
        self.conv6 = Conv2D(128, (3, 3), (1, 2), name="conv_6")
        self.conv7 = Conv2D(256, (3, 3), (1, 1), name="conv_7")
        self.conv8 = Conv2D(256, (3, 3), (1, 2), name="conv_8")
        self.conv9 = Conv2D(512, (3, 3), (1, 1), name="conv_9")
        self.conv10 = Conv2D(512, (3, 3), (1, 2), name="conv_10")
        self.max_pool = MaxPool2D((1, 2), (1, 2))
        self.adaptive_pool = GlobalMaxPool2D(
            data_format="channels_last", keep_dims=True
        )
        self.fc_representation = Dense(300)
        self.fc_classification = Dense(n_classes, activation="softmax")

        self.batch_norm = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.representation(x)
        return self.fc_classification(x)

    def representation(self, x):
        x = self.relu(self.batch_norm(self.conv1(x)))
        x = self.relu(self.batch_norm(self.conv2(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv3(x)))
        x = self.relu(self.batch_norm(self.conv4(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv5(x)))
        x = self.relu(self.batch_norm(self.conv6(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv7(x)))
        x = self.relu(self.batch_norm(self.conv8(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv9(x)))
        x = self.relu(self.batch_norm(self.conv10(x)))
        x = self.adaptive_pool(x)
        return self.fc_representation(x)


class SmallModel(tf.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = DepthwiseConv2D(
            (12, 3), (1, 1), depth_multiplier=32, name="conv_1"
        )
        self.conv2 = DepthwiseConv2D(
            (13, 3), (1, 2), depth_multiplier=64, name="conv_2"
        )
        self.conv3 = DepthwiseConv2D(
            (13, 3), (1, 1), depth_multiplier=64, name="conv_3"
        )
        self.conv4 = DepthwiseConv2D(
            (3, 3), (1, 2), depth_multiplier=64, name="conv_4"
        )
        self.conv5 = DepthwiseConv2D(
            (3, 3), (1, 1), depth_multiplier=128, name="conv_5"
        )
        self.conv6 = DepthwiseConv2D(
            (3, 3), (1, 2), depth_multiplier=128, name="conv_6"
        )
        self.conv7 = DepthwiseConv2D(
            (3, 3), (1, 1), depth_multiplier=256, name="conv_7"
        )
        self.conv8 = DepthwiseConv2D(
            (3, 3), (1, 2), depth_multiplier=256, name="conv_8"
        )
        self.conv9 = DepthwiseConv2D(
            (3, 3), (1, 1), depth_multiplier=512, name="conv_9"
        )
        self.conv10 = DepthwiseConv2D(
            (3, 3), (1, 2), depth_multiplier=512, name="conv_10"
        )
        self.max_pool = MaxPool2D((1, 2), (1, 2))
        self.adaptive_pool = GlobalMaxPool2D(
            data_format="channels_last", keep_dims=True
        )
        self.fc_representation = Dense(300)
        self.fc_classification = Dense(n_classes, activation="softmax")

        self.batch_norm = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.representation(x)
        return self.fc_classification(x)

    def representation(self, x):
        x = self.relu(self.batch_norm(self.conv1(x)))
        x = self.relu(self.batch_norm(self.conv2(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv3(x)))
        x = self.relu(self.batch_norm(self.conv4(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv5(x)))
        x = self.relu(self.batch_norm(self.conv6(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv7(x)))
        x = self.relu(self.batch_norm(self.conv8(x)))
        x = self.max_pool(x)
        x = self.relu(self.batch_norm(self.conv9(x)))
        x = self.relu(self.batch_norm(self.conv10(x)))
        x = self.adaptive_pool(x)
        return self.fc_representation(x)


class Model(ABC):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = SparseCategoricalCrossentropy()

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

        checkpoint_path = f"./logs/checkpoints/{self.name}-{dataset.name}"
        logging_path = f"./logs/training/{self.name}-{dataset.name}.log"
        tensorboard_path = f"./logs/tensorboard/{self.name}-{dataset.name}"
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
        print("Model saved.")

    def convert(self, dataset: Dataset = None):
        """Converts the current model to TensorflowLite and saves it (.tflite). Requires the model to be saved beforehand as a .keras file in the models folder. Override this method to implement quantization.

        Args:
            dataset (Dataset, optional): provides a representative dataset to use for conversion if necessary. Defaults to None.
        """
        model_path = f"./models/{self.name}-{dataset.name}.keras"
        tflite_model_path = f"./models/{self.name}-{dataset.name}.tflite"
        if Path(model_path).exists():
            self.model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        print("Conversion finished.")

        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print("Model saved.")

    def evaluate(self, dataset: Dataset):
        """
        After training, extract music representations and test on the test set
        Given a query q and references {rn} (rest of the test set), extract CQT descriptors, and get network representations for each
        Compute the pairwise cosine similarities between the query and the references and return a ranking list.
        Evaluate the mean average precision, precision at 10, and mean rank of first correctly identified cover for evaluation (cover means the songs are in the same dataset folder).

        Args:
            data (_type_): _description_
        """
        pass

    def export(self):
        # @tf.function
        # def __call__(self, x):
        #     # If they pass a string, load the file and decode it.
        #     if x.dtype == tf.string:
        #     x = tf.io.read_file(x)
        #     x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
        #     x = tf.squeeze(x, axis=-1)
        #     x = x[tf.newaxis, :]

        #     x = get_spectrogram(x)
        #     result = self.model(x, training=False)

        #     class_ids = tf.argmax(result, axis=-1)
        #     class_names = tf.gather(label_names, class_ids)
        #     return {'predictions':result,
        #             'class_ids': class_ids,
        #             'class_names': class_names}
        pass


class QuantizedModel(Model):
    def convert(self, dataset: Dataset):
        """Converts the current model to TensorflowLite, performs 8-bit integer quantization, and saves it (.tflite). Requires the model to be saved beforehand as a .keras file in the models folder.

        Args:
            dataset (Dataset): a representative dataset (the validation dataset) to use for quantization.
        """

        def _representative_dataset_gen():
            for sample, _ in dataset.get_val_dataset():
                yield [np.expand_dims(sample, axis=0)]

        model_path = f"./models/{self.name}-{dataset.name}.keras"
        tflite_model_path = f"./models/{self.name}-{dataset.name}.tflite"
        if Path(model_path).exists():
            self.model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
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
    def train(self, dataset: Dataset, checkpoint_used=False):
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

        super().train(dataset.get_train_dataset())
        model_for_pruning = tf.keras.models.clone_model(
            self.model, clone_function=_apply_pruning_to_layers
        )

        checkpoint_path = f"./logs/checkpoints/{self.name}-{dataset.name}"
        logging_path = f"./logs/training/{self.name}-{dataset.name}.log"
        tensorboard_path = f"./logs/tensorboard/{self.name}-{dataset.name}"
        pruning_path = f"./logs/pruning/{self.name}-{dataset.name}"
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
        )
        self.model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        print("Pruning Finished.")

        self.model.save(f"./models/{self.name}-{dataset.name}.keras")
        print("Model saved.")


class QuantizedFullModel(QuantizedModel):
    def __init__(self, n_classes):
        super().__init__("quantized_full_model", FullModel(n_classes))


class PrunedFullModel(PrunedModel):
    def __init__(self, n_classes):
        super().__init__("pruned_full_model", FullModel(n_classes))


class QuantizedPrunedFullModel(QuantizedModel, PrunedModel):
    def __init__(self, n_classes):
        super().__init__("quantized_pruned_full_model", FullModel(n_classes))


class QuantizedSmallModel(QuantizedModel):
    def __init__(self, n_classes):
        super().__init__("quantized_small_model", SmallModel(n_classes))


class PrunedSmallModel(PrunedModel):
    def __init__(self, n_classes):
        super().__init__("pruned_small_model", SmallModel(n_classes))


class QuantizedPrunedSmallModel(QuantizedModel, PrunedModel):
    def __init__(self, n_classes):
        super().__init__("quantized_pruned_small_model", SmallModel(n_classes))
