from abc import ABC
from functools import partial

import tensorflow as tf

from constants import BATCH_SIZE, VAL_SPLIT
from utils import process_cqt_dataset, process_cropped_scaled_dataset


class Dataset(ABC):
    """Using cross-entropy loss between t and y where t is a one-hot encoding vector for which song x belongs to
    Input X is a CQT descriptor calculated using Librosa (resampled to 22050Hz, 12 bins per octave, hop size 512 Hann window) -> followed by 20-point mean filter and 20x downsample (feature rate about 2Hz or 84 x T matrix)

    For each batch, sample recordings from the training set and extract CQTs.
    For each CQT, randomly crop three subsequences of length 200, 300, and 400 for training
    Randomly augment each sequence with a changing factor (uniform between 0.7 and 1.3) to simulate tempo changes
    """

    def __init__(self, dataset_path, original_sr, num_classes, name):
        self.name = name
        self.num_classes = num_classes
        # get training data from folders
        (
            audio_train_ds,
            audio_val_ds,
        ) = tf.keras.utils.audio_dataset_from_directory(
            directory=dataset_path,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT,
            ragged=True,
            seed=42,
            subset="both",
        )
        audio_test_ds = audio_val_ds.shard(num_shards=2, index=0)
        audio_val_ds = audio_val_ds.shard(num_shards=2, index=1)

        # convert audio training data into CQTs based on Yang et. al. 2020
        dataset_manip_function = partial(process_cqt_dataset, original_sr)
        cqt_train_ds = audio_train_ds.map(
            map_func=dataset_manip_function,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        self.val_ds = audio_val_ds.map(
            map_func=dataset_manip_function,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        self.test_ds = audio_test_ds.map(
            map_func=dataset_manip_function,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # randomly crop subsequences, augment, and combine datasets for the train dataset
        self.train_ds = process_cropped_scaled_dataset(cqt_train_ds)
        
        # group the test dataset into pairs for evaluation

    def get_train_dataset(self):
        return self.train_ds

    def get_val_dataset(self):
        return self.val_ds

    def get_test_dataset(self):
        return self.test_ds


class Mazurkas(Dataset):
    def __init__(self):
        super().__init__("./dataset/Mazurkas", 48000, 49, "mazurkas")


class RondoDB(Dataset):
    def __init__(self):
        super().__init__("./dataset/RondoDB100", 48000, 100, "rondodb")


class Covers80(Dataset):
    def __init__(self):
        super().__init__("./dataset/Covers80", 16000, 80, "covers80")
