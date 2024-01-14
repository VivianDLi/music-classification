from glob import glob
from abc import ABC

import tensorflow as tf

from constants import BATCH_SIZE, VAL_SPLIT
from utils import (
    generate_training_dataset_from_audio_dataset,
    generate_training_dataset_from_cqts,
    generate_cqts_from_file_dataset,
)

"""
Split RondoDB100 into train and test (70:30). Train on RondoDB100. Test on RondoDB, Mazurkas and Covers80.
Initially, train on RondoDB and test on Mazurkas and Covers80.

Use Rubinstein recordings for Mazurkas as reference (present for all categories).
Use first recording for Covers80.
"""


class Dataset(ABC):
    def __init__(self, dataset_path, original_sr, num_classes, name):
        self.dataset_path = dataset_path
        self.original_sr = original_sr
        self.name = name
        self.num_classes = num_classes

    def get_train_dataset(self):
        """Returns a train dataset for training ((batch_size * 3, cqt, song_index)) [randomly cropped to fixed subsequences]"""
        return NotImplemented

    def get_val_dataset(self):
        """Returns a validation dataset for testing training accuracy ((batch_size, cqt, song_index) [uncropped])"""
        return NotImplemented

    def get_test_datasets(self):
        """Returns a tuple of a reference dataset ((cqt, song_name) pairs) and a query dataset ((cqt, song_name) pairs)"""
        return NotImplemented


class RondoDB(Dataset):
    """
    The training dataset:
    Using cross-entropy loss between t and y where t is a one-hot encoding vector for which song x belongs to
    Input X is a CQT descriptor calculated using Librosa (resampled to 22050Hz, 12 bins per octave, hop size 512 Hann window) -> followed by 20-point mean filter and 20x downsample (feature rate about 2Hz or 84 x T matrix)

    For each batch, sample recordings from the training set and extract CQTs.
    For each CQT, randomly crop three subsequences of length 200, 300, and 400 for training
    Randomly augment each sequence with a changing factor (uniform between 0.7 and 1.3) to simulate tempo changes
    """

    def __init__(self, use_cqts: bool = True):
        super().__init__("./dataset/RondoDB100", 48000, 100, "rondodb")
        if use_cqts:
            self.train_ds, self.val_ds = generate_training_dataset_from_cqts(self.dataset_path)
        else:
            # get training data from folders
            (
                audio_train_ds,
                audio_val_ds,
            ) = tf.keras.utils.audio_dataset_from_directory(
                directory=self.dataset_path,
                batch_size=BATCH_SIZE,
                validation_split=VAL_SPLIT,
                shuffle=False,
                seed=42,
                subset="both",
            )

            # convert audio training data into CQTs based on Yang et. al. 2020
            self.train_ds = generate_training_dataset_from_audio_dataset(
                audio_train_ds, self.original_sr, True
            )
            self.val_ds = generate_training_dataset_from_audio_dataset(
                audio_val_ds, self.original_sr, False
            )

    def get_train_dataset(self):
        return self.train_ds

    def get_val_dataset(self):
        return self.val_ds


class Mazurkas(Dataset):
    def __init__(self):
        super().__init__("./dataset/Mazurkas", 48000, 49, "mazurkas")
        self.cover_artist_name = "Rubinstein"  # determines which recording to use as a reference for each song

        reference_filenames = set(
            glob(f"{self.dataset_path}/*/{self.cover_artist_name}.wav")
        )
        all_filenames = set(glob(f"{self.dataset_path}/*/*.wav"))

        reference_file_ds = tf.data.Dataset.list_files(
            list(reference_filenames)
        )
        cover_file_ds = tf.data.Dataset.list_files(
            list(all_filenames - reference_filenames)
        )

        self.reference_ds = generate_cqts_from_file_dataset(
            reference_file_ds
        ).prefetch(1)
        self.query_ds = generate_cqts_from_file_dataset(
            cover_file_ds
        ).prefetch(1)

    def get_test_datasets(self):
        """Returns a tuple of a reference dataset ((cqt, song_name) pairs) and a query dataset ((cqt, song_name) pairs)"""
        return self.reference_ds, self.query_ds


class Covers80(Dataset):
    """
    Dataset of 80 songs with 2 covers each. Used for evaluation in the literature (similarities between each song and the remaining songs and construct a distance matrix). [Yang et. al. 2018: KEY-INVARIANT CONVOLUTIONAL NEURAL NETWORK TOWARD EFFICIENT COVER SONG IDENTIFICATION]
    """

    def __init__(self):
        super().__init__("./dataset/Covers80", 16000, 80, "covers80")

        reference_file_ds = tf.data.TextLineDataset(
            f"{self.dataset_path}/reference.txt"
        )
        cover_file_ds = tf.data.TextLineDataset(
            f"{self.dataset_path}/covers.txt"
        )

        self.reference_ds = generate_cqts_from_file_dataset(
            reference_file_ds
        ).prefetch(1)
        self.query_ds = generate_cqts_from_file_dataset(
            cover_file_ds
        ).prefetch(1)

    def get_test_datasets(self):
        return self.reference_ds, self.query_ds
