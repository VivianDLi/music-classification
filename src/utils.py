import numpy as np
import tensorflow as tf
import librosa
import os

from constants import BATCH_SIZE, VAL_SPLIT


def convert_to_cqt(audio: tf.Tensor, original_sr: int):
    """
    For each audio tensor, resample to 22050 Hz mono-channel and convert to CQT using Librosa, with 12 bins per octave and a Hann window with hop-size 512.

    Process with a 20-point mean filter and downsample by 20 times (or until max length).

    'audio' has format (batch_size, sequence_length, num_channels).
    model needs format (batch_size, n_bins, sequence_length, 1)
    """
    np_audio = audio.numpy()
    # convert audio to mono (batch_size, sequence_length)
    mono_audio = np.mean(np_audio, axis=-1)
    # resample to 22050 Hz (batch_size, sequence_length, num_channels)
    resampled_audio = librosa.resample(
        mono_audio, orig_sr=original_sr, target_sr=22050
    )
    # create CQT
    cqt = np.abs(
        librosa.cqt(
            resampled_audio,
            sr=22050,
            hop_length=512,
            n_bins=84,
            bins_per_octave=12,
            window="hann",
        )
    )
    tensor_cqt = tf.convert_to_tensor(cqt[..., tf.newaxis], dtype=tf.float32)

    # cqt has shape (batch_size, n_bins (84), t)
    # apply mean filter (convolution) per bin
    mean_kernel = tf.ones((1, 20, 1, 1), tf.float32) / 20
    mean_cqt = tf.nn.conv2d(
        tensor_cqt,
        mean_kernel,
        1,
        "VALID",
    )
    # cqt is now a tensor of shape (batch_size, n_bins (84), t', 1)
    # convert to an 'image' of shape (batch_size, n_bins, t', 1) and use resize to downsample by 20x in t'
    resized_cqt = tf.image.resize(
        mean_cqt,
        [
            mean_cqt.shape[1],
            max(mean_cqt.shape[2] // 20, 200),
        ],  # to prevent too small inputs
    )
    return resized_cqt


def random_crop_and_scale(
    input_cqt: tf.Tensor,
    crop_length: int,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
):
    """
    Augment a CQT tensor by a random changing factor drawn uniformly to mimic tempo changes and randomly crop it.

    'cqt' has format (batch_size, n_bins, t, 1).

    Args:
        crop_length (int): Length of the section to randomly crop.
        min_scale (float): Minimum tempo scale. Defaults to 0.7.
        max_scale (float): Maximum tempo scale. Defaults to 1.3.
    """
    # resize cqt based on randomly chosen scaling factor
    r = tf.random.uniform(
        shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32
    )
    cqt = tf.image.resize(
        input_cqt,
        [input_cqt.shape[1], input_cqt.shape[2] * r],
    )
    # cqt is now a tensor of shape (batch_size, n_bins, t', 1)
    # randomly crop image
    cropped_cqt = tf.image.random_crop(
        value=cqt,
        size=(cqt.shape[0], cqt.shape[1], crop_length, 1),
    )
    # cqt is now a tensor of shape (batch_size, n_bins, crop_length, 1)

    return cropped_cqt


def generate_training_dataset_from_audio_dataset(
    audio_dataset: tf.data.Dataset, original_sr: int, crop_audio: bool = False
):
    """
    Takes an audio dataset created from tf.keras.utils.audio_dataset_from_directory and formats it into a training dataset based on Yang et. al. 2020: "Learning a Representation for Cover Song Identification Using Convolutional Neural Network"

    Args:
        audio_dataset (tf.data.Dataset): The audio dataset to convert. Iterating through gives a sample tensor tuple of the shape (batch_size, sequence_length, num_channels), label (int).
        original_sr (int): The original sr of recordings in the dataset. For resampling to a fixed sample rate.
        crop_audio (bool): Whether to randomly crop CQTs to a fixed length.
    """

    def generator_func(crop_length: tf.Tensor):
        for audio, label in audio_dataset:
            cqt_audio = convert_to_cqt(audio, original_sr)
            if crop_audio:
                cqt_audio = random_crop_and_scale(cqt_audio, crop_length)
            yield cqt_audio, label

    if crop_audio:
        # create three different datasets based on crop lengths of [200, 300, 400]
        crop_200_dataset = tf.data.Dataset.from_generator(
            generator_func,
            output_signature=(
                tf.TensorSpec(
                    shape=(BATCH_SIZE, 84, 200, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
            ),
            args=(tf.constant(200),),
        )
        crop_300_dataset = tf.data.Dataset.from_generator(
            generator_func,
            output_signature=(
                tf.TensorSpec(
                    shape=(BATCH_SIZE, 84, 300, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
            ),
            args=(tf.constant(300),),
        )
        crop_400_dataset = tf.data.Dataset.from_generator(
            generator_func,
            output_signature=(
                tf.TensorSpec(
                    shape=(BATCH_SIZE, 84, 400, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
            ),
            args=(tf.constant(400),),
        )

        # combine datasets by interleaving results
        index_dataset = tf.data.Dataset.range(3).repeat(len(audio_dataset))
        return tf.data.Dataset.choose_from_datasets(
            [crop_200_dataset, crop_300_dataset, crop_400_dataset],
            index_dataset,
            stop_on_empty_dataset=False,
        )
    else:
        return tf.data.Dataset.from_generator(
            generator_func,
            output_signature=(
                tf.TensorSpec(
                    shape=(BATCH_SIZE, 84, None, 1), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int32),
            ),
            args=(tf.constant(0),),
        )


def generate_cqts_from_file_dataset(file_dataset):
    """
    Take a file dataset and converts them into CQTs to input into the models.
    This will be run separately for the representative and query file datasets.

    Args:
        file_dataset (tf.data.Dataset): dataset of file names.
    """

    def generator_func():
        for file in file_dataset:
            string_file = file.numpy().decode("utf-8")
            label = string_file.split("/")[-2]
            raw_audio = tf.io.read_file(file)
            audio, sample_rate = tf.audio.decode_wav(raw_audio, 1)
            cqt = convert_to_cqt(
                tf.expand_dims(audio, axis=0), sample_rate.numpy()
            )
            yield tf.squeeze(cqt, axis=0), tf.convert_to_tensor(
                label, dtype=tf.string
            )

    return tf.data.Dataset.from_generator(
        generator_func,
        output_signature=(
            tf.TensorSpec(shape=(84, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    ).batch(1)


def generate_training_dataset_from_cqts(folder_path):
    class_names = [f.name for f in os.scandir(folder_path) if f.is_dir()]

    @tf.py_function(
        Tout=(
            tf.TensorSpec(shape=(84, None, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    def file_path_to_tensors(crop_length, file):
        string_file = file.numpy().decode("utf-8")
        label = string_file.split("/")[-2]
        class_index = class_names.index(label)
        if crop_length == 0:
            cqt = np.load(string_file)
        else:
            cqt = np.load(string_file[:-7] + f"{crop_length}_train.npy")
        return tf.convert_to_tensor(
            np.squeeze(cqt, axis=0), dtype=tf.float32
        ), tf.convert_to_tensor(class_index, dtype=tf.int32)

    def wrapper_function(crop_length, file):
        cqt, label = file_path_to_tensors(crop_length, file)
        cqt.set_shape(tf.TensorShape([84, None, 1]))
        label.set_shape(tf.TensorShape([]))
        return cqt, label

    all_files = tf.data.Dataset.list_files(
        folder_path + "/*/*_val.npy"
    ).shuffle(64)
    split = int(len(all_files) * VAL_SPLIT)
    val_files = all_files.take(split)
    train_files = all_files.skip(split)

    crop_200_dataset = train_files.map(
        lambda x: wrapper_function(200, x)
    ).batch(BATCH_SIZE)
    crop_300_dataset = train_files.map(
        lambda x: wrapper_function(300, x)
    ).batch(BATCH_SIZE)
    crop_400_dataset = train_files.map(
        lambda x: wrapper_function(400, x)
    ).batch(BATCH_SIZE)
    index_dataset = tf.data.Dataset.range(3).repeat(len(all_files))

    return tf.data.Dataset.choose_from_datasets(
        [crop_200_dataset, crop_300_dataset, crop_400_dataset],
        index_dataset,
        stop_on_empty_dataset=False,
    ), val_files.map(lambda x: wrapper_function(400, x)).batch(1)
