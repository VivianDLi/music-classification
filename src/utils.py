import numpy as np
import tensorflow as tf
import librosa
from scipy.signal import convolve, resample


# @tf.function
# def process_cqt_dataset(orig_sr: int, audio, label):
#     """
#     Wrapper for tensorflow.

#     'audio' has format (batch_size, sequence_length, num_channels) and 'labels' has format (batch_size,)
#     """
#     # determine sizes
#     batch_size, sequence_length = audio.shape
#     batched_cqt = process_cqt_dataset(orig_sr, audio)
#     batched_cqt.set_shape((batch_size, 84, sequence_length / 20))
#     return batched_cqt, label


def process_cqt_dataset(orig_sr: int, audio, label):
    """
    For each audio file, resample to 22050 Hz mono-channel and convert to CQT using Librosa, with 12 bins per octave and a Hann window with hop-size 512.
    Further process with a 20-point mean filter and downsample by 20 times.

    'audio' has format (batch_size, sequence_length, num_channels)
    """
    # mono + 22050 Hz
    audio = librosa.resample(
        librosa.to_mono(audio), orig_sr=orig_sr, target_sr=22050
    )
    batched_cqt = librosa.cqt(
        audio,
        sr=22050,
        hop_length=512,
        n_bins=84,
        n_bins_per_octave=12,
        window="hann",
    )
    # cqt has shape (batch_size, n_bins (84), t)
    for cqt in batched_cqt:
        for cqt_bin in cqt:
            # apply mean filter per bin
            cqt_bin = mean_filter(cqt_bin, 20)
        # downsample CQT by 20x
        cqt = resample(cqt, cqt.shape[1] / 20, axis=1)

    return batched_cqt, label


def process_cropped_scaled_dataset(dataset):
    """
    Split dataset into three cropped and scaled datasets, and combine back into one.

    For each training batch, randomly crop three different subsequences of lengths 200, 300, and 400, and randomly augment them with a changing factor (uniformly drawn between 0.7 and 1.3) for simulating tempo changes (essentially increasing the batch size by 3).
    """
    datasets = []
    for l in [200, 300, 400]:
        r = np.random.uniform(0.7, 1.3)
        datasets.append(
            dataset.map(
                map_func=lambda cqt, label: (
                    tf.image.resize(
                        tf.image.random_crop(
                            value=cqt, size=(cqt.shape[0], cqt.shape[1], l)
                        )[..., tf.newaxis],
                        [84, l * r],
                    )[..., 0],
                    label,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )
    return datasets[0].concatenate(datasets[1]).concatenate(datasets[2])


def mean_filter(x, window_size):
    kernel = np.ones(window_size) / window_size
    return convolve(x, kernel)
