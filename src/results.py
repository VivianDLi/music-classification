import pandas as pd
import seaborn as sns
import numpy as np
import keras.models
import json
import pickle

from dataset import *


def plot_training_loss():
    full_model_ds = pd.read_csv(
        "logs/training/full_model-2024-01-18-21-33-00.log"
    )
    full_model_ds["Model"] = "Full Model"
    small_model_ds = pd.read_csv(
        "./logs/training/small_model-2024-01-22-02-34-10.log"
    )
    small_model_ds["Model"] = "Small Model"
    combined_ds = pd.concat([full_model_ds, small_model_ds], axis=0)
    combined_ds = combined_ds.rename(
        columns={"epoch": "Epochs", "loss": "Training Loss"}
    )
    ax = sns.lineplot(
        combined_ds, x="Epochs", y="Training Loss", hue="Model", style="Model"
    )
    ax.set_title("Trainig Loss per Model")
    ax.figure.savefig("./results/training_loss_3.png")


def save_statistics_from_result_dict(
    filepath, is_covers, reference_first=False
):
    with open(filepath, "rb") as f:
        result_dict = pickle.load(f)
    if reference_first:
        result_dict_swapped = {}
        for reference in result_dict:
            for query in result_dict[reference]:
                for i, entry in enumerate(result_dict[reference][query]):
                    if (query, i) not in result_dict_swapped:
                        result_dict_swapped[(query, i)] = {}
                    result_dict_swapped[(query, i)][reference] = entry
        result_dict = result_dict_swapped
    maps = []
    pat10s = []
    meanrank = []
    distances = []
    for query, i in result_dict:
        distance_dict = result_dict[(query, i)]
        paired_list = []
        for reference in distance_dict:
            paired_list.append((reference, distance_dict[reference]))
        sorted_list = list(sorted(paired_list, key=lambda pair: pair[1]))
        (
            average_precision,
            precision_at10,
            rank,
            distance,
        ) = get_statistics_from_distances(query, sorted_list, is_covers)
        maps.append(average_precision)
        pat10s.append(precision_at10)
        meanrank.append(rank)
        distances.append((query, distance))
    with open(f"{filepath}-cover{is_covers}.txt", "w") as f:
        string_maps = str(np.mean(maps))
        f.write(f"MAP: {string_maps}\n")
        f.write(f"P@10: {np.mean(pat10s)}\n")
        f.write(f"Mean Rank of 1st Correct Entry: {np.mean(meanrank)}\n")
        f.write(f"Distances (for covers): {distances}\n")


def get_statistics_from_distances(query, distances, is_covers):
    """Distances is a list of (reference, distance) pairs."""
    if is_covers:
        total_matches = 0
        total_matches_ap = 0
        total_precision = 0
        rank = -1
        min_dist = 0
        distance = -1
        for i, pair in enumerate(distances):
            if i == 0:
                min_dist = pair[1]
            # check for precision at 10
            if query == pair[0] and i + 1 < 10:
                total_matches += 1
            # check for rank of first match
            if query == pair[0] and rank == -1:
                rank = i + 1
            # check for average precision
            if query == pair[0]:
                distance = pair[1]
                total_matches_ap += 1
                total_precision += total_matches_ap / (i + 1)
        return (
            total_precision / total_matches_ap,
            total_matches / 10,
            rank,
            (min_dist, distance),
        )
    else:
        total_matches = 0
        total_matches_ap = 0
        total_precision = 0
        rank = -1
        min_dist = 0
        distance = -1
        for i, pair in enumerate(distances):
            if i == 0:
                min_dist = pair[1]
            # check for precision at 10
            if query == pair[0] and i + 1 < 10:
                total_matches += 1
            # check for rank of first match
            if query == pair[0] and rank == -1:
                rank = i + 1
            # check for average precision
            if query == pair[0]:
                distance = pair[1]
                total_matches_ap += 1
                total_precision += total_matches_ap / (i + 1)
        if total_matches_ap == 0:
            return None
        return (
            total_precision / total_matches_ap,
            total_matches / 10,
            rank,
            (min_dist, distance),
        )


def save_references_as_json(model_name, dataset):
    model = keras.models.load_model(
        f"./models/{model_name}-representation.keras"
    )
    reference_ds = dataset.get_test_datasets()[0]
    result = {}
    for cqt, label in reference_ds:
        representation = np.squeeze(model(cqt).numpy()).tolist()
        label = tf.squeeze(tf.io.decode_raw(label, tf.uint8))
        label = "".join(chr(i) for i in label)
        result[label] = representation
    with open(f"./results/{model_name}-{dataset.name}.json", "w") as f:
        json.dump(result, f)


def get_gzipped_model_size(file):
    import os
    import zipfile
    import tempfile

    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(
        zipped_file, "w", compression=zipfile.ZIP_DEFLATED
    ) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


def plot_cqt_example():
    import librosa
    import matplotlib.pyplot as plt

    y, sr = librosa.load("./results/example_notes.m4a")
    fig, axes = plt.subplots(2, 1, sharex=True)
    stft = librosa.stft(y)
    cqt = librosa.cqt(y, sr=sr)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    img = librosa.display.specshow(
        stft_db, cmap="viridis", sr=sr, x_axis="time", y_axis="log", ax=axes[0]
    )
    librosa.display.specshow(
        cqt_db,
        cmap="viridis",
        sr=sr,
        x_axis="time",
        y_axis="cqt_hz",
        ax=axes[1],
    )
    axes[0].set(title="STFT Spectrogram")
    axes[0].label_outer()
    axes[1].set(title="CQT Spectrogram")
    axes[1].label_outer()
    fig.colorbar(img, ax=axes, format="%+2.f dB")
    fig.savefig("./results/spectrogram_comparison.png")


def plot_error_distance():
    with open("./results/covers80-distances.json", "r") as f:
        distance_dict = json.load(f)
    records = []
    for model in distance_dict:
        if "full" in model:
            model_kind = "Normal Model"
        else:
            model_kind = "Smaller Model"
        if "quantized" in model and "pruned" in model:
            model_comp = "Quantized, Pruned"
        elif "quantized" in model:
            model_comp = "Quantized"
        elif "pruned" in model:
            model_comp = "Pruned"
        else:
            model_comp = "None"
        for song in distance_dict[model]:
            min_dist, correct_dist = distance_dict[model][song]
            records.append(
                {
                    "Model": model_kind,
                    "Compression": model_comp,
                    "Distance MSE": (correct_dist - min_dist) ** 2,
                }
            )
    df = pd.DataFrame.from_records(records)
    axes = sns.barplot(df, x="Model", y="Distance MSE", hue="Compression")
    axes.set(title="MSE Distance between First and Correct Entry")
    axes.figure.savefig("./results/mse_plot.png")


save_statistics_from_result_dict(
    "./results/quantized_full_model-mazurkas.pkl", True
)
save_statistics_from_result_dict(
    "./results/quantized_small_model-mazurkas.pkl", True
)
save_statistics_from_result_dict(
    "./results/quantized_pruned_full_model-mazurkas.pkl", False
)
# plot_error_distance()
