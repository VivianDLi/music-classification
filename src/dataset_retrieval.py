import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from yt_dlp import YoutubeDL
import ffmpeg

from utils import *


def match_mazurkas_func(performer_name, song_length):
    def _match_func(info_dict, incomplete=False):
        if (abs(info_dict.get("duration") - song_length) < 15) and (
            performer_name in info_dict.get("title")
            or performer_name in info_dict.get("description")
            or performer_name in info_dict.get("channel")
        ):
            return None
        else:
            return "Video does not pass Performer and Length filter"

    return _match_func


def search_song_from_youtube(
    query, song_length, dataset, filename, title_filter
):
    ydl_opts = {
        "outtmpl": f"./dataset/{dataset}/{filename}.%(ext)s",
        "noplaylist": True,
        "windowsfilenames": True,
        "playlistend": 10,
        "ignoreerrors": True,
        "ffmpeg_location": "./ffmpeg/bin",
        "match_filter": match_mazurkas_func(title_filter, song_length),
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        videos = ydl.extract_info(
            f"https://www.youtube.com/results?search_query={query.replace(' ', '+').replace('#', '-sharp')}",
            download=False,
        )["entries"]
        if videos is not None and len(videos) > 0:
            url = videos[0]["webpage_url"]
            ydl.download([url])


def get_song_from_youtube(url, dataset, filename, start_time=0):
    ydl_opts = {
        "outtmpl": f"./dataset/{dataset}/{filename}.%(ext)s",
        "noplaylist": True,
        "windowsfilenames": True,
        "ignoreerrors": True,
        "ffmpeg_location": "./ffmpeg/bin",
        "format": "wav/bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
    }
    if start_time != 0:
        ydl_opts["download_ranges"] = lambda info_dict, yt_instance: [
            {
                "start_time": start_time,
                "end_time": info_dict["duration"],
            },
        ]

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_mazurkas():
    data = pd.read_csv(
        "./dataset/Mazurkas/mazurka-discography.txt", sep="\t", header=0
    )
    for _, row in data.iterrows():
        piece_name = f"Chopin, Mazurkas, Op. {row['opus']}"
        Path(f"./dataset/Mazurkas/{piece_name}").mkdir(
            parents=True, exist_ok=True
        )

        query = f"Chopin Mazurka Op. {row['opus']} in {row['key']} {row['performer']}"
        search_song_from_youtube(
            query,
            row["seconds"],
            f"Mazurkas/{piece_name}",
            row["performer"],
            row["performer"],
        )


def get_covers80():
    dataset_path = Path("./dataset/Covers80/")
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            folder = folder.rename(
                Path(dataset_path / folder.name.replace("_", " "))
            )
            for song in folder.iterdir():
                if song.suffix == ".mp3":
                    ffmpeg.input(str(song.absolute())).output(
                        str(
                            Path(
                                folder
                                / (
                                    ", ".join(song.name.split("+")[0:2])
                                    .replace("_", " ")
                                    .title()
                                    + ".wav"
                                )
                            ).absolute()
                        )
                    ).run(cmd="./ffmpeg/bin/ffmpeg.exe")
                    song.unlink()


def get_covers80_listfiles():
    with open("./dataset/Covers80/list1.list") as fr:
        lines = fr.readlines()
        with open("./dataset/Covers80/reference.txt", "x") as fw:
            for line in lines:
                folder, song_name = line.split("/")
                fw.write(
                    f".\\dataset\\Covers80\\{folder.replace('_', ' ')}\\{', '.join(song_name.split('+')[0:2]).replace('_', ' ').title()}.wav\n"
                )
    with open("./dataset/Covers80/list2.list") as fr:
        lines = fr.readlines()
        with open("./dataset/Covers80/covers.txt", "x") as fw:
            for line in lines:
                folder, song_name = line.split("/")
                fw.write(
                    f".\\dataset\\Covers80\\{folder.replace('_', ' ')}\\{', '.join(song_name.split('+')[0:2]).replace('_', ' ').title()}.wav\n"
                )


def get_rondodb100():
    with open("./dataset/RondoDB100/dataset_rdb_100.json") as f:
        data = json.load(f)
        for composition in data:
            piece_name = (
                composition["piece_full_name"]
                .replace("/", "-")
                .replace('"', "")
                .replace(":", ",")
            )
            Path(
                f"./dataset/RondoDB100/{composition['composer']['name']}, {piece_name}"
            ).mkdir(parents=True, exist_ok=True)
            for i, recording in enumerate(composition["recordings"]):
                get_song_from_youtube(
                    recording["url"],
                    f"RondoDB100/{composition['composer']['name']}, {piece_name}",
                    i,
                    recording["start_at"],
                )


def get_rondodb100_cqts():
    file_dataset = tf.data.Dataset.list_files(f"./dataset/RondoDB100/*/*.wav")

    for file in file_dataset:
        print(file)
        raw_audio = tf.io.read_file(file)
        audio, sample_rate = tf.audio.decode_wav(raw_audio, 1)
        print(audio.shape)
        print(sample_rate)
        cqt = convert_to_cqt(
            tf.expand_dims(audio, axis=0), sample_rate.numpy()
        ).numpy()
        print("cqt")
        cqt_200 = random_crop_and_scale(cqt, 200).numpy()
        print("cqt200")
        cqt_300 = random_crop_and_scale(cqt, 300).numpy()
        print("cqt300")
        cqt_400 = random_crop_and_scale(cqt, 400).numpy()
        print("cqt400")

        np.save(file.numpy().decode('utf-8')[:-4] + "_200_train.npy", cqt_200)
        np.save(file.numpy().decode('utf-8')[:-4] + "_300_train.npy", cqt_300)
        np.save(file.numpy().decode('utf-8')[:-4] + "_400_train.npy", cqt_400)
        np.save(file.numpy().decode('utf-8')[:-4] + "_val.npy", cqt)


# get_shs100k()
# get_mazurkas()
# get_rondodb100()
# get_covers80()
# get_covers80_listfiles()
get_rondodb100_cqts()
