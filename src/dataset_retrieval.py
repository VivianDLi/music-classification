import json
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL
import ffmpeg


def search_song_from_youtube(query, song_length, dataset, filename):
    ydl_opts = {
        "outtmpl": f"/dataset/{dataset}/{filename}.%(ext)s",
        "noplaylist": True,
        "ignoreerrors": True,
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
        videos = ydl.extract_info(f"ytsearch:{query}", download=False)[
            "entries"
        ]

        for video in videos:
            if abs(video["duration"] - song_length) < 3:  # in seconds
                url = video["webpage_url"]
                break
        ydl.download([url])


def get_song_from_youtube(url, dataset, filename, start_time=0):
    ydl_opts = {
        "outtmpl": f"/dataset/{dataset}/{filename}.%(ext)s",
        "noplaylist": True,
        "ignoreerrors": True,
        "format": "wav/bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ret = ydl.download([url])

    if ret != 1 and start_time != 0:
        file_path = str(Path(f"/dataset/{dataset}/{filename}.wav").absolute())
        ffmpeg.input(file_path, ss=start_time).output(file_path).run()


def get_mazurkas():
    data = pd.read_csv(
        "dataset/Mazurkas/mazurka-discography.txt", sep="\t", header=0
    )
    for i, row in data.iterrows():
        piece_name = f"Chopin: Mazurkas, Op. {row['opus']} in {row['key']}"
        Path(f"/dataset/Mazurkas/{piece_name}").mkdir(
            parents=False, exist_ok=True
        )

        query = f"Chopin Mazurka Op. {row['opus']} in {row['key']} {row['performer']}"
        search_song_from_youtube(
            query,
            row["seconds"],
            "Mazurkas",
            row["performer"],
        )


def get_covers80():
    for folder in Path("/dataset/Covers80/").iterdir():
        if folder.is_dir():
            folder.rename(Path(folder.name.replace("_", " ")))
            for song in folder.iterdir():
                song.rename(
                    Path(song.name.split("+")[0].replace("_", " ").title())
                )


def get_rondodb100():
    with open("/dataset/RondoDB100/dataset_rdb_100.json") as f:
        data = json.load(f)
        for composition in data:
            Path(
                f"/dataset/RondoDB100/{composition['piece_full_name_with_composer']}"
            ).mkdir(parents=False, exist_ok=True)
            for i, recording in enumerate(data["recordings"]):
                get_song_from_youtube(
                    recording["url"],
                    f"RondoDB100/{composition['piece_full_name_with_composer']}",
                    i,
                    recording["start_at"],
                )


# get_shs100k()
get_mazurkas()
get_rondodb100()
get_covers80()
