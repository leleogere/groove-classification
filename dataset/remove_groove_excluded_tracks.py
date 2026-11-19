import csv
import os
import subprocess
from pathlib import Path

import click
import joblib
from tqdm import tqdm


def comma_separated_list(string: str) -> list[str]:
    return string.split(",")


# fmt: off
@click.command()
@click.argument("groove_path", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True, path_type=Path))
@click.argument("output_metadata_file", type=click.Path(exists=False, file_okay=True, dir_okay=False, readable=True, writable=True, path_type=Path))
@click.option("-g", "--allowed-genres", type=comma_separated_list, help="Allowed genres, comma separated (like 'funk,jazz,latin,rock').")
@click.option("-t", "--allowed-time-signatures", type=comma_separated_list, help="Allowed time signatures, comma separated (like '4/4,3/4'")
@click.option("-G", "--grammar", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, writable=False, path_type=Path), help="Grammar to use to exclude tracks.")
@click.option("--allow-fill-tracks", is_flag=True, help="Allow fill tracks.")
# fmt: on
def cli(**kwargs):
    filter_groove_dataset_metadata(**kwargs)


def filter_groove_dataset_metadata(
    groove_path: Path,
    output_metadata_file: Path,
    allowed_genres: list[str] = None,
    allowed_time_signatures: list[str] = None,
    grammar: Path = None,
    allow_fill_tracks: bool = False,
):
    groove_info = groove_path / "info.csv"

    # Read the original info.csv
    with groove_info.open("r") as f:
        tracks = list(csv.DictReader(f))

    def is_track_valid(track: dict[str, str]) -> tuple[bool, str | None]:
        if not allow_fill_tracks and track["beat_type"] == "fill":
            return False, "Fill tracks not allowed"
        genre = track["style"].split("/")[0]
        if allowed_genres and genre not in allowed_genres:
            return False, f"Genre {genre} not allowed"
        time_signature = track["time_signature"].replace("-", "/")
        if allowed_time_signatures and time_signature not in allowed_time_signatures:
            return False, f"Time signature {time_signature} not allowed"
        if grammar:
            try:
                subprocess.run(
                    [
                        "python",
                        "utils/_parse_midi_subprocess.py",
                        str(groove_path / track["midi_filename"]),
                        "--tempo",
                        str(track["bpm"]),
                        "--time-signature",
                        *track["time_signature"].split("-"),
                        "--grammar-file",
                        str(grammar),
                    ],
                    check=True,
                    env=os.environ,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                return False, "Error parsing track with grammar"
        return True, None

    validities = joblib.Parallel(n_jobs=-1, return_as="generator")(
        joblib.delayed(is_track_valid)(track) for track in tracks
    )

    midi_sizes = [Path(groove_path / track["midi_filename"]).stat().st_size for track in tracks]
    pbar = tqdm(
        tracks,
        desc=f"Checking validity of tracks in {groove_path}",
        total=sum(midi_sizes),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    valid_tracks = []
    for track, valid_and_reason, size in zip(pbar, validities, midi_sizes):
        valid, reason = valid_and_reason
        if valid:
            valid_tracks.append(track)
        else:
            pbar.write(f"Skipping {track['midi_filename']}: {reason}.")
        pbar.update(size)

    with output_metadata_file.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=tracks[0].keys())
        writer.writeheader()
        writer.writerows(valid_tracks)


if __name__ == "__main__":
    cli()
