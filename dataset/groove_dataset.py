import copy
import csv
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Iterable
from typing import Type
from typing import TypeVar
from urllib.request import urlretrieve

import lightning as L
import tabulate
from joblib import Memory
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_augmentation.midi_augmentor import MidiAugmentor
from data_augmentation.midi_splitter import MidiSplitter
from data_augmentation.midi_splitter import Splitter
from data_augmentation.midi_splitter import SplitterAbstract
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessor
from data_processing.track_processors_base import TrackProcessor
from utils.utils import Track
from utils.constants import DRUMS
from utils.utils import load_json
from utils.utils import process_label


############
# DATASETS #
############


class TrackDataset(Dataset):
    def __init__(
        self,
        tracks: list[Track],
        filter_tracks: Callable[[Track], bool] | None = None,
        track_processor: TrackProcessor | None = None,
    ):
        super().__init__()
        self.tracks = tracks
        if filter_tracks is not None:
            self.tracks = [track for track in self.tracks if filter_tracks(track)]
        self.track_processor = track_processor

    def __getitem__(self, index: int):
        track = self.tracks[index]
        if self.track_processor is not None:
            return self.track_processor(track)
        else:
            return track

    def __len__(self) -> int:
        return len(self.tracks)

    def copy(self):
        return TrackDataset(
            tracks=[track.copy() for track in self.tracks],
            filter_tracks=None,
            track_processor=self.track_processor,
        )


class GrooveMidiDataset(TrackDataset):
    _metadata_file_name: str = "info.csv"

    def __init__(
        self,
        path: Path,
        filter_tracks: Callable[[Track], bool] | None = None,
        override_metadata: Path | None = None,
        track_processor: TrackProcessor | None = None,
    ):
        self.path = path
        metadata = override_metadata or self.path / self._metadata_file_name
        with open(self.path / metadata) as f:
            self._metadata = list(csv.DictReader(f))
        super().__init__(
            [
                Track(
                    midi_path=self.path.absolute() / row["midi_filename"],
                    label=process_label(row["style"]),
                    bpm=float(row["bpm"]),
                    time_signature=tuple(map(int, row["time_signature"].split("-"))),
                    split=row["split"],
                )
                for row in self._metadata
                if row["beat_type"] == "beat"
            ],
            filter_tracks=filter_tracks,
            track_processor=track_processor,
        )


################
# DATA MODULES #
################


class GrooveMidiModule(L.LightningDataModule):
    _download_url: str = (
        "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
    )
    _success_file_name: str = ".success"
    _groove_folder_name: str = "groove"
    _cache_folder_name: str = "cache"
    _augmentation_folder_prefix: str = "augmented"
    _split_folder_prefix: str = "split"
    _labels_file_name: str = "labels.txt"
    _already_setup: bool = False
    num_workers: int = 20

    def __init__(
        self,
        path: Path,
        processed_data_path: Path = Path(tempfile.gettempdir()),
        batch_size: int = 128,
        track_processor: TrackProcessor | None = None,
        augmentor: MidiAugmentor | None = None,
        splitter: Splitter | None = None,
        override_metadata: Path = None,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.path = path
        self.processed_data_path = processed_data_path
        self.batch_size = batch_size
        self.track_processor = track_processor
        self.augmentor = augmentor
        self.splitter = splitter
        self.override_metadata = override_metadata
        self.n_jobs = n_jobs

        # Other attributes
        self.drums = DRUMS
        if self.track_processor is not None:
            cache_folder = (
                self.processed_data_path
                / self._cache_folder_name
                / self.track_processor.str_state()
            )
            self.memory = Memory(location=cache_folder, verbose=0)
            self.track_processor.track_to_tensor_with_positions = self.memory.cache(
                self.track_processor.track_to_tensor_with_positions,
                ignore=["self"],
            )

    def prepare_data(self) -> None:
        success_file = self.path / self._success_file_name
        # Download dataset
        if not success_file.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            zip_path = self.path / "groove.zip"
            print(f"Downloading dataset from {self._download_url}...")
            urlretrieve(self._download_url, zip_path)
            # Unzip dataset
            print("Unzipping dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.path)
            # Write success file
            success_file.touch()

    def setup(self, stage: str) -> None:
        if self._already_setup:
            return
        # Split dict keys
        train_key, val_key, test_key = "train", "validation", "test"
        split_keys = [train_key, val_key, test_key]

        # Structure to gather information about labels
        # The structure is the following: label_infos[label][split][field] = value
        self.label_infos = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Load datasets
        dataset_path = self.path / self._groove_folder_name
        datasets = {}
        for split in split_keys:
            datasets[split] = GrooveMidiDataset(
                dataset_path,
                filter_tracks=lambda t: t.split == split,
                override_metadata=self.override_metadata,
                track_processor=self.track_processor,
            )
            for track in datasets[split].tracks:
                self.label_infos[track.label][split]["tracks"] += 1
        # Write a first version of the label infos in the dataset folder
        self.export_label_infos(dataset_path / self._labels_file_name)

        # Create label indexer
        self.unique_labels = sorted([label for label, infos in self.label_infos.items()])
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        for split in split_keys:
            for track in datasets[split].tracks:
                track.label_idx = self.label_to_id[track.label]

        # Augment dataset if needed
        if self.augmentor is not None:
            augmented_path = (
                self.processed_data_path
                / f"{self._augmentation_folder_prefix}_num={self.augmentor.num_variations_per_track}"
            )
            # Augment only the train set
            augmented_tracks_dict = self.augmentor.augment_list_of_tracks(
                tracks=datasets[train_key].tracks,
                output_dir=augmented_path,
                skip_existing=True,
                root_path=self.path,
                n_jobs=self.n_jobs,
            )
            datasets[train_key].tracks = sum(augmented_tracks_dict.values(), start=[])
            for track in datasets[train_key].tracks:
                self.label_infos[track.label][train_key]["augmented"] += 1

        # Split dataset into measures if needed

        if self.splitter is not None:
            split_path = (
                self.processed_data_path
                / f"{self._split_folder_prefix}_measures={self.splitter.num_measures_per_chunk}"
            )
            # Split all datasets
            for split in split_keys:
                root_path = (
                    augmented_path
                    if self.augmentor is not None and split == train_key
                    else self.path
                )
                split_tracks_dict = self.splitter.split_list_of_tracks(
                    tracks=datasets[split].tracks,
                    output_dir=split_path,
                    root_path=root_path,
                    skip_existing=True,
                    n_jobs=self.n_jobs,
                )
                datasets[split].tracks = sum(split_tracks_dict.values(), start=[])
                for track in datasets[split].tracks:
                    self.label_infos[track.label][split]["chunks"] += 1

        # Set attributes
        self.train_dataset = datasets[train_key]
        self.val_dataset = datasets[val_key]
        self.test_dataset = datasets[test_key]

        self._already_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.track_processor.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.track_processor.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.track_processor.collate_fn,
            num_workers=self.num_workers,
        )

    def label_infos_to_string(self, tablefmt: str = "rounded_outline") -> str:
        def unique_list(it: Iterable):
            return list(dict.fromkeys(it))

        infos = copy.deepcopy(self.label_infos)
        labels = unique_list(label for label in infos)
        splits = unique_list(split for label in infos for split in infos[label])
        fields = unique_list(
            field for label in infos for split in infos[label] for field in infos[label][split]
        )
        # add split "Total" to dict infos
        for label in infos:
            for field in infos[label][splits[0]]:
                infos[label]["total"][field] = sum(infos[label][split][field] for split in splits)
        splits.insert(0, "total")

        result = ""
        for split in splits:
            result += f"{split}\n"
            table = []
            for label in labels:
                table.append([label] + [infos[label][split][field] for field in fields])
            result += tabulate.tabulate(table, headers=["label"] + list(fields), tablefmt=tablefmt)
            result += "\n\n"

        return result

    def export_label_infos(self, path: Path, tablefmt: str = "rounded_outline") -> None:
        with open(path, "w") as f:
            f.write(self.label_infos_to_string(tablefmt=tablefmt))


if __name__ == "__main__":
    DATASET_PATH = Path("/tmp/groove_module")

    data_module = GrooveMidiModule(
        DATASET_PATH,
        track_processor=RelativeMultivariateTrackProcessor(DRUMS),
        augmentor=MidiAugmentor(num_variations_per_track=3, include_original_track=True),
        splitter=MidiSplitter(num_measures_per_chunk=4, chunk_overlap=False),
        # n_jobs=int(0.75 * os.cpu_count()),
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")
    batch = next(iter(data_module.train_dataloader()))
    print(batch)
    print(batch["samples"].shape)
