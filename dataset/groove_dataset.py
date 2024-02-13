import copy
import csv
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable
from typing import Iterable
from urllib.request import urlretrieve

import lightning as L
import tabulate
from joblib import Memory
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_augmentation.midi_augmentor import MidiAugmentor
from data_augmentation.midi_splitter import Splitter
from data_processing.track_processor_relative import RelativeMultivariateTrackProcessor
from data_processing.track_processors_base import TrackProcessor
from utils.classes import Track
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
        track_processor: TrackProcessor | None = None,
    ):
        self.path = path
        with open(self.path / self._metadata_file_name) as f:
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


######################
# DATASET PROCESSING #
######################


def split_midi_dataset(
    dataset: TrackDataset,
    split_path: Path,
    num_measures_per_chunk: int = 4,
    chunk_overlap: bool = False,
    force_split_recomputation: bool = False,
    in_place: bool = False,
) -> TrackDataset:
    split_path = split_path.absolute()
    splitter: Splitter = Splitter(num_measures_per_chunk, chunk_overlap)
    chunk_paths = splitter.split_midi_dataset(
        dataset.tracks,
        split_path,
        overwrite_existing=force_split_recomputation,
        subpbar=False,
    )
    new_tracks = [
        Track.from_partial(track, midi_path=chunk_path)
        for track in dataset.tracks
        for chunk_path in chunk_paths[track.midi_path]
    ]
    if not in_place:
        dataset = dataset.copy()
    dataset.tracks = new_tracks
    return dataset


def augment_midi_dataset(
    dataset: TrackDataset,
    augmented_path: Path,
    num_augmentations: int = 1,
    force_augmentation_recomputation: bool = False,
    in_place: bool = False,
    seed: int | None = None,
    **augmentator_kwargs,
) -> TrackDataset:
    augmentor = MidiAugmentor(seed=seed, **augmentator_kwargs)
    augmented_file_paths: dict[Path, list[Path]] = augmentor.augment_midi_dataset(
        dataset.tracks,
        augmented_path,
        num_variations_per_track=num_augmentations,
        only_if_doesnt_exist=not force_augmentation_recomputation,
    )
    # new_tracks = [
    #     Track.from_partial(track, midi_path=augmented_path)
    #     for track in dataset.tracks
    #     for augmented_path in augmented_file_paths[track.midi_path]
    # ]
    bpm_files = {}

    def get_bpm(file_path: Path):
        parent = file_path.parent
        if parent not in bpm_files:
            bpm_files[parent] = load_json(parent / MidiAugmentor.bpm_filename)
        return bpm_files[parent][file_path.name]

    new_tracks = [
        Track.from_partial(track, midi_path=augmented_path, bpm=get_bpm(augmented_path))
        for track in dataset.tracks
        for augmented_path in augmented_file_paths[track.midi_path]
    ]
    del bpm_files
    if not in_place:
        dataset = dataset.copy()
    dataset.tracks = new_tracks
    return dataset


################
# DATA MODULES #
################


class GrooveMidiModule(L.LightningDataModule):
    _download_url: str = (
        "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
    )
    _success_file_name: str = ".success"
    _groove_folder_name: str = "groove"
    _labels_file_name: str = "labels.txt"
    _already_setup: bool = False
    num_workers: int = 20

    def __init__(
        self,
        path: Path,
        batch_size: int = 128,
        track_processor: TrackProcessor | None = None,
        min_tracks_per_label: int = 20,
        cache_location: Path = Path(tempfile.gettempdir()),
        split_dataset_into_measures: bool = False,
        split_dataset_into_measures_kwargs: dict = None,
        augment_dataset: bool = False,
        augment_dataset_kwargs: dict = None,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.track_processor = track_processor
        self.min_tracks_per_label = min_tracks_per_label
        self.cache_location = cache_location
        self.split_dataset_into_measures = split_dataset_into_measures
        self.split_dataset_into_measures_kwargs = (
            split_dataset_into_measures_kwargs
            if split_dataset_into_measures_kwargs is not None
            else {}
        )
        self.augment_dataset = augment_dataset
        self.augment_dataset_kwargs = (
            augment_dataset_kwargs if augment_dataset_kwargs is not None else {}
        )

        # Other attributes
        self.drums = DRUMS
        if self.track_processor is not None:
            self.memory = Memory(location=self.cache_location, verbose=0)
            self.track_processor.notes_to_tensor = self.memory.cache(
                self.track_processor.notes_to_tensor
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
                filter_tracks=lambda x: x.split == split,
                track_processor=self.track_processor,
            )
            for track in datasets[split].tracks:
                self.label_infos[track.label][split]["tracks"] += 1
        # Write a first version of the label infos in the dataset folder
        self.export_label_infos(dataset_path / self._labels_file_name)

        # Filter labels with less than min_tracks_per_label tracks in the train set
        self.unique_labels = sorted([
            label
            for label, infos in self.label_infos.items()
            if infos[train_key]["tracks"] >= self.min_tracks_per_label
        ])
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        for split in split_keys:
            datasets[split].tracks = list(
                filter(lambda x: x.label in self.unique_labels, datasets[split].tracks)
            )
            for track in datasets[split].tracks:
                self.label_infos[track.label][split]["filtered"] += 1
                track.label_idx = self.label_to_id[track.label]

        # Augment dataset if needed
        if self.augment_dataset:
            if "augmented_path" not in self.augment_dataset_kwargs:
                raise ValueError(
                    "You must specify the 'augmented_path' argument in 'dataset_class_kwargs' to"
                    " augment the dataset."
                )
            self.augment_dataset_kwargs["augmented_path"] = (
                self.augment_dataset_kwargs["augmented_path"]
                / f"augmentations={self.augment_dataset_kwargs['num_augmentations']}"
            )
            # Augment only the train set
            augment_midi_dataset(datasets[train_key], in_place=True, **self.augment_dataset_kwargs)
            for track in datasets[train_key].tracks:
                self.label_infos[track.label][train_key]["augmented"] += 1

        # Split dataset into measures if needed
        if self.split_dataset_into_measures:
            if "split_path" not in self.split_dataset_into_measures_kwargs:
                raise ValueError(
                    "You must specify the 'split_path' argument in 'dataset_class_kwargs' to split"
                    " the dataset."
                )
            split_folder_name_elements = [
                f"augmentations={self.augment_dataset_kwargs['num_augmentations']}",
                f"measures={self.split_dataset_into_measures_kwargs['num_measures_per_chunk']}",
                f"overlap={self.split_dataset_into_measures_kwargs['chunk_overlap']}",
            ]
            self.split_dataset_into_measures_kwargs["split_path"] = (
                self.split_dataset_into_measures_kwargs["split_path"]
                / "_".join(split_folder_name_elements)
            )
            for split in split_keys:
                split_midi_dataset(
                    datasets[split], in_place=True, **self.split_dataset_into_measures_kwargs
                )
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
    DATASET_PATH = Path("../data/test_data")

    data_module = GrooveMidiModule(
        DATASET_PATH,
        track_processor=RelativeMultivariateTrackProcessor(DRUMS),
        split_dataset_into_measures=True,
        split_dataset_into_measures_kwargs={
            "split_path": DATASET_PATH / "split",
            "num_measures_per_chunk": 2,
            "chunk_overlap": False,
        },
        augment_dataset=True,
        augment_dataset_kwargs={
            "augmented_path": DATASET_PATH / "augmented",
            "num_augmentations": 2,
        },
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")
    batch = next(iter(data_module.train_dataloader()))
    print(batch)
    print(batch["vectors"].shape)
