import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from scipy import stats
from tqdm import tqdm

from utils.utils import Track
from utils.utils import dump_json
from utils.utils import find_common_root
from utils.utils import get_midi_paths
from utils.utils import load_and_validate_midi_file
from utils.utils import load_json
from utils.utils import merge_tracks
from utils.utils import parse_bpm_from_file_name


def _chunk_filename(
    original_midi_path: Path,
    start_bar: int,
    num_bars: int,
    formatting_number_width: int,
):
    filename_end = f"startbar={start_bar:0{formatting_number_width}}_numbars={num_bars}"
    return f"{original_midi_path.stem}_{filename_end}.mid"


def truncated_normal(
    mean: float = 0, std: float = 1, a: float = -np.inf, b: float = np.inf, size: int = 1
) -> float | np.ndarray:
    """
    Sample from a truncated normal distribution.
    :param mean: Mean of the normal distribution.
    :param std: Standard deviation of the normal distribution.
    :param a: Lower bound of the truncation.
    :param b: Upper bound of the truncation.
    :param size: Number of samples to draw.
    :return: Samples from the truncated normal distribution.
    """
    if std == 0:
        x = np.full(size, mean)
    else:
        x = stats.truncnorm((a - mean) / std, (b - mean) / std, loc=mean, scale=std).rvs(size=size)
    return x[0] if size == 1 else x


class MidiAugmentor:
    bpm_filename: str = "bpm.json"

    def __init__(
        self,
        num_variations_per_track: int = 1,
        include_original_track: bool = True,
        trunc_gaussian_parameters_for_std_time_shift: tuple[float, float] = (0.01, 0.005),
        trunc_gaussian_parameters_global_velocity_shape: tuple[float, float] = (0, 0.15),
        trunc_gaussian_parameters_for_std_velocity_shift: tuple[float, float] = (3, 1),
        trunc_gaussian_parameters_for_proba_error: tuple[float, float] = (0.03, 0.02),
        gaussian_parameters_for_mean_error_shift: tuple[float, float] = (0.05, 0.02),
        trunc_gaussian_parameters_for_std_error_shift: tuple[float, float] = (0.03, 0.01),
        gaussian_parameters_for_tempo_variation: tuple[float, float] = (0, 7),
        _global_velocity_shape_force: float = 0.5,
        seed: int | None = None,
    ):
        self.num_variations_per_track = num_variations_per_track
        self.include_original_track = include_original_track
        self.trunc_gaussian_parameters_for_std_time_shift = (
            trunc_gaussian_parameters_for_std_time_shift
        )
        self.trunc_gaussian_parameters_global_velocity_shape = (
            trunc_gaussian_parameters_global_velocity_shape
        )
        self.trunc_gaussian_parameters_for_std_velocity_shift = (
            trunc_gaussian_parameters_for_std_velocity_shift
        )
        self.trunc_gaussian_parameters_for_proba_error = trunc_gaussian_parameters_for_proba_error
        self.gaussian_parameters_for_mean_error_shift = gaussian_parameters_for_mean_error_shift
        self.trunc_gaussian_parameters_for_std_error_shift = (
            trunc_gaussian_parameters_for_std_error_shift
        )
        self.gaussian_parameters_for_tempo_variation = gaussian_parameters_for_tempo_variation
        self._global_velocity_shape_force = _global_velocity_shape_force
        self.seed = seed

    def single_augmentation(
        self,
        track: Track,
        output_path: Path,
        seed: int | None = None,
    ) -> Track:
        """
        Augment a MIDI file by adding noise to the note events and playing on tempo.
        :param track: Track to augment.
        :param output_path: Output file path.
        :param seed: Seed for the random number generator.
        :return: Tempo of the new MIDI file.
        :todo: What if a note starts before the chunk but ends after the chunk?
        """
        # Read midi and associated information
        midi = load_and_validate_midi_file(
            track.midi_path,
            unique_track=True,
            unique_tempo=True,
            unique_time_signature=True,
        )
        bpm = midi.get_tempo_changes()[1][0]

        # Sample augmentation parameters
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        std_time_shift = truncated_normal(
            *self.trunc_gaussian_parameters_for_std_time_shift, a=0, b=np.inf
        )
        global_velocity_shape = truncated_normal(
            *self.trunc_gaussian_parameters_global_velocity_shape, a=-1, b=1
        )
        std_velocity_shift = truncated_normal(
            *self.trunc_gaussian_parameters_for_std_velocity_shift, a=0, b=np.inf
        )
        proba_error = truncated_normal(*self.trunc_gaussian_parameters_for_proba_error, a=0, b=1)
        mean_error_time_shift = np.random.normal(*self.gaussian_parameters_for_mean_error_shift)
        std_error_time_shift = truncated_normal(
            *self.trunc_gaussian_parameters_for_std_error_shift, a=0, b=np.inf
        )
        tempo_variation = np.random.normal(*self.gaussian_parameters_for_tempo_variation)

        # Create new midi
        new_midi = copy.deepcopy(midi)

        # Change tempo if needed
        if tempo_variation != 0:
            new_tempo = int(bpm + tempo_variation)
            bpm_ratio = bpm / new_tempo
            end_time = midi.get_end_time()
            new_midi.adjust_times([0, end_time], [0, end_time * bpm_ratio])
            bpm = new_tempo

        # Prepare noise
        num_notes = len(new_midi.instruments[0].notes)
        spb = 60 / bpm  # X beat/minute = 1/X minute/beat = 60/X second/beat
        time_shifts = np.random.normal(0, std_time_shift * spb, num_notes)
        velocity_shaper = lambda x: x ** (self._global_velocity_shape_force**-global_velocity_shape)
        velocity_shifts = np.random.normal(0, std_velocity_shift, num_notes)
        error_shifts = np.random.normal(
            mean_error_time_shift * spb, std_error_time_shift * spb, num_notes
        )
        error_shifts[np.random.rand(num_notes) > proba_error] = 0

        # Apply noise
        VELOCITY_MAX = 127
        for i, note in enumerate(new_midi.instruments[0].notes):
            shift = time_shifts[i] + error_shifts[i]
            note.start += shift
            note.end += shift
            note.velocity = int(
                np.clip(
                    velocity_shaper(note.velocity / VELOCITY_MAX) * VELOCITY_MAX
                    + velocity_shifts[i],
                    0,
                    VELOCITY_MAX,
                )
            )

        # Write midi
        new_midi.write(str(output_path))
        # Merge tracks into one (needed for grammar functions)
        merge_tracks(output_path)
        return Track.from_partial(track, midi_path=output_path, bpm=bpm)

    def augment_track(
        self,
        track: Track,
        output_dir: Path,
        num_augmentations: int = None,
        skip_existing: bool = False,
        include_original_track: bool = True,
        seed: int | None = None,
    ) -> list[Track]:
        """
        Augment a MIDI file by adding noise to the note events and playing on tempo.
        :param track: Track to augment.
        :param output_dir: Output directory in which the augmentations will be written.
        :param num_augmentations: Number of augmentations to generate.
        :param skip_existing: Whether to skip files that already have an associated output directory (it doesn't
            mean that the chunks have already been computed, an error could have occurred during the previous process).
        :param include_original_track: Whether to include the original track in the output.
        :param seed: Seed for the random number generator.
        :return: A dictionary mapping each MIDI path to its variations.
        """
        if num_augmentations is None:
            num_augmentations = self.num_variations_per_track
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32 - 1, num_augmentations)

        augmented_tracks = []
        bpm_filepath = output_dir / self.bpm_filename
        bpm_infos = {}
        # Try to read a previous BPM file
        if output_dir.exists():
            try:
                bpm_infos = load_json(bpm_filepath)
            except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
                print(
                    f"Error while loading {track} variations ({type(e).__name__}. "
                    "Deleting the directory and restarting."
                )
                shutil.rmtree(output_dir)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for variation in range(num_augmentations):
                current_augmented_path = (
                    output_dir / f"{track.midi_path.stem}_variation={variation}.mid"
                )
                # If already exist, use it (read tempo from bpm_infos)
                if skip_existing and current_augmented_path.exists():
                    augmented_track = Track.from_partial(
                        track,
                        midi_path=current_augmented_path,
                        bpm=bpm_infos[current_augmented_path.name],
                    )
                # Else augment the track
                else:
                    current_augmented_path.parent.mkdir(parents=True, exist_ok=True)
                    augmented_track = self.single_augmentation(
                        track,
                        output_path=current_augmented_path,
                        seed=int(seeds[variation]),
                    )
                    bpm_infos[current_augmented_path.name] = augmented_track.bpm
                augmented_tracks.append(augmented_track)
            if include_original_track:
                out_file = output_dir / f"{track.midi_path.stem}_variation=original.mid"
                shutil.copy(track.midi_path, out_file)
                augmented_tracks.append(Track.from_partial(track, midi_path=out_file))
                bpm_infos[out_file.name] = track.bpm
            dump_json(bpm_infos, bpm_filepath, overwrite=True, indent=4)
        except KeyboardInterrupt:
            dump_json(bpm_infos, bpm_filepath, overwrite=True, indent=4)
            print(f"KeyboardInterrupt: BPM of current augmentations were saved in {bpm_filepath}")
            raise
        return augmented_tracks

    def augment_list_of_tracks(
        self,
        tracks: list[Track],
        output_dir: Path,
        root_path: Path = None,
        include_original_track: bool = True,
        num_variations_per_track: int | None = None,
        skip_existing: bool = False,
        seed: int | None = None,
        n_jobs: int = 1,
        pbar: bool = True,
    ) -> dict[Track, list[Track]]:
        """
        Augment a MIDI dataset by generating variations of each track.
        :param tracks: List of tracks to augment.
        :param output_dir: Output directory. The structure of the dataset will be preserved.
        :param root_path: Root path of the output structure. If None, the output structure will be
            based on the nearest common parent directory of the MIDI files in the dataset and the
            given path.
        :param include_original_track: Whether to include the original track in the output.
        :param num_variations_per_track: Number of variations to generate for each track.
        :param skip_existing: Whether to skip files that already have an associated output directory (it doesn't
            mean that the chunks have already been computed, an error could have occurred during the previous process).
        :param seed: Seed for the random number generator.
        :param n_jobs: Number of jobs to use for parallel processing.
        :param pbar: Whether to display a progress bar.
        :return: Dictionary mapping each MIDI path to its variations.
        """
        # Prepare output
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set default parameters
        if num_variations_per_track is None:
            num_variations_per_track = self.num_variations_per_track
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32 - 1, len(tracks))

        # Loop over MIDI files
        midi_paths = [t.midi_path for t in tracks]
        midi_sizes = [path.stat().st_size for path in midi_paths]
        if root_path is None:
            root_path = find_common_root(midi_paths)

        def _augment_track(track: Track, seed: int):
            """Wrapper to augment a track to be used in parallel."""
            base_path = output_dir / track.midi_path.relative_to(root_path).with_suffix("")
            return self.augment_track(
                track=track,
                output_dir=base_path,
                num_augmentations=num_variations_per_track,
                skip_existing=skip_existing,
                include_original_track=include_original_track,
                seed=seed,
            )

        # Augment data in parallel
        # TODO: use "unordered_generator" when available in joblib?
        augmented_tracks_generator = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
            joblib.delayed(_augment_track)(t, seed) for t, seed in zip(tracks, seeds)
        )

        # Loop instead of dict comprehension to track progression
        augmented_tracks_dict = {}
        pbar = tqdm(
            midi_paths,
            desc=f"Augmenting MIDI files into {num_variations_per_track} variations each",
            total=sum(midi_sizes),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not pbar,
        )
        for track, augmented_tracks, size in zip(tracks, augmented_tracks_generator, midi_sizes):
            augmented_tracks_dict[track] = augmented_tracks
            pbar.update(size)
        pbar.close()
        return augmented_tracks_dict


if __name__ == "__main__":
    TEST = "track"
    augmentor = MidiAugmentor(
        num_variations_per_track=20,
        # trunc_gaussian_parameters_for_std_time_shift=(0.01, 0.005),
        # trunc_gaussian_parameters_for_proba_error=(0, 0),
        # gaussian_parameters_for_tempo_variation=(0, 0),
        seed=10,
    )
    if TEST == "track":
        MIDI_PATH = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
        # MIDI_PATH = Path('data/groove_augmented/drummer5/session1/8_latin-venezuelan-merengue_162_beat_5-8/8_latin-venezuelan-merengue_162_beat_5-8_variation=7.mid')
        # MIDI_PATH = Path('data/groove/drummer10/session1/1_jazz-swing_124_beat_4-4.mid')
        # MIDI_PATH = Path('groove_light/drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer7/session2/22_rock_118_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer2/session3/1_rock-shuffle_85_beat_4-4.mid')
        # MIDI_PATH = Path('groove/drummer1/session2/111_rock-prog_110_beat_5-4.mid')
        OUT_PATH = Path("/tmp/augmented_tracks")
        track = Track(midi_path=MIDI_PATH, bpm=parse_bpm_from_file_name(MIDI_PATH.name))
        augmentor.augment_track(
            track,
            output_dir=OUT_PATH,
            include_original_track=True,
            skip_existing=False,
        )
    elif TEST == "dataset":
        DATASET_PATH = Path("../groove_data/groove/drummer1/session1")
        OUT_PATH = Path("/tmp/dataset_augmented")
        midi_paths = get_midi_paths(DATASET_PATH)
        tracks = [
            Track(midi_path=path, bpm=parse_bpm_from_file_name(path.name)) for path in midi_paths
        ]
        augmented_tracks = augmentor.augment_list_of_tracks(
            tracks,
            output_dir=OUT_PATH,
            include_original_track=True,
            skip_existing=False,
            n_jobs=4,
        )
    else:
        raise ValueError(f"ERROR: {TEST=}")
