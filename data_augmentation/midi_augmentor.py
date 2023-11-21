import copy
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

from utils.classes import Track
from utils.utils import load_and_validate_midi_file, get_midi_paths, find_common_root


def _chunk_filename(original_midi_path: Path, start_bar: int, num_bars: int, formatting_number_width: int):
    filename_end = f'startbar={start_bar:0{formatting_number_width}}_numbars={num_bars}'
    return f'{original_midi_path.stem}_{filename_end}.mid'


def truncated_normal(
        mean: float = 0,
        std: float = 1,
        a: float = -np.inf,
        b: float = np.inf,
        size: int = 1
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
    x = stats.truncnorm((a - mean) / std, (b - mean) / std, loc=mean, scale=std).rvs(size=size)
    return x[0] if size == 1 else x


class MidiAugmentor:
    def __init__(
            self,
            num_variations_per_track: int = 1,
            include_original_track: bool = True,
            trunc_gaussian_parameters_for_std_time_shift: tuple[float, float] = (0.02, 0.005),
            trunc_gaussian_parameters_global_velocity_shape: tuple[float, float] = (0, 0.15),
            trunc_gaussian_parameters_for_std_velocity_shift: tuple[float, float] = (3, 1),
            trunc_gaussian_parameters_for_proba_error: tuple[float, float] = (0.07, 0.02),
            gaussian_parameters_for_mean_error_shift: tuple[float, float] = (0.1, 0.03),
            trunc_gaussian_parameters_for_std_error_shift: tuple[float, float] = (0.04, 0.01),
            gaussian_parameters_for_tempo_variation: tuple[float, float] = (0, 7),
            _global_velocity_shape_force: float = 0.5,
            seed: int | None = None
    ):
        self.num_variations_per_track = num_variations_per_track
        self.include_original_track = include_original_track
        self.trunc_gaussian_parameters_for_std_time_shift = trunc_gaussian_parameters_for_std_time_shift
        self.trunc_gaussian_parameters_global_velocity_shape = trunc_gaussian_parameters_global_velocity_shape
        self.trunc_gaussian_parameters_for_std_velocity_shift = trunc_gaussian_parameters_for_std_velocity_shift
        self.trunc_gaussian_parameters_for_proba_error = trunc_gaussian_parameters_for_proba_error
        self.gaussian_parameters_for_mean_error_shift = gaussian_parameters_for_mean_error_shift
        self.trunc_gaussian_parameters_for_std_error_shift = trunc_gaussian_parameters_for_std_error_shift
        self.gaussian_parameters_for_tempo_variation = gaussian_parameters_for_tempo_variation
        self._global_velocity_shape_force = _global_velocity_shape_force
        self.seed = seed

    def augment_midi_file(
            self,
            midi_path: Path,
            out_path: Path,
            seed: int | None = None,
    ) -> None:
        """
        Split a MIDI file into bars.
        :param midi_path: Path to the MIDI file.
        :param out_path: Output directory. All chunks will be written in that directory.
        :param seed: Seed for the random number generator.
        :return: List of paths to the chunks.
        :todo: What if a note starts before the chunk but ends after the chunk?
        """
        # Read midi and associated information
        midi = load_and_validate_midi_file(midi_path, unique_track=True, unique_tempo=True, unique_time_signature=True)
        bpm = midi.get_tempo_changes()[1][0]

        # Sample augmentation parameters
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        std_time_shift = truncated_normal(*self.trunc_gaussian_parameters_for_std_time_shift, a=0, b=np.inf)
        global_velocity_shape = truncated_normal(*self.trunc_gaussian_parameters_global_velocity_shape, a=-1, b=1)
        std_velocity_shift = truncated_normal(*self.trunc_gaussian_parameters_for_std_velocity_shift, a=0, b=np.inf)
        proba_error = truncated_normal(*self.trunc_gaussian_parameters_for_proba_error, a=0, b=1)
        mean_error_time_shift = np.random.normal(*self.gaussian_parameters_for_mean_error_shift)
        std_error_time_shift = truncated_normal(*self.trunc_gaussian_parameters_for_std_error_shift, a=0, b=np.inf)
        tempo_variation = np.random.normal(*self.gaussian_parameters_for_tempo_variation)

        # Create new midi
        new_midi = copy.deepcopy(midi)

        # Change tempo if needed
        if tempo_variation != 0:
            new_tempo = bpm + tempo_variation
            bpm_ratio = bpm / new_tempo
            end_time = midi.get_end_time()
            new_midi.adjust_times([0, end_time], [0, end_time * bpm_ratio])
            bpm = new_tempo

        # Prepare noise
        num_notes = len(new_midi.instruments[0].notes)
        if seed is not None:
            np.random.seed(seed)
        mspb = 60000 / bpm
        time_shifts = np.random.normal(0, std_time_shift * mspb, num_notes)
        velocity_shaper = lambda x: x ** (self._global_velocity_shape_force ** -global_velocity_shape)
        velocity_shifts = np.random.normal(0, std_velocity_shift, num_notes)
        error_shifts = np.random.normal(mean_error_time_shift * mspb, std_error_time_shift * mspb, num_notes)
        error_shifts[np.random.rand(num_notes) > proba_error] = 0

        # Apply noise
        vel_max = 127
        for i, note in enumerate(new_midi.instruments[0].notes):
            shift_s = (time_shifts[i] + error_shifts[i]) / 1000
            note.start += shift_s
            note.end += shift_s
            note.velocity = int(
                np.clip(velocity_shaper(note.velocity / vel_max) * vel_max + velocity_shifts[i], 0, vel_max))

        # Write midi
        new_midi.write(str(out_path))

    def augment_midi_dataset(
            self,
            dataset: Path | list[Track],
            out_path: Path,
            include_original_track: bool = True,
            num_variations_per_track: int | None = None,
            only_if_doesnt_exist: bool = False,
            seed: int | None = None,
    ) -> dict[Path, list[Path]]:
        """
        Augment a MIDI dataset by generating variations of each track.
        :param dataset: Path to the MIDI dataset or list of tracks.
        :param out_path: Output directory. The structure of the dataset will be preserved.
        :param include_original_track: Whether to include the original track in the output.
        :param num_variations_per_track: Number of variations to generate for each track.
        :param only_if_doesnt_exist: Whether to skip files that already have an associated output directory (it doesn't
            mean that the chunks have already been computed, an error could have occurred during the previous process).
        :param seed: Seed for the random number generator.
        :return: Dictionary mapping each MIDI path to its variations.
        """
        # Prepare output
        out_path.mkdir(parents=True, exist_ok=True)
        augmented_file_paths = defaultdict(list)

        # Set default parameters
        if num_variations_per_track is None:
            num_variations_per_track = self.num_variations_per_track
        if seed is None:
            seed = self.seed
        np.random.seed(seed)

        # Loop over MIDI files
        if isinstance(dataset, list):
            midi_paths = [track.midi_path for track in dataset]
            midi_sizes = [path.stat().st_size for path in midi_paths]
            root_path = find_common_root(midi_paths)
        elif isinstance(dataset, Path):
            midi_paths, midi_sizes = get_midi_paths(dataset, return_sizes=True)
            root_path = dataset
        else:
            raise TypeError(f"dataset must be a Path or a list of Tracks, not {type(dataset)}")
        seeds = np.random.randint(0, 2**32-1, (len(midi_paths), num_variations_per_track))
        pbar = tqdm(
            midi_paths,
            desc=f"Augmenting MIDI files into {num_variations_per_track} variations each",
            total=sum(midi_sizes),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        for i, (midi_path, midi_size) in enumerate(zip(midi_paths, midi_sizes)):
            base_path = out_path / midi_path.relative_to(root_path).with_suffix('')
            for variation in range(num_variations_per_track):
                current_augmented_path = base_path / f"{midi_path.stem}_variation={variation}.mid"
                augmented_file_paths[midi_path].append(current_augmented_path)
                if only_if_doesnt_exist and current_augmented_path.exists():
                    continue
                current_augmented_path.parent.mkdir(parents=True, exist_ok=True)
                self.augment_midi_file(midi_path, current_augmented_path, seed=int(seeds[i, variation]))
            if include_original_track:
                out_file = base_path / f"{midi_path.stem}_variation=original.mid"
                shutil.copy(midi_path, out_file)
            pbar.update(midi_size)
        pbar.close()
        return augmented_file_paths


if __name__ == '__main__':
    midi_path = Path('data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid')
    # midi_path = Path('data/groove_augmented/drummer5/session1/8_latin-venezuelan-merengue_162_beat_5-8/8_latin-venezuelan-merengue_162_beat_5-8_variation=7.mid')
    # midi_path = Path('data/groove/drummer10/session1/1_jazz-swing_124_beat_4-4.mid')
    # midi_path = Path('groove_light/drummer1/eval_session/4_soul-groove4_80_beat_4-4.mid')
    # midi_path = Path('groove/drummer7/session2/22_rock_118_beat_4-4.mid')
    # midi_path = Path('groove/drummer2/session3/1_rock-shuffle_85_beat_4-4.mid')
    # midi_path = Path('groove/drummer1/session2/111_rock-prog_110_beat_5-4.mid')
    augmentor = MidiAugmentor(num_variations_per_track=5, gaussian_parameters_for_tempo_variation=(0, 30))
    # splitter(midi_path, out_path=Path('/tmp/tests'))
    augmentor.augment_midi_dataset(midi_path.parent, out_path=Path('/tmp/tests'), only_if_doesnt_exist=True)
