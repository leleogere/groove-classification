import copy
import json
from pathlib import Path
import shutil

import numpy as np
import yaml
from pretty_midi import PrettyMIDI
from tqdm import tqdm
import scipy.stats as stats


def truncated_normal(mean: float = 0, std: float = 1, a: float = -np.inf, b: float = np.inf, size: int = 1):
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


def augment_midi(
        midi_path: Path,
        out_file: Path,
        std_time_shift: float = 0.02,
        global_velocity_shape: float = 0,
        std_velocity_shift: float = 5,
        proba_error: float = 0.1,
        mean_error_time_shift: float = 0,
        std_error_time_shift: float = 0.1,
        tempo_variation: float = 0,
        seed: int = None,
        _global_velocity_shape_force: float = 2,
):
    """
    Augment a MIDI file.

    :param midi_path: Path to the MIDI file.
    :param out_file: Path to the augmented MIDI file.
    :param std_time_shift: Standard deviation of the time shift of each note (in proportion of ms per beat = 60000/BPM).
        For a track at 120 BPM, std_time_shift = 0.02 will result in a time shift of 10 ms on average. For 60 BPM, it
        will result in a time shift of 20 ms on average (slower tracks allow for more time shift as it will be less
        noticeable).
    :param global_velocity_shape: Value in [-1, 1] that will be used to shape the gamma distribution applied to the
        velocity of each note. Negative values will reduce the velocity of the notes, positive values will increase it.
        The velocity (normalised between 0 and 1) will be scaled as f(v) = v ** (force ** -global_velocity_shape).
        See parameter _global_velocity_shape_force to control the force value.
    :param std_velocity_shift: Standard deviation of the velocity shift of each note (applied after global shaping).
    :param proba_error: Probability of simulating an error for a note (strong shift in time).
    :param mean_error_time_shift: Mean time shift of an error (in proportion of ms per beat).
    :param std_error_time_shift: Standard deviation of the time shift of an error (in proportion of ms per beat)
    :param tempo_variation: Variation of the tempo (in BPM). The value will be added to the original tempo. Note that
        the tempo is changed before applying time shift augmentations.
    :param seed: Seed for the random number generator.
    :param _global_velocity_shape_force: Force of the velocity shaping. The higher the force, the more the normalised
        velocity will be impacted. For global_velocity_shape in [-1, 1], the velocity will be raised to a power in
        [1/force, force]. For example, for force = 2, global_velocity_shape = -1 will raise the velocity to the power
        1/2, and global_velocity_shape = 1 will raise the velocity to the power 2. You probably don't want to set this
        greater than 2 as it will lead to very high or very low velocities.
    """
    # Check distribution parameters
    if std_time_shift < 0:
        raise ValueError(f"std_time_shift must be >= 0, got {std_time_shift}.")
    if std_velocity_shift < 0:
        raise ValueError(f"std_velocity_shift must be >= 0, got {std_velocity_shift}.")
    if proba_error < 0 or proba_error > 1:
        raise ValueError(f"proba_error must be in [0, 1], got {proba_error}.")
    if std_error_time_shift < 0:
        raise ValueError(f"std_error_time_shift must be >= 0, got {std_error_time_shift}.")
    if global_velocity_shape < -1 or global_velocity_shape > 1:
        raise ValueError(f"global_velocity_shape must be in [-1, 1], got {global_velocity_shape}.")
    if _global_velocity_shape_force < 1:
        raise ValueError(f"_global_velocity_shape_force must be >= 1, got {_global_velocity_shape_force}.")

    # Read midi
    if not midi_path.exists():
        raise FileNotFoundError(f'File {midi_path} does not exist.')
    midi = PrettyMIDI(str(midi_path))


    # Check tempo
    tempo_timestamp, bpm = midi.get_tempo_changes()
    if len(tempo_timestamp) != 1:
        raise ValueError(f"Expected 1 tempo change, got {len(tempo_timestamp)} in {midi_path}.")
    bpm = bpm[0]

    # Check instrument
    if len(midi.instruments) != 1 and midi.instruments[0].is_drum:
        raise ValueError(f"Expected 1 unique drum track, got {midi.instruments} in {midi_path}.")

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
    velocity_shaper = lambda x: x ** (_global_velocity_shape_force ** -global_velocity_shape)
    velocity_shifts = np.random.normal(0, std_velocity_shift, num_notes)
    error_shifts = np.random.normal(mean_error_time_shift * mspb, std_error_time_shift * mspb, num_notes)
    error_shifts[np.random.rand(num_notes) > proba_error] = 0

    # Apply noise
    vel_max = 127
    for i, note in enumerate(new_midi.instruments[0].notes):
        # print(">", note)
        shift_s = (time_shifts[i] + error_shifts[i]) / 1000
        note.start += shift_s
        note.end += shift_s
        note.velocity = int(np.clip(velocity_shaper(note.velocity/vel_max) * vel_max + velocity_shifts[i], 0, vel_max))
        # print(" ", note)

    # Write midi
    new_midi.write(str(out_file))


def augment_midi_dataset(
        dataset_path: Path,
        out_path: Path,
        num_variations_per_track: int = 1,
        include_original_track: bool = True,
        trunc_gaussian_parameters_for_std_time_shift: tuple[float, float] = (0.02, 0.005),
        trunc_gaussian_parameters_global_velocity_shape: tuple[float, float] = (0, 0.15),
        trunc_gaussian_parameters_for_std_velocity_shift: tuple[float, float] = (3, 1),
        trunc_gaussian_parameters_for_proba_error: tuple[float, float] = (0.07, 0.02),
        gaussian_parameters_for_mean_error_shift: tuple[float, float] = (0.1, 0.03),
        trunc_gaussian_parameters_for_std_error_shift: tuple[float, float] = (0.04, 0.01),
        gaussian_parameters_for_tempo_variation: tuple[float, float] = (0, 7),
        seed: int = None,
):
    """
    Augment a MIDI dataset.

    :param dataset_path: Path to the dataset.
    :param out_path: Path to the augmented dataset.
    :param num_variations_per_track: Number of augmented tracks to generate per track.
    :param include_original_track: Whether to include the original track in the augmented dataset.
    :param trunc_gaussian_parameters_for_std_time_shift: Parameters of the gaussian used to sample parameters for
        argument `std_time_shift_ms` of `augment_midi`.
    :param trunc_gaussian_parameters_global_velocity_shape: Parameters of the truncated gaussian used to sample
        parameters for argument `gamma_velocity_factor` of `augment_midi`.
    :param trunc_gaussian_parameters_for_std_velocity_shift: Parameters of the truncated gaussian used to sample
        parameters for argument `std_velocity_shift` of `augment_midi`.
    :param trunc_gaussian_parameters_for_proba_error: Parameters of the truncated gaussian used to sample parameters for
        argument `proba_error` of `augment_midi`.
    :param gaussian_parameters_for_mean_error_shift: Parameters of the gaussian used to sample parameters for
        argument `mean_error_shift` of `augment_midi`.
    :param trunc_gaussian_parameters_for_std_error_shift: Parameters of the truncated gaussian used to sample
        parameters for argument `std_error_shift` of `augment_midi`.
    :param gaussian_parameters_for_tempo_variation: Parameters of the gaussian used to sample parameters for
        argument `tempo_variation` of `augment_midi`.
    :param seed: Seed for the random number generator.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    midi_paths = list(dataset_path.glob("**/*.mid"))
    midi_sizes = [midi_path.stat().st_size for midi_path in midi_paths]
    if seed is not None:
        np.random.seed(seed)
    seeds = np.random.randint(0, 2**32-1, (len(midi_paths), num_variations_per_track))
    pbar = tqdm(
        midi_paths,
        desc="Augmenting MIDI files",
        total=sum(midi_sizes),
        unit="B",
        unit_scale=True,
        unit_divisor=1024
    )
    for i, midi_path in enumerate(pbar):
        pbar.set_postfix({"Path": midi_path}, refresh=False)
        base_path = out_path / midi_path.relative_to(dataset_path).with_suffix('')
        base_path.mkdir(parents=True, exist_ok=True)
        augmentation_parameters = dict(original_file=str(midi_path), variations={})
        for j in range(num_variations_per_track):
            out_file = base_path / f"{midi_path.stem}_variation={j}.mid"
            kwargs = dict(
                std_time_shift=truncated_normal(*trunc_gaussian_parameters_for_std_time_shift, a=0, b=np.inf),
                global_velocity_shape=truncated_normal(*trunc_gaussian_parameters_global_velocity_shape, a=-1, b=1),
                std_velocity_shift=truncated_normal(*trunc_gaussian_parameters_for_std_velocity_shift, a=0, b=np.inf),
                proba_error=truncated_normal(*trunc_gaussian_parameters_for_proba_error, a=0, b=1),
                mean_error_time_shift=np.random.normal(*gaussian_parameters_for_mean_error_shift),
                std_error_time_shift=truncated_normal(*trunc_gaussian_parameters_for_std_error_shift, a=0, b=np.inf),
                tempo_variation=np.random.normal(*gaussian_parameters_for_tempo_variation),
                seed=int(seeds[i, j]),  # int as numpy int64 is not serializable
            )
            augment_midi(midi_path=midi_path, out_file=out_file, **kwargs)
            augmentation_parameters['variations'][out_file.name] = kwargs
        if include_original_track:
            out_file = base_path / f"{midi_path.stem}_variation=original.mid"
            shutil.copy(midi_path, out_file)
        with open(base_path / f"{midi_path.stem}_augmentation_parameters.json", 'w') as f:
            json.dump(augmentation_parameters, f, indent=4)
        pbar.update(midi_sizes[i])

        



if __name__ == '__main__':
    # midi_path = Path("data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid")
    # out_path = Path("/tmp/augmented") / midi_path.name
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # augment_midi(
    #     midi_path=midi_path,
    #     out_file=out_path,
    #     std_time_shift=0.02,
    #     global_velocity_shape=0.5,
    #     std_velocity_shift=0,
    #     proba_error=0.1,
    #     mean_error_time_shift=0,
    #     std_error_time_shift=0.1,
    #     tempo_variation=80,
    #     seed=24,
    # )

    midi_dataset_path = Path("data/groove")
    out_path = Path("/tmp/augmented")
    augment_midi_dataset(
        dataset_path=midi_dataset_path,
        out_path=out_path,
        num_variations_per_track=2,
        include_original_track=True,
        trunc_gaussian_parameters_for_std_time_shift=(0.02, 0.005),
        trunc_gaussian_parameters_global_velocity_shape=(0, 0.15),
        trunc_gaussian_parameters_for_std_velocity_shift=(3, 1),
        trunc_gaussian_parameters_for_proba_error=(0.07, 0.02),
        gaussian_parameters_for_mean_error_shift=(0.1, 0.03),
        trunc_gaussian_parameters_for_std_error_shift=(0.04, 0.01),
        gaussian_parameters_for_tempo_variation=(0, 7),
        seed=24,
    )
