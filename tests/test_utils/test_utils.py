import unittest
from importlib.resources import files
from pathlib import Path
import tempfile

import pretty_midi as pm
import mido
import symusic

from utils.utils import *


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.note = Note(start=0.3, end=1.6, pitch=60, velocity=100)
        self.track = Track(
            midi_path=Path(files("tests.resources").joinpath("midi.mid")),
            bpm=80,
        )

    def test_note(self):
        self.assertEqual(self.note.duration, 1.3)

    def test_relative_note(self):
        rnote = RelativeNote.from_note(self.note, timeshift=0.5)
        self.assertEqual(rnote.pitch, self.note.pitch)
        self.assertEqual(rnote.velocity, self.note.velocity)
        self.assertEqual(rnote.duration, self.note.duration)
        self.assertEqual(rnote.timeshift, 0.5)

    def test_compute_relative_notes(self):
        import random

        n = 10
        starts = [i + random.random() for i in range(n)]
        ends = [s + 2 * random.random() for s in starts]
        pitches = [random.randint(0, 127) for _ in range(n)]
        velocities = [random.randint(0, 127) for _ in range(n)]
        notes = [
            Note(start=s, end=e, pitch=p, velocity=v)
            for s, e, p, v in zip(starts, ends, pitches, velocities)
        ]
        rnotes = compute_relative_notes(notes)
        for i in range(n):
            self.assertEqual(rnotes[i].pitch, notes[i].pitch)
            self.assertEqual(rnotes[i].velocity, notes[i].velocity)
            self.assertEqual(rnotes[i].duration, notes[i].duration)
            ts = notes[i].start - notes[i - 1].start if i > 0 else 0
            self.assertEqual(rnotes[i].timeshift, ts)

    def test_compute_relative_notes_empty(self):
        rnotes = compute_relative_notes([])
        self.assertEqual(rnotes, [])

    def test_track(self):
        notes = self.track.notes()
        self.assertEqual(len(notes), 44)
        self.assertEqual(notes[0], Note(start=0.0, end=0.0921875, pitch=42, velocity=44))
        self.assertEqual(
            notes[3], Note(start=0.36250000000000004, end=0.4625, pitch=36, velocity=40)
        )
        self.assertEqual(
            notes[-1], Note(start=5.8328125, end=5.932812500000001, pitch=42, velocity=68)
        )

    def test_track_io(self):
        tmp_json = Path(tempfile.mktemp(suffix=".json"))
        self.track.to_json_file(tmp_json)
        track = Track.from_json_file(tmp_json)
        tmp_json.unlink()
        for attr in self.track.__dict__:
            self.assertEqual(getattr(self.track, attr), getattr(track, attr))
        for note, note2 in zip(self.track.notes(), track.notes()):
            self.assertEqual(note, note2)
        for rnote, rnote2 in zip(self.track.relative_notes(), track.relative_notes()):
            self.assertEqual(rnote, rnote2)

    def test_process_label(self):
        self.assertEqual(process_label("rock"), "rock")
        self.assertEqual(process_label("rock/funk"), "rock")
        self.assertEqual(process_label("rock/funk/pop"), "rock")
        self.assertEqual(process_label("rock/groove1-shuffle"), "rock")
        self.assertEqual(process_label("rock/shuffle", keep_sublabel_for=["rock"]), "rock/shuffle")

    def test_parse_bpm_from_filename(self):
        self.assertEqual(parse_bpm_from_file_name("1_funk-groove1_138_beat_4-4.mid"), 138)
        self.assertEqual(parse_bpm_from_file_name("1_funk-groove1_138_beat_4-4_m=1.mid"), 138)

    def test_parse_time_signature_from_filename(self):
        self.assertEqual(
            parse_time_signature_from_file_name("1_funk-groove1_138_beat_4-4.mid"), (4, 4)
        )
        self.assertEqual(
            parse_time_signature_from_file_name("1_funk-groove1_138_beat_4-4_m=1.mid"), (4, 4)
        )
        self.assertEqual(
            parse_time_signature_from_file_name("1_funk-groove1_138_beat_3-4.mid"), (3, 4)
        )
        self.assertEqual(
            parse_time_signature_from_file_name("1_funk-groove1_138_beat_6-8.mid"), (6, 8)
        )

    def test_parse_genre_from_file_name(self):
        self.assertEqual(parse_genre_from_file_name("1_funk-groove1_138_beat_4-4.mid"), "funk")
        self.assertEqual(parse_genre_from_file_name("1_funk-groove1_138_beat_4-4_m=1.mid"), "funk")
        self.assertEqual(
            parse_genre_from_file_name("1_funk-groove1_138_beat_4-4_m=1.mid", only_main=False),
            "funk-groove1",
        )

    def test_load_and_validate_midi_file(self):
        self.assertIsInstance(
            load_and_validate_midi_file(self.track.midi_path, backend="pretty_midi"), pm.PrettyMIDI
        )
        self.assertIsInstance(
            load_and_validate_midi_file(self.track.midi_path, backend="mido"), mido.MidiFile
        )
        self.assertIsInstance(
            load_and_validate_midi_file(self.track.midi_path, backend="symusic"), symusic.Score
        )
        self.assertRaises(
            ValueError, load_and_validate_midi_file, self.track.midi_path, backend="invalid"
        )

    def test_find_common_root(self):
        paths = [
            Path("a/b/c/d"),
            Path("a/b/c/e"),
            Path("a/b/f/g"),
        ]
        self.assertEqual(find_common_root(paths), Path("a/b"))

    def test_find_common_root_none(self):
        paths = [
            Path("a/b/c/d"),
            Path("x/y/z"),
        ]
        self.assertEqual(find_common_root(paths), Path(""))


if __name__ == "__main__":
    unittest.main()
