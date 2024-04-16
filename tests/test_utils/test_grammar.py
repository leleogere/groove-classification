import unittest
from importlib.resources import files

import numpy as np

from utils.utils import Track
from utils.grammar import *


class TestGrammar(unittest.TestCase):
    def setUp(self):
        self.grammar_file = Path(files("tests.resources").joinpath("grammar.wta"))
        self.grammar = Grammar.from_file(self.grammar_file)
        self.track = Track(
            midi_path=Path(files("tests.resources").joinpath("midi.mid")),
            bpm=80,
            time_signature=(4, 4),
        )

    def test_build_scoring_env(self):
        try:
            build_scoring_env(
                path=self.track.midi_path,
                tempo=self.track.bpm,
                time_signature=self.track.time_signature,
                grammar_file=self.grammar_file,
                track_nb=0,
            )
        except Exception as e:
            self.fail(f"build_scoring_env raised {e}")

    def build_tree(self):
        pyqparse.set_verbosity_level(0)
        return Node.from_midi(
            midi_path=self.track.midi_path,
            tempo=self.track.bpm,
            time_signature=self.track.time_signature,
            grammar_file=self.grammar_file,
        )

    def test_grammar(self):
        self.assertEqual(self.grammar.num_simplified_rules(), 15)
        self.assertEqual(len(self.grammar.basic_rules), 30)
        self.assertEqual(self.grammar.first_measure_split_id, 33)

    def test_simplify_rule(self):
        initial_rules = list(range(40))
        initial_rules.pop(32)  # TODO: investigate why this rule is not simplified
        # fmt: off
        simplified_rules = [0, 0, 0, 5, 6, 0, 0, 0, 7, 8, 0, 0, 0, 9, 10, 11, 0, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0, 0, 0, 14, 3, 4, 1, 2, 1, 2, 1, 2, 1]
        # fmt: on
        self.assertListEqual(
            [self.grammar.simplify_rule(r) for r in initial_rules],
            simplified_rules,
        )

    def test_tree(self):
        tree = self.build_tree()
        self.assertEqual(tree.nbc, 2)
        loaded_tree = Node.load(
            files("tests.resources").joinpath("midi_tree.json"), grammar=tree.grammar
        )
        self.assertEqual(tree, loaded_tree)

    def test_tree_strings(self):
        tree = self.build_tree()
        tree_td = "                           1                           \n         ┌───────────────────────────┐                 \n         5                           5                 \n      ┌─────────┐        ┌─────────────────┐           \n      7         7        7                 7           \n  ┌──────┐     ┌──┐    ┌─────┐     ┌───────────┐       \n  9     11     9  0   11     9    11          10       \n ┌──┐ ┌─┬─┬─┐ ┌─┐   ┌─┬─┬─┐ ┌─┐ ┌─┬─┬─┐   ┌────┬────┐  \n12  0 0 0 0 0 0 0   0 0 0 0 0 0 0 0 0 0  12   12   12  \n┌─┐                                     ┌──┐  ┌─┐  ┌──┐\n0 0                                     0 13  0 0 13  0\n                                          ┌─┐     ┌─┐  \n                                          0 0     0 0  "
        self.assertEqual(tree.to_string_top_down(), tree_td)
        tree_lr = "└─ 1\n   ├─ 5\n   │  ├─ 7\n   │  │  ├─ 9\n   │  │  │  ├─ 12\n   │  │  │  │  ├─ 0\n   │  │  │  │  └─ 0\n   │  │  │  └─ 0\n   │  │  └─ 11\n   │  │     ├─ 0\n   │  │     ├─ 0\n   │  │     ├─ 0\n   │  │     └─ 0\n   │  └─ 7\n   │     ├─ 9\n   │     │  ├─ 0\n   │     │  └─ 0\n   │     └─ 0\n   └─ 5\n      ├─ 7\n      │  ├─ 11\n      │  │  ├─ 0\n      │  │  ├─ 0\n      │  │  ├─ 0\n      │  │  └─ 0\n      │  └─ 9\n      │     ├─ 0\n      │     └─ 0\n      └─ 7\n         ├─ 11\n         │  ├─ 0\n         │  ├─ 0\n         │  ├─ 0\n         │  └─ 0\n         └─ 10\n            ├─ 12\n            │  ├─ 0\n            │  └─ 13\n            │     ├─ 0\n            │     └─ 0\n            ├─ 12\n            │  ├─ 0\n            │  └─ 0\n            └─ 12\n               ├─ 13\n               │  ├─ 0\n               │  └─ 0\n               └─ 0\n"
        self.assertEqual(tree.to_string_left_right(), tree_lr)

    def test_tree_positions(self):
        tree = self.build_tree()
        positions = tree.get_positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.load(files("tests.resources").joinpath("midi_tree_positions.npy")),
            )
        )


if __name__ == "__main__":
    unittest.main()
