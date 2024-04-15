import json
from pathlib import Path

import pyqparse

from grammar import Grammar
from grammar import Node
from grammar import build_scoring_env

pyqparse.set_verbosity_level(0)


def main(
    midi_path: Path,
    tempo: int,
    time_signature: tuple[int, int],
    grammar_file: Path,
):
    scoring_env = build_scoring_env(
        path=midi_path,
        tempo=tempo,
        time_signature=time_signature,
        grammar_file=grammar_file,
        track_nb=0,
    )
    raw_tree = scoring_env.best_tree(pyqparse.Model.DrumAgostini, 1)
    scoring_env.close_score()
    # tree = Node(raw_tree, Grammar.from_file(grammar_file))
    tree = Node.from_qparse(raw_tree, Grammar.from_file(grammar_file), flatten_measures=True)
    tree_str = json.dumps(
        tree.to_dict(),
        separators=(",", ":"),
    )
    print(tree_str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("midi_path", type=Path)
    parser.add_argument("--tempo", type=int, required=True)
    parser.add_argument("--time-signature", type=int, nargs=2, required=True)
    parser.add_argument("--grammar-file", type=Path, required=True)
    args = parser.parse_args()
    main(
        midi_path=args.midi_path,
        tempo=args.tempo,
        time_signature=tuple(args.time_signature),
        grammar_file=args.grammar_file,
    )
