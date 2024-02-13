import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Self

import pyqparse

sys.setrecursionlimit(10000)  # to avoid RecursionError when parsing large trees

FIRST_MEASURE_RULE = 35


class Node:
    _frozen = False

    def __init__(self, pyqparse_tree: pyqparse.LRT, fix_production_rules: bool = True):
        self._pyqparse_tree = pyqparse_tree
        self.production_rule_id = self._pyqparse_tree.production()
        self.children = [
            self.__class__(self._pyqparse_tree.child(i), fix_production_rules=False)
            for i in range(self._pyqparse_tree.nbc())
        ]
        has_midi_events = self._pyqparse_tree.has_jam()
        self.midi_start_index = self._pyqparse_tree.label_first() if has_midi_events else None
        self.midi_num_events = self._pyqparse_tree.label_length() if has_midi_events else None
        if fix_production_rules:
            self._fix_production_rules()
        self._frozen = True

    @property
    def is_measure_split(self) -> bool:
        """
        Whether the node is a measure split.
        :return: Whether the node is a measure split.
        """
        return self.production_rule_id >= FIRST_MEASURE_RULE or self.production_rule_id == -1

    def __setattr__(self, key, value):
        if self._frozen and key not in self.__dict__:
            raise AttributeError(
                f"Cannot set attribute {key} in frozen {self.__class__.__name__} except using"
                " .update_production_rule_id()."
            )
        super().__setattr__(key, value)

    @property
    def nbc(self) -> int:
        """
        Get the number of children.
        :return: Number of children.
        """
        return len(self.children)

    @property
    def is_terminal(self) -> bool:
        """
        Whether the node is terminal.
        :return: Whether the node is terminal.
        """
        return self.nbc == 0

    @property
    def is_last(self) -> bool:
        """
        Whether the node is the last node of the tree.
        :return: Whether the node is the last node of the tree.
        """
        return self.is_terminal and not self.has_midi_events

    @property
    def production_rule_explicit(self) -> str:
        """
        Get a string representation of the production rule.
        :return: String representation of the production rule.
        """
        raise NotImplementedError

    @property
    def has_midi_events(self) -> bool:
        """
        Whether the node has MIDI events.
        :return: Whether the node has MIDI events.
        """
        return self.midi_start_index is not None and self.midi_num_events is not None

    @property
    def midi_last_index(self) -> int:
        """
        Get the last MIDI index of the node (exclusive).
        :return: Last MIDI index of the node (exclusive).
        """
        return self.midi_start_index + self.midi_num_events

    @property
    def midi_last_index_inclusive(self) -> int:
        """
        Get the last MIDI index of the node (inclusive).
        :return: Last MIDI index of the node (inclusive).
        """
        return self.midi_last_index - 1

    def _fix_production_rules(self, rules_mapping: Callable[[int], int] = None):
        self.update_production_rule_id(FIRST_MEASURE_RULE, update_children=True)
        if rules_mapping is None:
            rules_mapping = _rules_mapping

        def update_recursively(node: Node):
            node.production_rule_id = rules_mapping(node)
            for child in node.children:
                update_recursively(child)

        update_recursively(self)

    def update_production_rule_id(self, new_id: int, update_children: bool = True):
        """
        Update the production rule ID, and optionally update the children's production rule IDs.
        :param new_id: New production rule ID.
        :param update_children: Whether to update the children's production rule IDs. If True, the
            children's production rule IDs are updated to child_id + new_id - old_id.
        :return:
        """

        def modify_recursively(node: Node, offset: int):
            node.production_rule_id += offset
            if update_children:  # use update_children from the outer scope
                for child in node.children:
                    modify_recursively(child, offset)

        offset = new_id - self.production_rule_id
        modify_recursively(self, offset)

    def traverse_tree(self, function: Callable[[Self], Any] = None) -> list[Any]:
        """
        Traverse the tree in a prefix order, eventually getting a value for a function to each node.
        :param function: Function to apply to each node. If None, the node is returned.
        :return: List of nodes (or values returned by the function).
        """
        if function is None:
            function = lambda x: x
        return sum(
            [child.traverse_tree(function) for child in self.children],
            [function(self)],
        )

    def get_production_rule_ids(self) -> list[int]:
        """
        Get the production rule IDs of the tree.
        :return: List of production rule IDs.
        """
        return self.traverse_tree(lambda n: n.production_rule_id)

    def get_production_rule_ids_and_midi_indices(self) -> list[tuple[int, tuple[int, int] | None]]:
        """
        Get the production rule IDs and MIDI indices of the tree.
        :return: List of tuples (production rule ID, MIDI indices).
        """

        def get_info_for_one_node(node: Self) -> tuple[int, tuple[int, int] | None]:
            label_slice = (
                (node.midi_start_index, node.midi_last_index) if node.has_midi_events else None
            )
            return node.production_rule_id, label_slice

        return self.traverse_tree(get_info_for_one_node)

    def print_tree(self, indent=0, is_last_child=False):
        prefix = "│  " * indent
        if is_last_child:
            print(prefix + "└─", self.production_rule_id)
            prefix = "   " * indent
        else:
            print(prefix + "├─", self.production_rule_id)

        for i, child in enumerate(self.children):
            next_indent = indent + 1
            is_last = i == len(self.children) - 1
            child.print_tree(next_indent, is_last)

    def to_expanded_tree_string(self, display_midi_indices: bool = False) -> str:
        """
        Get a string representation of the tree.
        :param display_midi_indices: Whether to display the MIDI indices after the production rule.
        :return: String representation of the tree.
        """

        def recursive_part(
            node: Self,
            prefix: str = "",
            is_last: bool = False,
        ):
            string = prefix + ("└─ " if is_last else "├─ ") + str(node.production_rule_id)
            if node.is_measure_split:
                string = str(node.production_rule_id)
                prefix = ""
            if display_midi_indices and node.is_terminal:
                if not node.is_last:
                    string += f"  [{node.midi_start_index}-{node.midi_last_index}]"
                else:
                    string += f" [END OF TRACK]"
            num_children = len(node)
            for i in range(num_children):
                child = node[i]
                new_prefix = prefix + ("   " if is_last else "│  ")
                is_last_child = i == num_children - 1
                string += "\n" + recursive_part(child, new_prefix, is_last_child)
            return string

        return recursive_part(self)

    def dump_rules(self, file_path: Path) -> None:
        """
        Dump the production rule IDs and MIDI indices to a JSON file as a list of :
        (production rule ID, (MIDI start index, MIDI last index)).
        :param file_path: Path to the JSON file.
        """
        content = self.get_production_rule_ids_and_midi_indices()
        with open(file_path, "w") as f:
            json.dump(content, f, separators=(",", ":"))  # separators to remove spaces

    def __str__(self):
        return self.to_expanded_tree_string(display_midi_indices=True)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.production_rule_id}, nbc={self.nbc},"
            f" terminal={self.is_terminal}, tree={self._pyqparse_tree})"
        )

    def __len__(self):
        return self.nbc

    def __getitem__(self, item):
        return self.children[item]


def _rules_mapping(node: Node):
    if node.is_measure_split:
        return -1
    elif node.is_terminal:
        return -2
    else:
        return node.production_rule_id


# def split_tree_into_measures(node: pyqparse.LRT) -> list[pyqparse.LRT]:
#     """
#     Split a tree into a list of trees, each tree corresponding to a measure.
#     :param node: Tree to split.
#     :return: List of trees.
#     """
#     assert node.production() == FIRST_MEASURE_RULE, f"Node {node} should be the root node."
#     measures: list[pyqparse.LRT] = []
#     while not is_node_last(node):
#         assert node.nbc() == 2, f"Node {node} should have exactly 2 children ({node.nbc()} found)."
#         measures.append(node)
#         measures.append(node.child(0))
#         node = node.child(1)
#     return measures


def build_scoring_env(
    path: Path,
    tempo: int,
    time_signature: tuple[int, int],
    grammar_file: Path,
) -> pyqparse.ScoringEnv:
    # Setup basic options for scoring environment
    scoring_env = pyqparse.ScoringEnv()
    scoring_env.set_order(False)  # To get the best trees (equivalent of option -max)
    scoring_env.set_trs(pyqparse.Mode.Drum)
    scoring_env.set_debug(False)
    # Load grammar file
    scoring_env.import_WTA(str(grammar_file), pyqparse.WeightCode.Tropical)
    # Set time signature
    assert time_signature == (4, 4)  # Only 4/4 supported for now
    scoring_env.update_time_sig(time_signature[0], time_signature[1])
    # Set tempo
    num_beat_in_measure = scoring_env.get_bar_beats()
    measure_duration = (60 / tempo) * num_beat_in_measure
    scoring_env.set_tempo(mod=pyqparse.Tempo.Constant, bardur=measure_duration)
    # Open file
    scoring_env.import_file(
        filename=str(path),
        format=pyqparse.Format.MIDI,
        mode=pyqparse.Mode.Drum,
        tracknb=0,
    )
    return scoring_env


if __name__ == "__main__":
    MIDI_FILE = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
    GRAMMAR_FILE = Path("utils/grammar.wta")

    bpm = int(re.search(r"\d+_[a-z0-9\-]+_(\d+|\d+\.\d+)_.*\.mid", MIDI_FILE.name).group(1))

    scoring_env = build_scoring_env(MIDI_FILE, bpm, (4, 4), GRAMMAR_FILE)
    scoring_env.export_MEI("/tmp/test.mei")

    raw_tree = scoring_env.best_tree(pyqparse.Model.DrumAgostini, 1)
    tree = Node(raw_tree)
    print(tree.to_expanded_tree_string())
    breakpoint()
