import inspect
import json
import re
import sys
from collections import defaultdict
from collections import namedtuple
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Self

import pyqparse

sys.setrecursionlimit(10000)  # to avoid RecursionError when parsing large trees


class Rule:
    def __init__(self, id_: int, name: str, is_terminal: bool):
        self.id = id_
        self.name = name
        self.is_terminal = is_terminal

    def __repr__(self):
        return f"Rule(id={self.id}, name={self.name}, is_terminal={self.is_terminal})"


class Grammar:
    _SpecialRuleInfos = namedtuple(
        "RuleInfo", ["as_string", "simple_id", "offset_with_basic_rules"]
    )
    SIMPLE_TERMINAL_RULE_ID = _SpecialRuleInfos("TERMINAL", 0, None)
    SIMPLE_MEASURE_SPLIT_RULE = _SpecialRuleInfos("MEASURE_SPLIT", 1, 3)
    SIMPLE_END_OF_TRACK_RULE = _SpecialRuleInfos("END_OF_TRACK", 2, None)
    SIMPLE_FAIL_ZERO_RULE = _SpecialRuleInfos("FAIL_ZERO", 3, 0)
    SIMPLE_FAIL_INF_RULE = _SpecialRuleInfos("FAIL_INF", 4, 1)
    SIMPLE_SPECIAL_RULES = {
        SIMPLE_TERMINAL_RULE_ID,
        SIMPLE_MEASURE_SPLIT_RULE,
        SIMPLE_END_OF_TRACK_RULE,
        SIMPLE_FAIL_ZERO_RULE,
        SIMPLE_FAIL_INF_RULE,
    }

    def __init__(self, basic_rules: dict[int, Rule]):
        self.basic_rules = basic_rules
        self._create_simplified_ids()

    @property
    def first_measure_split_id(self) -> int:
        return len(self.basic_rules) + self.SIMPLE_MEASURE_SPLIT_RULE.offset_with_basic_rules

    def num_simplified_rules(self) -> int:
        return max(self._simplified_ids.values()) + 1

    def _create_simplified_ids(self):
        self._simplified_ids = {}
        current_idx = len(self.SIMPLE_SPECIAL_RULES)  # Skip ids for special rules
        for rule in self.basic_rules.values():
            if rule.is_terminal:
                self._simplified_ids[rule.id] = self.SIMPLE_TERMINAL_RULE_ID.simple_id
            else:
                self._simplified_ids[rule.id] = current_idx
                current_idx += 1

    def simplify_rule(self, rule_id: int) -> int:
        if rule_id in self._simplified_ids:
            return self._simplified_ids[rule_id]
        else:
            fail_zero_position = (
                len(self.basic_rules) + self.SIMPLE_FAIL_ZERO_RULE.offset_with_basic_rules
            )
            fail_inf_position = (
                len(self.basic_rules) + self.SIMPLE_FAIL_INF_RULE.offset_with_basic_rules
            )
            first_measure_split_position = (
                len(self.basic_rules) + self.SIMPLE_MEASURE_SPLIT_RULE.offset_with_basic_rules
            )
            if rule_id == fail_zero_position:
                simplified_id = self.SIMPLE_FAIL_ZERO_RULE[1]
            elif rule_id == fail_inf_position:
                simplified_id = self.SIMPLE_FAIL_INF_RULE[1]
            elif rule_id >= first_measure_split_position:
                if (rule_id - first_measure_split_position) % 2 == 0:
                    simplified_id = self.SIMPLE_MEASURE_SPLIT_RULE[1]
                else:
                    simplified_id = self.SIMPLE_END_OF_TRACK_RULE[1]
            else:
                raise ValueError(f"Unknown rule ID {rule_id}")
        return simplified_id

    @classmethod
    def from_file(cls, grammar_file: Path):
        BasicRule = namedtuple("BasicRule", ["as_string", "is_terminal"])
        pattern = re.compile(
            r"^\d+\s*->\s*\w+\s*(\(\d+(?:,\s*\d+)*\))?\s*\d+(?:\.\d*)?\s*(?://.*)?$"
        )
        basic_rules_list: list[BasicRule] = []
        with open(grammar_file, "r") as f:
            for line in f:
                if match := pattern.match(line):
                    is_terminal = match.group(1) is None
                    basic_rules_list.append(BasicRule(line.strip(), is_terminal))
        rules = {
            id_: Rule(id_, rule.as_string, rule.is_terminal)
            for id_, rule in enumerate(basic_rules_list)
        }
        return cls(rules)


class Node:
    _frozen = False

    def __init__(
        self,
        pyqparse_tree: pyqparse.LRT,
        grammar: Grammar,
    ):
        self._pyqparse_tree = pyqparse_tree
        self.grammar = grammar
        # self.rules = rules
        # self.rules_simplifier = rules_to_simplified_mapping(self.rules)
        self.production_rule_id = self._pyqparse_tree.production()
        self.children = [
            self.__class__(self._pyqparse_tree.child(i), grammar)
            for i in range(self._pyqparse_tree.nbc())
        ]
        has_midi_events = self._pyqparse_tree.has_jam()
        self.midi_start_index = self._pyqparse_tree.label_first() if has_midi_events else None
        self.midi_num_events = self._pyqparse_tree.label_length() if has_midi_events else None
        self._frozen = True
        if self._pyqparse_tree.fail():
            raise ValueError(f"Parsing failed in {self}")

    @property
    def is_measure_split(self) -> bool:
        """
        Whether the node is a measure split.
        :return: Whether the node is a measure split.
        """
        return self.production_rule_id >= self.grammar.first_measure_split_id

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

    # def production_id_simplified(self):
    #     return self.rules_simplifier[self.production_rule_id]

    # @property
    # def production_rule_explicit(self) -> str:
    #     """
    #     Get a string representation of the production rule.
    #     :return: String representation of the production rule.
    #     """
    #     if self.rules is not None:
    #         return self.rules.get(
    #             self.production_rule_id,
    #             f"No representation for rule {self.production_rule_id}",
    #         )
    #     else:
    #         raise ValueError("No rules mapping.")

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

    def get_production_rule_ids_and_midi_indices(
        self,
        simplified_ids: bool = True,
    ) -> list[tuple[int, tuple[int, int] | None]]:
        """
        Get the production rule IDs and MIDI indices of the tree.
        :return: List of tuples (production rule ID, MIDI indices).
        """

        def get_info_for_one_node(node: Self) -> tuple[int, tuple[int, int] | None]:
            label_slice = (
                (node.midi_start_index, node.midi_last_index) if node.has_midi_events else None
            )
            id_ = (
                node.production_rule_id
                if not simplified_ids
                else self.grammar.simplify_rule(node.production_rule_id)
            )
            return id_, label_slice

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

    def to_expanded_tree_string(
        self,
        display_midi_indices: bool = False,
        simplified_ids: bool = False,
    ) -> str:
        """
        Get a string representation of the tree.
        :param display_midi_indices: Whether to display the MIDI indices after the production rule.
        :param simplified_ids: Whether to display the simplified IDs.
        :return: String representation of the tree.
        """

        def recursive_part(
            node: Self,
            prefix: str = "",
            is_last: bool = False,
        ):
            id_ = (
                node.production_rule_id
                if not simplified_ids
                else self.grammar.simplify_rule(node.production_rule_id)
            )
            string = prefix + ("└─ " if is_last else "├─ ") + str(id_)
            if node.is_measure_split:
                string = str(id_)
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
        return self.to_expanded_tree_string(display_midi_indices=True, simplified_ids=True)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.production_rule_id}, nbc={self.nbc},"
            f" terminal={self.is_terminal}, tree={self._pyqparse_tree})"
        )

    def __len__(self):
        return self.nbc

    def __getitem__(self, item):
        return self.children[item]


# def read_rules(grammar_file: Path, additional_rules: dict[int, str] = None) -> dict[int, str]:
#     """
#     Read the production rules from a grammar file.
#     :param grammar_file: Path to the grammar file.
#     :param additional_rules: Additional rules to add to the dictionary.
#     :return: Dictionary of production rules.
#     """
#     pattern = re.compile(
#         r"^(\d+)\s*->\s*\w+\s*(?:\(\d+(?:,\s*\d+)*\))?\s*\d+(?:\.\d*)?\s*(?://.*)?$", flags=re.M
#     )
#     grouped_rules = {}
#     with open(grammar_file, "r") as f:
#         grouped_rules = defaultdict(list)
#         for line in f:
#             if match := pattern.match(line):
#                 grouped_rules[int(match.group(1))].append(line.strip())
#     # Sort the rules
#     sorted_rules = [rule for rule in sorted(grouped_rules.items(), key=lambda x: x[0])]
#     # Flatten the list of rules
#     flatten_rules = [rule for _, rules in sorted_rules for rule in rules]
#     # Rewrite rules with ID
#     rules = {i: rule for i, rule in enumerate(flatten_rules)}
#     if additional_rules is not None:
#         rules.update(additional_rules)
#     return rules


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
    track_nb: int = 0,
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
        tracknb=track_nb,
    )
    return scoring_env


if __name__ == "__main__":
    # MIDI_FILE = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
    MIDI_FILE = Path(
        "/tmp/training_tests/processed/augmented_num=0/groove/drummer4/session1/3_jazz-klezmer_152_beat_4-4/3_jazz-klezmer_152_beat_4-4_variation=original.mid"
    )
    GRAMMAR_FILE = Path("utils/grammar.wta")

    from pprint import pprint

    grammar = Grammar.from_file(GRAMMAR_FILE)
    pprint(grammar.basic_rules)
    print(f"{grammar.num_simplified_rules() = }")

    bpm = int(re.search(r"\d+_[a-z0-9\-]+_(\d+|\d+\.\d+)_.*\.mid", MIDI_FILE.name).group(1))

    scoring_env = build_scoring_env(MIDI_FILE, bpm, (4, 4), GRAMMAR_FILE)
    scoring_env.print_WTA()
    # scoring_env.export_MEI("/tmp/test.mei")

    raw_tree = scoring_env.best_tree(pyqparse.Model.DrumAgostini, 1)
    tree = Node(raw_tree, grammar)
    print(tree.to_expanded_tree_string(simplified_ids=True))
    breakpoint()
