import copy
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

    def __init__(self, basic_rules: dict[int, Rule], max_depth: int = 7):
        self.basic_rules = basic_rules
        self._create_simplified_ids()
        self.max_depth = max_depth  # TODO: compute it automatically from the grammar?

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
    _id_key = "i"
    _children_key = "c"
    _start_key = "s"
    _end_key = "e"

    def __init__(
        self,
        grammar: Grammar,
        id_: int,
        children: list[Self],
        midi_start_index: int = None,
        midi_last_index: int = None,
    ):
        assert not (midi_start_index is None) ^ (midi_last_index is None)
        self.grammar = grammar
        self.id = id_
        self.children = children
        self.midi_start_index = midi_start_index
        self.midi_last_index = midi_last_index

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.id}, children={len(self.children)},"
            f" midi=({self.midi_start_index}, {self.midi_last_index}))"
        )

    @property
    def nbc(self) -> int:
        """
        Get the number of children.
        :return: Number of children.
        """
        return len(self.children)

    def __getstate__(self):
        # Prevent pickling the qparse tree as it is not serializable
        state = {k: v for k, v in self.__dict__.items() if k != "_qparse_tree"}
        return state

    @property
    def has_midi_events(self) -> bool:
        """
        Whether the node has MIDI events.
        :return: Whether the node has MIDI events.
        """
        return self.midi_start_index is not None

    def traverse(self, fn: Callable[[Self], Any]) -> list[Any]:
        """
        Traverse the tree in a prefix order, returning the result of the function applied to each
        node in a prefix depth-first order.

        :param fn: Function to apply to each node.
        :return: List of results of the function applied to each node as a list.
        """
        return [fn(self)] + sum([child.traverse(fn) for child in self.children], [])

    def get_positions(self) -> list[list[int]]:
        """
        Get the positions of each node in the tree. The positions are returned in a prefix order
        depth-first order. For each node, the position is a list in which the first half indicates
        the index of the node of all its parents from the root, and the second half indicates the
        number of children of all its parents from the root. If the node is less deep than the
        maximum depth of the grammar, it is padded with zeros.

        For example, the position [3, 1, 2, 0, 0, 7, 3, 2, 0, 0] should be split into
        [3, 1, 2, 0, 0] and [7, 3, 2, 0, 0], and it means that this node can be reached from the
        root by going to the 3rd child of the 7 children of the root, then the 1st child of the 3
        children of the previous node, and finally the last child of the 2 children of the previous
        node. The last node has 0 children, so lists are padded with zeros.

        :return: List of positions.
        """

        def get_positions_for_one_node(
            node: Self,
            child_indices_so_far: list[int],
            child_numbers_so_far: list[int],
        ):
            assert len(child_indices_so_far) == len(child_numbers_so_far)
            child_positions = sum(
                [
                    get_positions_for_one_node(
                        child,
                        child_indices_so_far + [i],
                        child_numbers_so_far + [node.nbc],
                    )
                    for i, child in enumerate(node.children, start=1)
                ],
                [],
            )
            padded_and_concat = (
                child_indices_so_far
                + [0] * (self.grammar.max_depth - len(child_indices_so_far))
                + child_numbers_so_far
                + [0] * (self.grammar.max_depth - len(child_numbers_so_far))
            )
            return [padded_and_concat] + child_positions

        return get_positions_for_one_node(self, [], [])

    def to_dict(self) -> dict[str, Any]:
        d = {self._id_key: self.id}
        if self.children:
            d[self._children_key] = [child.to_dict() for child in self.children]
        if self.midi_start_index is not None:
            d[self._start_key] = self.midi_start_index
            d[self._end_key] = self.midi_last_index
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], grammar: Grammar) -> Self:
        return cls(
            grammar=grammar,
            id_=d[cls._id_key],
            children=[cls.from_dict(child, grammar) for child in d.get(cls._children_key, [])],
            midi_start_index=d.get(cls._start_key),
            midi_last_index=d.get(cls._end_key),
        )

    def save(self, file_path: Path) -> None:
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, file_path: Path, grammar: Grammar) -> Self:
        with open(file_path, "r") as f:
            return cls.from_dict(json.load(f), grammar)

    @classmethod
    def from_qparse(
        cls,
        tree: pyqparse.LRT,
        grammar: Grammar,
        flatten_measures: bool = True,
    ) -> Self:
        if tree.fail():
            raise ValueError(f"Parsing failed in {tree}")
        id_ = grammar.simplify_rule(tree.production())
        children = [
            cls.from_qparse(tree.child(i), grammar=grammar, flatten_measures=False)
            for i in range(tree.nbc())
        ]
        has_midi_events = tree.has_jam()
        midi_start_index = tree.label_first() if has_midi_events else None
        midi_last_index = midi_start_index + tree.label_length() if has_midi_events else None
        root_node = cls(
            grammar=grammar,
            id_=id_,
            children=children,
            midi_start_index=midi_start_index,
            midi_last_index=midi_last_index,
        )
        root_node._qparse_tree = tree
        root_node._qparse_id = tree.production()
        # Gather all measures under the root node (instead of deeper and deeper children)
        if flatten_measures and root_node.id == grammar.SIMPLE_MEASURE_SPLIT_RULE.simple_id:
            measures = []
            current_node = root_node
            while current_node.id == grammar.SIMPLE_MEASURE_SPLIT_RULE.simple_id:
                assert (
                    len(current_node.children) == 2
                ), f"Measure split nodes should have 2 children ({len(root_node.children)} found)."
                measures.append(current_node.children[0])
                current_node = current_node.children[1]
            root_node.children = measures
        return root_node

    @classmethod
    def from_midi(
        cls,
        midi_path: Path,
        tempo: int,
        time_signature: tuple[int, int],
        grammar_file: Path,
    ) -> Self:
        scoring_env = build_scoring_env(midi_path, tempo, time_signature, grammar_file)
        raw_tree = scoring_env.best_tree(pyqparse.Model.DrumAgostini, 1)
        return cls.from_qparse(raw_tree, Grammar.from_file(grammar_file))

    def to_string_top_down(
        self,
        node_representation: Callable[[Self], str] = lambda n: str(n.id),
    ) -> str:
        """
        Get a string representation of the tree in a top-down fashion.
        Adapted from https://stackoverflow.com/questions/15675261/displaying-a-tree-in-ascii
        :return: the string representation of the tree.
        """

        def block_width(block):
            try:
                return block.index("\n")
            except ValueError:
                return len(block)

        def stack_str_blocks(blocks):
            """Takes a list of multiline strings, and stacks them horizontally.

            For example, given 'aaa\naaa' and 'bbbb\nbbbb', it returns
            'aaa bbbb\naaa bbbb'.  As in:

            'aaa  +  'bbbb   =  'aaa bbbb
             aaa'     bbbb'      aaa bbbb'

            Each block must be rectangular (all lines are the same length), but blocks
            can be different sizes.
            """

            builder = []
            block_lens = [block_width(bl) for bl in blocks]
            split_blocks = [bl.split("\n") for bl in blocks]

            import itertools

            for line_list in itertools.zip_longest(*split_blocks, fillvalue=None):
                for i, line in enumerate(line_list):
                    if line is None:
                        builder.append(" " * block_lens[i])
                    else:
                        builder.append(line)
                    if i != len(line_list) - 1:
                        builder.append(" ")  # Padding
                builder.append("\n")

            return "".join(builder[:-1])

        name = node_representation(self)

        if not self.children:
            return name

        child_strs = [child.to_string_top_down(node_representation) for child in self.children]
        child_widths = [block_width(s) for s in child_strs]

        # How wide is this block?
        display_width = max(len(name), sum(child_widths) + len(child_widths) - 1)

        # Determines midpoints of child blocks
        child_midpoints = []
        child_end = 0
        for width in child_widths:
            child_midpoints.append(child_end + (width // 2))
            child_end += width + 1

        # Builds up the brace, using the child midpoints
        brace_builder = []
        for i in range(display_width):
            if i < child_midpoints[0] or i > child_midpoints[-1]:
                brace_builder.append(" ")
            elif i in child_midpoints:
                if i == child_midpoints[0]:
                    brace_builder.append("┌")
                elif i == child_midpoints[-1]:
                    brace_builder.append("┐")
                else:
                    brace_builder.append("┬")
            else:
                brace_builder.append("─")
        brace = "".join(brace_builder)

        name_str = "{:^{}}".format(name, display_width)
        below = stack_str_blocks(child_strs)

        return name_str + "\n" + brace + "\n" + below

    def to_string_left_right(
        self,
        node_representation: Callable[[Self], str] = lambda n: str(n.id),
    ) -> str:
        """
        Get a string representation of the tree in a left-right fashion.

        :return: the string representation of the tree.
        """

        def recursive_part(node: Self, prefix: str = "", is_last: bool = False):
            if is_last:
                new_prefix = prefix + "└─ "
                next_prefix = prefix + "   "
            else:
                new_prefix = prefix + "├─ "
                next_prefix = prefix + "│  "
            string = new_prefix + node_representation(node) + "\n"
            for i, child in enumerate(node.children):
                string += recursive_part(child, next_prefix, i == node.nbc - 1)
            return string

        return recursive_part(self, is_last=True)


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
    scoring_env.set_order(True)  # To get the best trees (equivalent of option -max)
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
    MIDI_FILE = Path("../groove_data/groove/drummer1/session1/1_funk_80_beat_4-4.mid")
    # MIDI_FILE = Path(
    #     "/tmp/training_tests/processed/augmented_num=0/groove/drummer4/session1/3_jazz-klezmer_152_beat_4-4/3_jazz-klezmer_152_beat_4-4_variation=original.mid"
    # )
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
    # tree = Node(raw_tree, grammar)
    # print(tree.to_expanded_tree_string(simplified_ids=True))
    # pprint(tree.get_tree_as_dict(simplified_ids=True, flatten_measures=True), sort_dicts=False)
    # tree.dump_tree(Path("/tmp/tree.json"))
    tree = Node.from_qparse(raw_tree, grammar, flatten_measures=True)
    pprint(tree.to_dict(), sort_dicts=False)
    tree.save(Path("/tmp/tree.json"))
    print(tree.to_string_top_down())
    print(tree.to_string_left_right())
    pprint(tree.children)
    new_tree = Node.load(Path("/tmp/tree.json"), grammar)

    assert tree.to_dict() == new_tree.to_dict()

    import pickle

    with open("/tmp/tree.pkl", "wb") as f:
        pickle.dump(tree, f)
    with open("/tmp/tree.pkl", "rb") as f:
        new_tree = pickle.load(f)

    # print(tree._qparse_tree)
    assert tree.to_dict() == new_tree.to_dict()
    # breakpoint()
