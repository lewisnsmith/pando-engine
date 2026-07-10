"""Tests for the connectivity engine. Requires the real pretrained model:

    python -m gensim.downloader --download word2vec-google-news-300
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connectivity import (
    DEFAULT_THRESHOLD,
    build_solution_tree,
    get_pool,
    is_valid_puzzle,
    link,
    shortest_route,
    verify_sequence,
)

HERE = Path(__file__).resolve().parent
CALIBRATION = json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))
SEED_PAIRS = [
    (pair["start"], pair["end"])
    for pair in json.loads((HERE / "seed_words.json").read_text(encoding="utf-8"))[
        "endpoint_pairs"
    ]
]
MARGIN = 0.015


class CalibrationTest(unittest.TestCase):
    def test_must_pass_pairs_verify(self):
        for left, right in CALIBRATION["must_pass"]:
            checked = link(left, right)
            self.assertTrue(checked.verified, f"{left}<->{right}: {checked}")

    def test_collocation_pairs_verify_despite_low_cosine(self):
        for left, right in CALIBRATION["must_pass_collocations"]:
            checked = link(left, right)
            self.assertTrue(checked.verified, f"{left}<->{right}: {checked}")

    def test_must_fail_pairs_are_rejected(self):
        for left, right in CALIBRATION["must_fail"]:
            checked = link(left, right)
            self.assertFalse(checked.verified, f"{left}<->{right}: {checked}")
            self.assertEqual(checked.reason, "below-threshold")

    def test_threshold_has_margin_on_both_sides(self):
        lowest_pass = min(
            link(left, right).cosine for left, right in CALIBRATION["must_pass"]
        )
        highest_fail = max(
            link(left, right).cosine for left, right in CALIBRATION["must_fail"]
        )
        self.assertGreaterEqual(lowest_pass, DEFAULT_THRESHOLD + MARGIN)
        self.assertLessEqual(highest_fail, DEFAULT_THRESHOLD - MARGIN)

    def test_pop_culture_link_star_wars(self):
        checked = link("star", "wars")
        self.assertTrue(checked.verified)
        self.assertIsNone(checked.reason)


class VerifySequenceTest(unittest.TestCase):
    def test_accepts_verified_sequence_and_signs_off_every_link(self):
        route = shortest_route("rain", "business")
        self.assertIsNotNone(route)
        result = verify_sequence("rain", "business", route.words)
        self.assertTrue(result.verified)
        self.assertEqual(result.length, len(route.words))
        for checked in result.links:
            self.assertTrue(checked.verified)
            self.assertIsNone(checked.reason)
            self.assertGreaterEqual(checked.cosine, DEFAULT_THRESHOLD)

    def test_rejects_below_threshold_link_with_reason_on_that_link(self):
        result = verify_sequence("carrot", "algebra", [])
        self.assertFalse(result.verified)
        self.assertEqual(result.links[0].reason, "below-threshold")

    def test_rejects_connector_outside_the_pool(self):
        result = verify_sequence("rain", "harvest", ["zzzzqqq"])
        self.assertFalse(result.verified)
        self.assertEqual(result.links[0].reason, "not-in-pool")

    def test_rejects_unknown_endpoint(self):
        result = verify_sequence("xqzzvw", "harvest", ["crop"])
        self.assertFalse(result.verified)
        self.assertEqual(result.links[0].reason, "not-in-vocabulary")

    def test_rejects_inflection_chain(self):
        result = verify_sequence("rain", "harvest", ["rains", "crop"])
        self.assertFalse(result.verified)
        self.assertEqual(result.links[0].reason, "inflection-variant")

    def test_rejects_repeated_word(self):
        result = verify_sequence("rain", "harvest", ["rain"])
        self.assertFalse(result.verified)
        self.assertEqual(result.links[0].reason, "repeated-word")


class ShortestRouteTest(unittest.TestCase):
    def test_deterministic_across_calls(self):
        first = shortest_route("idea", "business")
        second = shortest_route("idea", "business")
        self.assertEqual(first, second)

    def test_seed_pairs_have_nontrivial_benchmark_routes(self):
        for start, end in SEED_PAIRS:
            route = shortest_route(start, end)
            self.assertIsNotNone(route, f"{start}->{end} has no route")
            self.assertGreaterEqual(
                route.length, 1, f"{start}->{end} solved with no intermediates"
            )
            pool = get_pool()
            for word in route.words:
                self.assertIn(word, pool.indices)
            for checked in route.links:
                self.assertTrue(checked.verified)

    def test_returns_none_when_no_route_within_hop_limit(self):
        self.assertIsNone(shortest_route("carrot", "algebra", max_hops=0))


class SolutionTreeTest(unittest.TestCase):
    def test_tree_routes_reverify_and_edges_are_signed_off(self):
        tree = build_solution_tree("stone", "city", max_routes=5)
        self.assertIsNotNone(tree)
        self.assertGreaterEqual(len(tree.routes), 1)
        shortest = min(route.length for route in tree.routes)
        for route in tree.routes:
            self.assertEqual(route.length, shortest)
            result = verify_sequence(route.start, route.end, route.words)
            self.assertTrue(result.verified, f"route {route.words} failed: {result}")

        def walk(node):
            for branch in node.branches:
                self.assertTrue(branch.link.verified)
                self.assertIsNone(branch.link.reason)
                self.assertEqual(branch.link.left, node.word)
                self.assertEqual(branch.link.right, branch.node.word)
                walk(branch.node)

        self.assertEqual(tree.root.word, "stone")
        walk(tree.root)

    def test_tree_contains_every_route_as_a_path(self):
        tree = build_solution_tree("winter", "guitar", max_routes=5)
        self.assertIsNotNone(tree)
        for route in tree.routes:
            node = tree.root
            for word in (*route.words, route.end):
                match = [branch for branch in node.branches if branch.node.word == word]
                self.assertEqual(len(match), 1)
                node = match[0].node
            self.assertEqual(node.branches, [])


class PuzzleValidityTest(unittest.TestCase):
    def test_rejects_directly_connected_pair(self):
        self.assertTrue(link("winter", "snow").verified)
        self.assertFalse(is_valid_puzzle("winter", "snow"))

    def test_rejects_inflection_pair_and_unknown_words(self):
        self.assertFalse(is_valid_puzzle("rain", "raining"))
        self.assertFalse(is_valid_puzzle("xqzzvw", "harvest"))

    def test_accepts_seed_pairs(self):
        for start, end in SEED_PAIRS:
            self.assertTrue(is_valid_puzzle(start, end), f"{start}->{end} invalid")


if __name__ == "__main__":
    unittest.main()
