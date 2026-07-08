import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sequence_word2vec import (
    generate_browser_data,
    generate_puzzles,
    load_seed_corpus,
    train_model,
    write_browser_data,
)


FIXTURE = {
    "domains": [
        {
            "name": "nature",
            "sentences": [
                ["seed", "sprout", "plant", "tree", "forest"],
                ["rain", "water", "soil", "growth", "harvest"],
                ["river", "stream", "water", "valley", "forest"],
            ],
        },
        {
            "name": "commerce",
            "sentences": [
                ["idea", "plan", "product", "market", "business"],
                ["concept", "prototype", "customer", "sale", "business"],
                ["pitch", "team", "service", "revenue", "company"],
            ],
        },
        {
            "name": "knowledge",
            "sentences": [
                ["paper", "page", "book", "shelf", "library"],
                ["note", "draft", "manuscript", "archive", "record"],
                ["question", "search", "catalog", "index", "answer"],
            ],
        },
        {
            "name": "city",
            "sentences": [
                ["stone", "brick", "wall", "street", "city"],
                ["quarry", "cement", "building", "block", "neighborhood"],
                ["road", "bridge", "district", "square", "city"],
            ],
        },
    ],
    "endpoint_pairs": [
        {"start": "seed", "end": "forest", "prompt": "Connect a small beginning to a living system."},
        {"start": "idea", "end": "business", "prompt": "Connect a thought to an operating venture."},
        {"start": "paper", "end": "library", "prompt": "Connect a material to organized knowledge."},
    ],
}


class SequenceWord2VecTest(unittest.TestCase):
    def test_generates_deterministic_puzzles_with_ranked_routes(self):
        corpus = load_seed_corpus(FIXTURE)
        model = train_model(corpus)

        first = generate_puzzles(corpus, model, count_per_tier=1, routes_per_puzzle=2)
        second = generate_puzzles(corpus, model, count_per_tier=1, routes_per_puzzle=2)

        self.assertEqual(first, second)
        self.assertEqual({puzzle["tier"] for puzzle in first}, {1, 2, 3})

        for puzzle in first:
            self.assertNotEqual(puzzle["start"], puzzle["end"])
            self.assertEqual(len(puzzle["routes"]), 2)
            self.assertEqual(len(puzzle["clues"]), puzzle["tier"])
            for route in puzzle["routes"]:
                self.assertEqual(len(route), puzzle["tier"])
                self.assertNotIn(puzzle["start"], route)
                self.assertNotIn(puzzle["end"], route)
            for clue in puzzle["clues"]:
                self.assertIn("connectability", clue["category"])
                self.assertGreaterEqual(len(clue["related"]), 3)

    def test_browser_data_contains_vectors_without_training_in_browser(self):
        corpus = load_seed_corpus(FIXTURE)
        model = train_model(corpus)
        data = generate_browser_data(corpus, model)

        self.assertIn("puzzles", data)
        self.assertIn("scoring", data)
        self.assertIn("vectors", data["scoring"])
        self.assertIn("vocabulary", data["scoring"])
        self.assertIn("seed", data["scoring"]["vectors"])
        self.assertEqual(len(data["scoring"]["vectors"]["seed"]), 24)
        self.assertEqual([trial["id"] for trial in data["pocTrials"]], ["multi-route", "single-route"])
        self.assertEqual(len(data["pocTrials"][0]["routes"]), 3)
        self.assertEqual(len(data["pocTrials"][1]["routes"]), 1)

    def test_writes_static_browser_data_script(self):
        corpus = load_seed_corpus(FIXTURE)
        model = train_model(corpus)
        data = generate_browser_data(corpus, model)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "sequence-data.js"
            write_browser_data(data, output)
            text = output.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("window.SEQUENCE_DATA = "))
        payload = text.removeprefix("window.SEQUENCE_DATA = ").removesuffix(";\n")
        self.assertEqual(json.loads(payload), data)


if __name__ == "__main__":
    unittest.main()
