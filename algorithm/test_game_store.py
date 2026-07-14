"""Tests for game persistence that do not load the word-vector model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_store import GameStore, InvalidSequenceError, PlayerNotFoundError


def accept_sequence(start, end, words):
    return SimpleNamespace(
        start=start,
        end=end,
        words=tuple(words),
        verified=True,
        length=len(words),
    )


class GameStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = GameStore(verifier=accept_sequence)
        self.store.save_player("player-1", "Ada")

    def tearDown(self):
        self.store.close()

    def test_keeps_multiple_sequences_and_the_lowest_personal_score(self):
        first = self.store.save_sequence(
            "player-1", "rain", "business", ["water", "work", "office"]
        )
        second = self.store.save_sequence(
            "player-1", "rain", "business", ["weather", "commerce"]
        )
        third = self.store.save_sequence(
            "player-1", "rain", "business", ["storm", "trade", "company"]
        )

        self.assertTrue(first.is_personal_best)
        self.assertTrue(second.is_personal_best)
        self.assertFalse(third.is_personal_best)
        self.assertEqual(len(self.store.list_player_sequences("player-1")), 3)
        best = self.store.personal_best("player-1", "rain", "business")
        self.assertEqual(best.words, ("weather", "commerce"))
        self.assertEqual(best.word_count, 2)

    def test_personal_bests_are_independent_for_each_puzzle(self):
        self.store.save_sequence("player-1", "rain", "business", ["trade"])
        self.store.save_sequence("player-1", "winter", "guitar", ["cold", "music"])

        rain_best = self.store.personal_best("player-1", "rain", "business")
        winter_best = self.store.personal_best("player-1", "winter", "guitar")
        self.assertEqual(rain_best.word_count, 1)
        self.assertEqual(winter_best.word_count, 2)

    def test_global_leaderboard_contains_each_players_best_only(self):
        self.store.save_player("player-2", "Grace")
        self.store.save_player("player-3", "Linus")
        self.store.save_sequence("player-1", "rain", "business", ["a", "b", "c"])
        self.store.save_sequence("player-1", "rain", "business", ["a", "b"])
        self.store.save_sequence("player-2", "rain", "business", ["a"])
        self.store.save_sequence("player-3", "rain", "business", ["x", "y"])

        leaderboard = self.store.leaderboard("rain", "business")

        self.assertEqual(
            [entry.player_id for entry in leaderboard],
            ["player-2", "player-1", "player-3"],
        )
        self.assertEqual([entry.word_count for entry in leaderboard], [1, 2, 2])
        self.assertEqual([entry.rank for entry in leaderboard], [1, 2, 2])

    def test_current_sequence_can_be_replaced_and_cleared_without_losing_saves(self):
        self.store.save_sequence("player-1", "rain", "business", ["saved"])
        self.store.set_current_sequence(
            "player-1", "rain", "business", ["draft", "words"]
        )
        self.assertEqual(
            self.store.current_sequence("player-1", "rain", "business"),
            ("draft", "words"),
        )

        self.assertTrue(
            self.store.clear_current_sequence("player-1", "rain", "business")
        )
        self.assertEqual(
            self.store.current_sequence("player-1", "rain", "business"), ()
        )
        self.assertEqual(len(self.store.list_player_sequences("player-1")), 1)
        self.assertFalse(
            self.store.clear_current_sequence("player-1", "rain", "business")
        )

    def test_invalid_sequence_is_not_saved(self):
        rejection = SimpleNamespace(verified=False)
        store = GameStore(verifier=lambda *_args: rejection)
        self.addCleanup(store.close)
        store.save_player("player-1", "Ada")

        with self.assertRaises(InvalidSequenceError) as error:
            store.save_sequence("player-1", "rain", "business", ["nope"])

        self.assertIs(error.exception.verification, rejection)
        self.assertEqual(store.list_player_sequences("player-1"), ())

    def test_unknown_players_cannot_own_game_state(self):
        with self.assertRaises(PlayerNotFoundError):
            self.store.set_current_sequence("missing", "rain", "business", ["word"])

    def test_player_display_name_can_be_updated(self):
        self.store.save_player("player-1", "Ada Lovelace")
        self.store.save_sequence("player-1", "rain", "business", ["logic"])

        entry = self.store.leaderboard("rain", "business")[0]
        self.assertEqual(entry.display_name, "Ada Lovelace")

    def test_file_database_survives_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "pando.sqlite3"
            with GameStore(database, verifier=accept_sequence) as first:
                first.save_player("persistent-player", "Katherine")
                first.save_sequence(
                    "persistent-player", "rain", "business", ["weather"]
                )

            with GameStore(database, verifier=accept_sequence) as reopened:
                best = reopened.personal_best(
                    "persistent-player", "rain", "business"
                )
                self.assertEqual(best.words, ("weather",))
                self.assertEqual(
                    reopened.leaderboard("rain", "business")[0].display_name,
                    "Katherine",
                )


if __name__ == "__main__":
    unittest.main()
