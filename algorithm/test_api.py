"""HTTP API tests that avoid loading the word-vector model."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

try:
    from .api import Settings, create_app
except ImportError:  # Support unittest discovery from ``algorithm/``.
    from api import Settings, create_app


@dataclass(frozen=True)
class FakeLink:
    left: str
    right: str
    cosine: float
    verified: bool
    reason: str | None


@dataclass(frozen=True)
class FakeVerification:
    start: str
    end: str
    words: tuple[str, ...]
    links: tuple[FakeLink, ...]
    verified: bool
    length: int


def verify_for_api(start, end, words):
    connectors = tuple(words)
    chain = (start, *connectors, end)
    rejected = "invalid" in connectors
    links = tuple(
        FakeLink(
            left,
            right,
            0.1 if rejected else 0.9,
            not rejected,
            "below-threshold" if rejected else None,
        )
        for left, right in zip(chain, chain[1:])
    )
    return FakeVerification(
        start,
        end,
        connectors,
        links,
        not rejected,
        len(connectors),
    )


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        database = Path(self.temporary_directory.name) / "pando.sqlite3"
        settings = Settings(database_path=str(database))
        self.client = TestClient(create_app(settings, verifier=verify_for_api))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary_directory.cleanup()

    @staticmethod
    def headers(player_id="player-1"):
        return {"X-Player-ID": player_id}

    def register(self, player_id="player-1", display_name="Ada"):
        response = self.client.put(
            "/api/me",
            headers=self.headers(player_id),
            json={"display_name": display_name},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_health_and_public_leaderboard_do_not_require_identity(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

        response = self.client.get("/api/puzzles/rain/business/leaderboard")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"entries": []})

    def test_player_routes_require_authentication_and_registration(self):
        missing_identity = self.client.get("/api/puzzles/rain/business/draft")
        self.assertEqual(missing_identity.status_code, 401)

        unknown_player = self.client.get(
            "/api/puzzles/rain/business/draft", headers=self.headers("missing")
        )
        self.assertEqual(unknown_player.status_code, 404)
        self.assertEqual(unknown_player.json()["error"], "player_not_found")

    def test_draft_can_be_saved_replaced_and_cleared(self):
        self.register()
        route = "/api/puzzles/rain/business/draft"

        first = self.client.put(
            route, headers=self.headers(), json={"words": ["water", "office"]}
        )
        second = self.client.put(
            route, headers=self.headers(), json={"words": ["storm"]}
        )
        loaded = self.client.get(route, headers=self.headers())
        cleared = self.client.delete(route, headers=self.headers())
        empty = self.client.get(route, headers=self.headers())

        self.assertEqual(first.json()["words"], ["water", "office"])
        self.assertEqual(second.json()["words"], ["storm"])
        self.assertEqual(loaded.json()["words"], ["storm"])
        self.assertEqual(cleared.json(), {"cleared": True})
        self.assertEqual(empty.json(), {"words": []})

    def test_submissions_history_and_personal_best_share_one_contract(self):
        self.register()
        route = "/api/puzzles/rain/business/submissions"

        first = self.client.post(
            route,
            headers=self.headers(),
            json={"words": ["water", "work", "office"]},
        )
        second = self.client.post(
            route,
            headers=self.headers(),
            json={"words": ["weather"]},
        )
        history = self.client.get(route, headers=self.headers())
        best = self.client.get(
            "/api/puzzles/rain/business/personal-best", headers=self.headers()
        )

        self.assertTrue(first.json()["is_personal_best"])
        self.assertTrue(second.json()["is_personal_best"])
        self.assertEqual(len(history.json()["sequences"]), 2)
        self.assertEqual(best.json()["sequence"]["words"], ["weather"])
        self.assertEqual(best.json()["sequence"]["word_count"], 1)

    def test_leaderboard_is_global_and_uses_each_players_best(self):
        self.register("player-1", "Ada")
        self.register("player-2", "Grace")
        route = "/api/puzzles/rain/business/submissions"
        self.client.post(
            route,
            headers=self.headers("player-1"),
            json={"words": ["water", "work"]},
        )
        self.client.post(
            route,
            headers=self.headers("player-1"),
            json={"words": ["water"]},
        )
        self.client.post(
            route,
            headers=self.headers("player-2"),
            json={"words": ["storm", "trade"]},
        )

        response = self.client.get("/api/puzzles/rain/business/leaderboard")
        entries = response.json()["entries"]

        self.assertEqual([entry["display_name"] for entry in entries], ["Ada", "Grace"])
        self.assertEqual([entry["word_count"] for entry in entries], [1, 2])
        self.assertEqual([entry["rank"] for entry in entries], [1, 2])

    def test_invalid_submission_returns_link_level_verification(self):
        self.register()

        response = self.client.post(
            "/api/puzzles/rain/business/submissions",
            headers=self.headers(),
            json={"words": ["invalid"]},
        )

        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"], "invalid_sequence")
        self.assertFalse(body["verification"]["verified"])
        self.assertEqual(
            body["verification"]["links"][0]["reason"], "below-threshold"
        )

    def test_request_validation_limits_words_and_leaderboard_size(self):
        self.register()
        too_many_words = self.client.put(
            "/api/puzzles/rain/business/draft",
            headers=self.headers(),
            json={"words": ["word"] * 101},
        )
        invalid_limit = self.client.get(
            "/api/puzzles/rain/business/leaderboard?limit=0"
        )

        self.assertEqual(too_many_words.status_code, 422)
        self.assertEqual(invalid_limit.status_code, 422)


if __name__ == "__main__":
    unittest.main()
