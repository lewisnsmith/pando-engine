"""Tests for the local Pando inspector without loading the word-vector model."""

from __future__ import annotations

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app


def fake_link(
    left: str = "seed",
    right: str = "plant",
    cosine: float | None = 0.42,
    verified: bool = True,
    reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        left=left,
        right=right,
        cosine=cosine,
        verified=verified,
        reason=reason,
    )


def fake_verification(
    *, verified: bool = True, reason: str | None = None
) -> SimpleNamespace:
    links = (
        fake_link(verified=verified, reason=reason),
        fake_link("plant", "forest", 0.31234567, verified, reason),
    )
    return SimpleNamespace(
        start="seed",
        end="forest",
        words=("plant",),
        links=links,
        verified=verified,
        length=1,
    )


class RequestServiceTest(unittest.TestCase):
    def test_verify_serializes_a_valid_chain(self):
        engine = Mock()
        engine.verify_sequence.return_value = fake_verification()

        response = app.verify_request(
            {"start": " seed ", "end": " forest ", "words": [" plant "]},
            engine,
        )

        engine.verify_sequence.assert_called_once_with("seed", "forest", ["plant"])
        self.assertTrue(response["verified"])
        self.assertEqual(response["length"], 1)
        self.assertEqual(response["links"][1]["cosine"], 0.312346)

    def test_verify_preserves_every_engine_rejection_reason(self):
        reasons = (
            "below-threshold",
            "not-in-vocabulary",
            "not-in-pool",
            "repeated-word",
            "inflection-variant",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                engine = Mock()
                engine.verify_sequence.return_value = fake_verification(
                    verified=False, reason=reason
                )
                response = app.verify_request(
                    {"start": "seed", "end": "forest", "words": ["plant"]},
                    engine,
                )
                self.assertFalse(response["verified"])
                self.assertEqual(response["links"][0]["reason"], reason)

    def test_verify_rejects_blank_connector_before_calling_engine(self):
        engine = Mock()
        with self.assertRaisesRegex(app.RequestError, "must not be blank"):
            app.verify_request(
                {"start": "seed", "end": "forest", "words": [" "]}, engine
            )
        engine.verify_sequence.assert_not_called()

    def test_benchmark_serializes_a_found_route(self):
        engine = Mock()
        engine.shortest_route.return_value = SimpleNamespace(
            start="seed",
            end="forest",
            words=("plant",),
            links=(fake_link(), fake_link("plant", "forest", 0.31234567)),
            length=1,
            total_cosine=0.73234567,
        )

        response = app.benchmark_request(
            {"start": "seed", "end": "forest", "maxHops": 8}, engine
        )

        engine.shortest_route.assert_called_once_with("seed", "forest", max_hops=8)
        self.assertTrue(response["found"])
        self.assertEqual(response["route"]["words"], ["plant"])
        self.assertEqual(response["route"]["totalCosine"], 0.732346)

    def test_benchmark_returns_explicit_no_route_result(self):
        engine = Mock()
        engine.shortest_route.return_value = None
        response = app.benchmark_request(
            {"start": "carrot", "end": "algebra", "maxHops": 0}, engine
        )
        self.assertEqual(
            response,
            {
                "start": "carrot",
                "end": "algebra",
                "maxHops": 0,
                "found": False,
                "route": None,
            },
        )

    def test_benchmark_rejects_hop_limit_outside_engine_contract(self):
        with self.assertRaisesRegex(app.RequestError, "from 0 through 8"):
            app.benchmark_request(
                {"start": "seed", "end": "forest", "maxHops": 9}, Mock()
            )

    def test_engine_runtime_errors_are_not_converted_to_fake_results(self):
        engine = Mock()
        engine.verify_sequence.side_effect = RuntimeError("model missing")
        with self.assertRaisesRegex(RuntimeError, "model missing"):
            app.verify_request(
                {"start": "seed", "end": "forest", "words": ["plant"]},
                engine,
            )


class HttpEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = app.make_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def get_json(self, path: str):
        with urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return response.status, json.load(response)

    def post_json(self, path: str, payload):
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.load(response)

    def test_puzzles_endpoint_returns_seeds_and_engine_metadata(self):
        status, response = self.get_json("/api/puzzles")
        self.assertEqual(status, 200)
        self.assertEqual(len(response["puzzles"]), 6)
        self.assertEqual(response["engine"]["threshold"], 0.19)
        self.assertEqual(response["engine"]["maxHops"], 8)

    def test_verify_endpoint_returns_serialized_engine_result(self):
        engine = Mock()
        engine.verify_sequence.return_value = fake_verification()
        with patch.object(app, "get_engine", return_value=engine):
            status, response = self.post_json(
                "/api/verify",
                {"start": "seed", "end": "forest", "words": ["plant"]},
            )
        self.assertEqual(status, 200)
        self.assertTrue(response["verified"])
        self.assertEqual(response["links"][0]["left"], "seed")

    def test_invalid_request_returns_structured_400(self):
        try:
            self.post_json(
                "/api/verify", {"start": "seed", "end": "forest", "words": [""]}
            )
        except HTTPError as error:
            response = json.load(error)
            self.assertEqual(error.code, 400)
            self.assertEqual(response["error"]["code"], "invalid-request")
        else:
            self.fail("Expected an HTTP 400 response")

    def test_missing_model_returns_structured_503(self):
        with patch.object(
            app, "get_engine", side_effect=RuntimeError("Download the model first.")
        ):
            try:
                self.post_json(
                    "/api/verify",
                    {"start": "seed", "end": "forest", "words": ["plant"]},
                )
            except HTTPError as error:
                response = json.load(error)
                self.assertEqual(error.code, 503)
                self.assertEqual(response["error"]["code"], "engine-unavailable")
                self.assertIn("Download the model", response["error"]["message"])
            else:
                self.fail("Expected an HTTP 503 response")


if __name__ == "__main__":
    unittest.main()
