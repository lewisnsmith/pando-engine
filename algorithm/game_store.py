"""SQLite persistence for Pando players, sequences, and leaderboards.

The connectivity engine decides whether a sequence is valid.  This module
owns the game state around that decision: drafts, submission history,
personal bests, and per-puzzle global leaderboards.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence


class Verification(Protocol):
    verified: bool


Verifier = Callable[[str, str, Sequence[str]], Verification]


class PlayerNotFoundError(LookupError):
    """Raised when game state is requested for an unknown player."""


class InvalidSequenceError(ValueError):
    """Raised when a player tries to save an unverified sequence."""

    def __init__(self, verification: Verification):
        super().__init__("Only verified sequences can be saved.")
        self.verification = verification


@dataclass(frozen=True)
class SavedSequence:
    id: int
    player_id: str
    puzzle_id: int
    start: str
    end: str
    words: tuple[str, ...]
    word_count: int
    submitted_at: str


@dataclass(frozen=True)
class SaveResult:
    sequence: SavedSequence
    is_personal_best: bool


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    player_id: str
    display_name: str
    word_count: int
    words: tuple[str, ...]
    achieved_at: str


class GameStore:
    """Transactional game-state store backed by SQLite.

    ``player_id`` is an opaque identifier supplied by the eventual account
    system. Scores count connector words only; puzzle endpoints are excluded
    because they are fixed for every player.
    """

    def __init__(
        self,
        database: str | Path = ":memory:",
        verifier: Optional[Verifier] = None,
        initialize_schema: bool = True,
    ) -> None:
        database_name = str(database)
        if database_name != ":memory:":
            Path(database_name).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._connection = sqlite3.connect(database_name, timeout=5)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        if database_name != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
        self._verifier = verifier if verifier is not None else _verify_sequence
        if initialize_schema:
            self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "GameStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS players (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS puzzles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                start_key TEXT NOT NULL,
                end_key TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (start_key, end_key)
            );

            CREATE TABLE IF NOT EXISTS saved_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                puzzle_id INTEGER NOT NULL REFERENCES puzzles(id) ON DELETE CASCADE,
                words_json TEXT NOT NULL,
                word_count INTEGER NOT NULL CHECK (word_count >= 0),
                submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS personal_bests (
                player_id TEXT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                puzzle_id INTEGER NOT NULL REFERENCES puzzles(id) ON DELETE CASCADE,
                sequence_id INTEGER NOT NULL REFERENCES saved_sequences(id) ON DELETE CASCADE,
                word_count INTEGER NOT NULL CHECK (word_count >= 0),
                achieved_at TEXT NOT NULL,
                PRIMARY KEY (player_id, puzzle_id)
            );

            CREATE TABLE IF NOT EXISTS active_sequences (
                player_id TEXT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                puzzle_id INTEGER NOT NULL REFERENCES puzzles(id) ON DELETE CASCADE,
                words_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, puzzle_id)
            );

            CREATE INDEX IF NOT EXISTS saved_sequences_player_puzzle
                ON saved_sequences(player_id, puzzle_id, submitted_at DESC);
            CREATE INDEX IF NOT EXISTS personal_bests_leaderboard
                ON personal_bests(puzzle_id, word_count, achieved_at);
            """
        )

    def save_player(self, player_id: str, display_name: str) -> None:
        player_id = _required_text(player_id, "player_id")
        display_name = _required_text(display_name, "display_name")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO players (id, display_name) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET display_name = excluded.display_name
                """,
                (player_id, display_name),
            )

    def save_sequence(
        self,
        player_id: str,
        start: str,
        end: str,
        words: Sequence[str],
    ) -> SaveResult:
        """Verify and save one submission, retaining all prior submissions."""
        player_id = _required_text(player_id, "player_id")
        start = _required_text(start, "start")
        end = _required_text(end, "end")
        connectors = _clean_words(words)
        self._require_player(player_id)

        verification = self._verifier(start, end, connectors)
        if not verification.verified:
            raise InvalidSequenceError(verification)

        words_json = json.dumps(connectors, ensure_ascii=False)
        word_count = len(connectors)
        with self._connection:
            puzzle_id = self._get_or_create_puzzle(start, end)
            cursor = self._connection.execute(
                """
                INSERT INTO saved_sequences
                    (player_id, puzzle_id, words_json, word_count)
                VALUES (?, ?, ?, ?)
                """,
                (player_id, puzzle_id, words_json, word_count),
            )
            sequence_id = int(cursor.lastrowid)
            row = self._connection.execute(
                """
                SELECT submitted_at FROM saved_sequences WHERE id = ?
                """,
                (sequence_id,),
            ).fetchone()
            submitted_at = str(row["submitted_at"])

            best_cursor = self._connection.execute(
                """
                INSERT INTO personal_bests
                    (player_id, puzzle_id, sequence_id, word_count, achieved_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id, puzzle_id) DO UPDATE SET
                    sequence_id = excluded.sequence_id,
                    word_count = excluded.word_count,
                    achieved_at = excluded.achieved_at
                WHERE excluded.word_count < personal_bests.word_count
                """,
                (player_id, puzzle_id, sequence_id, word_count, submitted_at),
            )

        saved = SavedSequence(
            id=sequence_id,
            player_id=player_id,
            puzzle_id=puzzle_id,
            start=start,
            end=end,
            words=connectors,
            word_count=word_count,
            submitted_at=submitted_at,
        )
        return SaveResult(saved, best_cursor.rowcount == 1)

    def list_player_sequences(
        self,
        player_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> tuple[SavedSequence, ...]:
        """Return every saved submission, newest first, optionally by puzzle."""
        player_id = _required_text(player_id, "player_id")
        self._require_player(player_id)
        parameters: list[object] = [player_id]
        puzzle_filter = ""
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("start and end must be provided together")
            puzzle_filter = " AND p.start_key = ? AND p.end_key = ?"
            parameters.extend((_word_key(start), _word_key(end)))

        rows = self._connection.execute(
            f"""
            SELECT s.id, s.player_id, s.puzzle_id, p.start, p.end,
                   s.words_json, s.word_count, s.submitted_at
            FROM saved_sequences AS s
            JOIN puzzles AS p ON p.id = s.puzzle_id
            WHERE s.player_id = ?{puzzle_filter}
            ORDER BY s.id DESC
            """,
            parameters,
        ).fetchall()
        return tuple(_saved_sequence(row) for row in rows)

    def personal_best(
        self, player_id: str, start: str, end: str
    ) -> Optional[SavedSequence]:
        player_id = _required_text(player_id, "player_id")
        self._require_player(player_id)
        row = self._connection.execute(
            """
            SELECT s.id, s.player_id, s.puzzle_id, p.start, p.end,
                   s.words_json, s.word_count, s.submitted_at
            FROM personal_bests AS b
            JOIN saved_sequences AS s ON s.id = b.sequence_id
            JOIN puzzles AS p ON p.id = b.puzzle_id
            WHERE b.player_id = ? AND p.start_key = ? AND p.end_key = ?
            """,
            (player_id, _word_key(start), _word_key(end)),
        ).fetchone()
        return None if row is None else _saved_sequence(row)

    def leaderboard(
        self, start: str, end: str, limit: int = 100
    ) -> tuple[LeaderboardEntry, ...]:
        """Return each player's best sequence for a puzzle, shortest first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        rows = self._connection.execute(
            """
            SELECT b.player_id, pl.display_name, b.word_count, s.words_json,
                   b.achieved_at
            FROM personal_bests AS b
            JOIN players AS pl ON pl.id = b.player_id
            JOIN puzzles AS p ON p.id = b.puzzle_id
            JOIN saved_sequences AS s ON s.id = b.sequence_id
            WHERE p.start_key = ? AND p.end_key = ?
            ORDER BY b.word_count ASC, b.achieved_at ASC,
                     pl.display_name COLLATE NOCASE ASC, b.player_id ASC
            LIMIT ?
            """,
            (_word_key(start), _word_key(end), limit),
        ).fetchall()

        entries = []
        previous_score: Optional[int] = None
        rank = 0
        for position, row in enumerate(rows, start=1):
            score = int(row["word_count"])
            if score != previous_score:
                rank = position
                previous_score = score
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    player_id=str(row["player_id"]),
                    display_name=str(row["display_name"]),
                    word_count=score,
                    words=tuple(json.loads(row["words_json"])),
                    achieved_at=str(row["achieved_at"]),
                )
            )
        return tuple(entries)

    def set_current_sequence(
        self, player_id: str, start: str, end: str, words: Sequence[str]
    ) -> tuple[str, ...]:
        """Create or replace a player's in-progress sequence for a puzzle."""
        player_id = _required_text(player_id, "player_id")
        start = _required_text(start, "start")
        end = _required_text(end, "end")
        connectors = _clean_words(words)
        self._require_player(player_id)
        with self._connection:
            puzzle_id = self._get_or_create_puzzle(start, end)
            self._connection.execute(
                """
                INSERT INTO active_sequences (player_id, puzzle_id, words_json)
                VALUES (?, ?, ?)
                ON CONFLICT(player_id, puzzle_id) DO UPDATE SET
                    words_json = excluded.words_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (player_id, puzzle_id, json.dumps(connectors, ensure_ascii=False)),
            )
        return connectors

    def current_sequence(
        self, player_id: str, start: str, end: str
    ) -> tuple[str, ...]:
        player_id = _required_text(player_id, "player_id")
        self._require_player(player_id)
        row = self._connection.execute(
            """
            SELECT a.words_json
            FROM active_sequences AS a
            JOIN puzzles AS p ON p.id = a.puzzle_id
            WHERE a.player_id = ? AND p.start_key = ? AND p.end_key = ?
            """,
            (player_id, _word_key(start), _word_key(end)),
        ).fetchone()
        return () if row is None else tuple(json.loads(row["words_json"]))

    def clear_current_sequence(self, player_id: str, start: str, end: str) -> bool:
        """Clear only the draft; saved submissions and scores are untouched."""
        player_id = _required_text(player_id, "player_id")
        self._require_player(player_id)
        with self._connection:
            cursor = self._connection.execute(
                """
                DELETE FROM active_sequences
                WHERE player_id = ? AND puzzle_id = (
                    SELECT id FROM puzzles WHERE start_key = ? AND end_key = ?
                )
                """,
                (player_id, _word_key(start), _word_key(end)),
            )
        return cursor.rowcount == 1

    def _require_player(self, player_id: str) -> None:
        row = self._connection.execute(
            "SELECT 1 FROM players WHERE id = ?", (player_id,)
        ).fetchone()
        if row is None:
            raise PlayerNotFoundError(player_id)

    def _get_or_create_puzzle(self, start: str, end: str) -> int:
        self._connection.execute(
            """
            INSERT INTO puzzles (start, end, start_key, end_key)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(start_key, end_key) DO NOTHING
            """,
            (start, end, _word_key(start), _word_key(end)),
        )
        row = self._connection.execute(
            "SELECT id FROM puzzles WHERE start_key = ? AND end_key = ?",
            (_word_key(start), _word_key(end)),
        ).fetchone()
        return int(row["id"])


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _verify_sequence(start: str, end: str, words: Sequence[str]) -> Verification:
    """Import the model-backed engine only when a submission needs checking."""
    try:
        from .connectivity import verify_sequence
    except ImportError:  # Support importing with ``algorithm/`` on sys.path.
        from connectivity import verify_sequence
    return verify_sequence(start, end, words)


def _word_key(word: str) -> str:
    return _required_text(word, "word").casefold()


def _clean_words(words: Sequence[str]) -> tuple[str, ...]:
    if isinstance(words, (str, bytes)):
        raise ValueError("words must be a sequence of strings")
    cleaned = []
    for word in words:
        cleaned.append(_required_text(word, "connector word"))
    return tuple(cleaned)


def _saved_sequence(row: sqlite3.Row) -> SavedSequence:
    return SavedSequence(
        id=int(row["id"]),
        player_id=str(row["player_id"]),
        puzzle_id=int(row["puzzle_id"]),
        start=str(row["start"]),
        end=str(row["end"]),
        words=tuple(json.loads(row["words_json"])),
        word_count=int(row["word_count"]),
        submitted_at=str(row["submitted_at"]),
    )
