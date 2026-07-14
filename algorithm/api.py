"""HTTP API for the Pando verification engine and game-state store."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Iterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from .game_store import (
        GameStore,
        InvalidSequenceError,
        PlayerNotFoundError,
        Verifier,
    )
except ImportError:  # Support running with ``algorithm/`` on sys.path.
    from game_store import (  # type: ignore[no-redef]
        GameStore,
        InvalidSequenceError,
        PlayerNotFoundError,
        Verifier,
    )


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    database_path: str = "data/pando.sqlite3"
    auth_mode: str = "development"
    player_id_header: str = "X-Player-ID"
    cors_origins: tuple[str, ...] = ()
    warm_model: bool = False

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv("PANDO_CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            database_path=os.getenv("PANDO_DATABASE_PATH", "data/pando.sqlite3"),
            auth_mode=os.getenv("PANDO_AUTH_MODE", "development"),
            player_id_header=os.getenv("PANDO_PLAYER_ID_HEADER", "X-Player-ID"),
            cors_origins=origins,
            warm_model=_environment_flag("PANDO_WARM_MODEL"),
        )

    def validate(self) -> None:
        if self.auth_mode not in {"development", "trusted-header"}:
            raise ValueError(
                "PANDO_AUTH_MODE must be 'development' or 'trusted-header'"
            )
        if self.database_path == ":memory:":
            raise ValueError(
                "PANDO_DATABASE_PATH must be a file because API requests use "
                "independent database connections"
            )
        if not self.player_id_header.strip():
            raise ValueError("PANDO_PLAYER_ID_HEADER cannot be empty")


@dataclass(frozen=True)
class Identity:
    player_id: str


class PlayerPayload(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)


class WordsPayload(BaseModel):
    words: list[str] = Field(default_factory=list, max_length=100)


def create_app(
    settings: Optional[Settings] = None,
    verifier: Optional[Verifier] = None,
) -> FastAPI:
    """Create an API instance; injectable settings/verifier keep tests model-free."""
    settings = settings or Settings.from_environment()
    settings.validate()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        Path(settings.database_path).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        with GameStore(settings.database_path, verifier=verifier):
            pass
        if settings.warm_model:
            try:
                from .connectivity import get_pool
            except ImportError:
                from connectivity import get_pool
            get_pool()
        yield

    api = FastAPI(
        title="Pando API",
        version="1.0.0",
        lifespan=lifespan,
    )
    api.state.settings = settings

    if settings.cors_origins:
        api.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization", settings.player_id_header],
        )

    def open_store() -> Iterator[GameStore]:
        with GameStore(
            settings.database_path,
            verifier=verifier,
            initialize_schema=False,
        ) as store:
            yield store

    def authenticated_identity(request: Request) -> Identity:
        player_id = request.headers.get(settings.player_id_header, "").strip()
        if not player_id:
            detail = (
                "Missing authenticated identity header: "
                f"{settings.player_id_header}"
            )
            if settings.auth_mode == "trusted-header":
                detail += "; configure the authentication proxy to supply it"
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
        return Identity(player_id=player_id)

    @api.exception_handler(PlayerNotFoundError)
    async def player_not_found(
        _request: Request, error: PlayerNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "player_not_found",
                "message": f"Player '{error.args[0]}' has not been registered.",
            },
        )

    @api.exception_handler(InvalidSequenceError)
    async def invalid_sequence(
        _request: Request, error: InvalidSequenceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "invalid_sequence",
                "message": str(error),
                "verification": _json_value(error.verification),
            },
        )

    @api.exception_handler(ValueError)
    async def invalid_value(_request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "invalid_request", "message": str(error)},
        )

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.put("/api/me")
    def save_player(
        payload: PlayerPayload,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, str]:
        store.save_player(identity.player_id, payload.display_name)
        return {
            "player_id": identity.player_id,
            "display_name": payload.display_name.strip(),
        }

    @api.put("/api/puzzles/{start}/{end}/draft")
    def save_draft(
        start: str,
        end: str,
        payload: WordsPayload,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        words = store.set_current_sequence(
            identity.player_id, start, end, payload.words
        )
        return {"words": words}

    @api.get("/api/puzzles/{start}/{end}/draft")
    def get_draft(
        start: str,
        end: str,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        return {"words": store.current_sequence(identity.player_id, start, end)}

    @api.delete("/api/puzzles/{start}/{end}/draft")
    def clear_draft(
        start: str,
        end: str,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, bool]:
        return {
            "cleared": store.clear_current_sequence(identity.player_id, start, end)
        }

    @api.post("/api/puzzles/{start}/{end}/submissions")
    def submit_sequence(
        start: str,
        end: str,
        payload: WordsPayload,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        result = store.save_sequence(identity.player_id, start, end, payload.words)
        return {
            "verified": True,
            "sequence": asdict(result.sequence),
            "is_personal_best": result.is_personal_best,
        }

    @api.get("/api/puzzles/{start}/{end}/submissions")
    def list_submissions(
        start: str,
        end: str,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        sequences = store.list_player_sequences(identity.player_id, start, end)
        return {"sequences": [asdict(sequence) for sequence in sequences]}

    @api.get("/api/puzzles/{start}/{end}/personal-best")
    def get_personal_best(
        start: str,
        end: str,
        identity: Identity = Depends(authenticated_identity),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        best = store.personal_best(identity.player_id, start, end)
        return {"sequence": None if best is None else asdict(best)}

    @api.get("/api/puzzles/{start}/{end}/leaderboard")
    def get_leaderboard(
        start: str,
        end: str,
        limit: int = Query(default=100, ge=1, le=500),
        store: GameStore = Depends(open_store),
    ) -> dict[str, object]:
        entries = store.leaderboard(start, end, limit)
        return {"entries": [asdict(entry) for entry in entries]}

    return api


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


app = create_app()
