"""Word-connectivity verification engine for Pando.

Words connect when the cosine similarity of their pretrained word2vec vectors
(word2vec-google-news-300) clears a calibrated threshold. A player sequence of
any length verifies when every adjacent link clears the bar; each link is
signed off with its similarity value. The solver publishes the shortest
verified route as the benchmark, searching the same cleaned candidate pool
players must draw connector words from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = "word2vec-google-news-300"
# The archive holds 3M entries; beyond the first half-million it is almost
# entirely rare phrase tokens, so conversion prunes there to keep the
# converted file (~600MB) inside modest disk and RAM budgets.
VOCAB_LIMIT = 500_000
POOL_SIZE = 50_000
# Words whose above-threshold neighbor count lands in the top percentile are
# promiscuous hubs; dropping them keeps shortest routes interesting. An
# absolute degree cap does not survive threshold changes — at a permissive
# threshold nearly every word exceeds any fixed count.
HUB_PERCENTILE = 99.0
DEFAULT_MAX_HOPS = 8
POOL_WORD_PATTERN = re.compile(r"^[a-z]{2,}$")

# Function words are excluded outright; generic-noun hubs are caught by the
# degree cap at pool build time.
STOP_WORDS = frozenset("""
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can could did do does
    doing down during each few for from further had has have having he her
    here hers herself him himself his how i if in into is it its itself just
    let me more most my myself no nor not now of off on once only or other
    our ours ourselves out over own same she should so some such than that
    the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself yourselves
""".split())

_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"
_CALIBRATION = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
DEFAULT_THRESHOLD: float = _CALIBRATION["threshold"]

_MODEL = None
_POOL = None


@dataclass(frozen=True)
class Link:
    """One adjacent connection, signed off (verified) or rejected with a reason."""

    left: str
    right: str
    cosine: Optional[float]
    verified: bool
    reason: Optional[str]


@dataclass(frozen=True)
class SequenceVerification:
    start: str
    end: str
    words: Tuple[str, ...]
    links: Tuple[Link, ...]
    verified: bool
    length: int


@dataclass(frozen=True)
class Route:
    start: str
    end: str
    words: Tuple[str, ...]
    links: Tuple[Link, ...]
    length: int
    total_cosine: float


@dataclass
class TreeNode:
    word: str
    branches: List["TreeBranch"]


@dataclass(frozen=True)
class TreeBranch:
    link: Link
    node: TreeNode


@dataclass(frozen=True)
class SolutionTree:
    root: TreeNode
    routes: Tuple[Route, ...]


@dataclass(frozen=True)
class Pool:
    words: Tuple[str, ...]
    indices: Dict[str, int]
    matrix: np.ndarray  # (N, dims) float32, unit rows


def load_model():
    """Load the pretrained vectors once per process.

    The first load converts gensim's downloaded archive to a native
    KeyedVectors file so later loads memory-map in seconds. If the model was
    never downloaded, fail fast with the exact command to fetch it.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from gensim.models import KeyedVectors

    base = Path.home() / "gensim-data" / MODEL_NAME
    native = base / f"{MODEL_NAME}.kv"
    archive = base / f"{MODEL_NAME}.gz"

    if native.exists():
        _MODEL = KeyedVectors.load(str(native), mmap="r")
    elif archive.exists():
        _MODEL = KeyedVectors.load_word2vec_format(
            str(archive), binary=True, limit=VOCAB_LIMIT
        )
        _MODEL.save(str(native))
    else:
        raise RuntimeError(
            f"Pretrained model '{MODEL_NAME}' not found. Download it once with:\n"
            f"  python -m gensim.downloader --download {MODEL_NAME}"
        )
    return _MODEL


def _pool_cache_path(size: int, threshold: float, percentile: float) -> Path:
    return (
        Path.home()
        / "gensim-data"
        / f"sequence-pool-v2-{size}-{threshold:.4f}-{percentile:g}.json"
    )


def get_pool(model=None, size: int = POOL_SIZE, hub_percentile: float = HUB_PERCENTILE) -> Pool:
    """The canonical candidate pool: top-frequency clean lowercase words,
    stop-words removed, top-percentile hub words dropped.

    Built with the calibrated default threshold regardless of per-call
    overrides, so players and the solver always share one canonical pool.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    model = model if model is not None else load_model()
    cache = _pool_cache_path(size, DEFAULT_THRESHOLD, hub_percentile)
    if cache.exists():
        words = json.loads(cache.read_text(encoding="utf-8"))
    else:
        words = _build_pool_words(model, size, DEFAULT_THRESHOLD, hub_percentile)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(words), encoding="utf-8")

    matrix = np.stack([model.get_vector(word, norm=True) for word in words]).astype(
        np.float32
    )
    _POOL = Pool(tuple(words), {word: i for i, word in enumerate(words)}, matrix)
    return _POOL


def _build_pool_words(model, size: int, threshold: float, percentile: float) -> List[str]:
    words: List[str] = []
    for word in model.index_to_key:  # frequency order
        if POOL_WORD_PATTERN.match(word) and word not in STOP_WORDS:
            words.append(word)
            if len(words) == size:
                break

    matrix = np.stack([model.get_vector(word, norm=True) for word in words]).astype(
        np.float32
    )
    degrees = np.zeros(len(words), dtype=np.int64)
    chunk = 1024
    for row in range(0, len(words), chunk):
        sims = matrix[row : row + chunk] @ matrix.T
        degrees[row : row + chunk] = (sims >= threshold).sum(axis=1) - 1

    cap = np.percentile(degrees, percentile)
    return [word for word, degree in zip(words, degrees) if degree <= cap]


def _lookup_vector(model, word: str) -> Optional[np.ndarray]:
    """Unit vector for a word, trying sensible case and phrasing variants."""
    cleaned = word.strip()
    candidates = [cleaned, cleaned.lower(), cleaned.capitalize(), cleaned.title(), cleaned.upper()]
    if " " in cleaned:
        candidates += [variant.replace(" ", "_") for variant in list(candidates)]
    for candidate in candidates:
        if candidate in model.key_to_index:
            return model.get_vector(candidate, norm=True)
    return None


_SUFFIXES = ("ings", "ing", "iest", "ies", "ied", "edly", "ed", "ers", "er", "est", "es", "ly", "s")


@lru_cache(maxsize=None)
def _stems(word: str) -> frozenset:
    """Plausible stems for the inflection guard. Deterministic, dependency-free."""
    base = word.casefold()
    stems = {base}
    for suffix in _SUFFIXES:
        if base.endswith(suffix) and len(base) - len(suffix) >= 3:
            stem = base[: -len(suffix)]
            stems.add(stem)
            if len(stem) >= 4 and stem[-1] == stem[-2]:  # running -> runn -> run
                stems.add(stem[:-1])
            if suffix in ("ies", "ied", "iest"):  # cities -> city
                stems.add(stem + "y")
    return frozenset(stems)


def _are_variants(a: str, b: str) -> bool:
    if a.casefold() == b.casefold():
        return True
    return bool(_stems(a) & _stems(b))


def _is_collocation(model, left: str, right: str) -> bool:
    """True when the two words form a known multi-word entity in the model
    (e.g. star + wars -> Star_Wars). Pop-culture pairings often live in these
    entity vectors rather than in the single lowercase tokens.
    """
    for a in (left, left.capitalize()):
        for b in (right, right.capitalize()):
            if f"{a}_{b}" in model.key_to_index or f"{b}_{a}" in model.key_to_index:
                return True
    return False


def link(left: str, right: str, threshold: Optional[float] = None, model=None) -> Link:
    """Check one connection and sign it off or reject it with a reason.

    A link verifies when its cosine clears the threshold, or when the pair
    forms a known collocation entity (the star<->wars rule).
    """
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    model = model if model is not None else load_model()

    left_vector = _lookup_vector(model, left)
    right_vector = _lookup_vector(model, right)
    if left_vector is None or right_vector is None:
        return Link(left, right, None, False, "not-in-vocabulary")

    cosine = float(left_vector @ right_vector)
    if _are_variants(left, right):
        return Link(left, right, cosine, False, "inflection-variant")
    if cosine < threshold and not _is_collocation(model, left, right):
        return Link(left, right, cosine, False, "below-threshold")
    return Link(left, right, cosine, True, None)


def verify_sequence(
    start: str,
    end: str,
    words: Sequence[str],
    threshold: Optional[float] = None,
    model=None,
) -> SequenceVerification:
    """Verify a player sequence: every adjacent link must be signed off.

    Connector words must come from the canonical pool (the same word space the
    benchmark solver searches) and may not repeat any word in the chain.
    """
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    model = model if model is not None else load_model()
    pool = get_pool(model)

    connectors = tuple(word.strip() for word in words)
    chain = [start.strip(), *connectors, end.strip()]
    seen = {chain[0].casefold(), chain[-1].casefold()}
    links: List[Link] = []

    for index in range(1, len(chain)):
        left, right = chain[index - 1], chain[index]
        checked = link(left, right, threshold, model)
        if index <= len(connectors):  # `right` is a player connector
            if right.casefold() in seen:
                checked = Link(left, right, checked.cosine, False, "repeated-word")
            elif right.casefold() not in pool.indices:
                checked = Link(left, right, checked.cosine, False, "not-in-pool")
            seen.add(right.casefold())
        links.append(checked)

    return SequenceVerification(
        start=chain[0],
        end=chain[-1],
        words=connectors,
        links=tuple(links),
        verified=all(item.verified for item in links),
        length=len(connectors),
    )


def shortest_route(
    start: str,
    end: str,
    threshold: Optional[float] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    model=None,
) -> Optional[Route]:
    """Shortest verified route from start to end, or None when none exists
    within max_hops intermediate words. Deterministic: among routes of minimal
    length, the highest total link cosine wins, then alphabetical order.

    The search only follows cosine-threshold edges, not collocation links —
    a player who spots a collocation shortcut can legitimately beat the
    published benchmark, which is exactly what bounties are for.
    """
    routes = _shortest_routes(start, end, threshold, max_hops, model, limit=1)
    return routes[0] if routes else None


def build_solution_tree(
    start: str,
    end: str,
    threshold: Optional[float] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_routes: int = 25,
    model=None,
) -> Optional[SolutionTree]:
    """The shortest verified routes merged into a tree rooted at start.

    Every edge is a signed-off Link; every root-to-leaf path ends at `end`
    and re-verifies with verify_sequence. None when no route exists.
    """
    routes = _shortest_routes(start, end, threshold, max_hops, model, limit=max_routes)
    if not routes:
        return None

    root = TreeNode(routes[0].start, [])
    for route in routes:
        node = root
        for edge in route.links:
            branch = next(
                (item for item in node.branches if item.node.word == edge.right), None
            )
            if branch is None:
                branch = TreeBranch(edge, TreeNode(edge.right, []))
                node.branches.append(branch)
            node = branch.node
    return SolutionTree(root=root, routes=tuple(routes))


def is_valid_puzzle(
    start: str,
    end: str,
    threshold: Optional[float] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    model=None,
) -> bool:
    """A publishable puzzle: both endpoints known, not trivially connected
    (no direct link, no shared stem), and at least one verified route exists.
    """
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    model = model if model is not None else load_model()

    if _lookup_vector(model, start) is None or _lookup_vector(model, end) is None:
        return False
    if link(start, end, threshold, model).verified or _are_variants(start, end):
        return False
    return shortest_route(start, end, threshold, max_hops, model) is not None


# For each discovered pool index: all shortest-depth parents as
# (parent_index, edge_cosine), where -1 stands for the start word.
_Parents = Dict[int, List[Tuple[int, float]]]


def _shortest_routes(
    start: str,
    end: str,
    threshold: Optional[float],
    max_hops: int,
    model,
    limit: int,
) -> List[Route]:
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    model = model if model is not None else load_model()
    pool = get_pool(model)

    start_vector = _lookup_vector(model, start)
    end_vector = _lookup_vector(model, end)
    if start_vector is None or end_vector is None:
        return []

    direct = link(start, end, threshold, model)
    if direct.verified:
        return [
            Route(start, end, (), (direct,), 0, round(direct.cosine, 6))
        ]

    excluded = {start.casefold(), end.casefold()}
    blocked = np.zeros(len(pool.words), dtype=bool)
    for word in excluded:
        if word in pool.indices:
            blocked[pool.indices[word]] = True

    end_sims = pool.matrix @ end_vector
    visits: _Parents = {}
    frontier: List[int] = []

    # Level 1: expand from the start word itself.
    start_sims = pool.matrix @ start_vector
    for index in np.flatnonzero((start_sims >= threshold) & ~blocked):
        index = int(index)
        if _are_variants(start, pool.words[index]):
            continue
        visits[index] = [(-1, float(start_sims[index]))]
        frontier.append(index)

    for depth in range(1, max_hops + 1):
        if not frontier:
            return []

        finishers = [
            index
            for index in frontier
            if end_sims[index] >= threshold and not _are_variants(pool.words[index], end)
        ]
        if finishers:
            return _assemble_routes(
                start, end, threshold, pool, visits, finishers, end_sims, model, limit
            )

        if depth == max_hops:
            return []
        frontier = _expand(pool, visits, frontier, blocked, threshold)

    return []


def _expand(
    pool: Pool,
    visits: _Parents,
    frontier: List[int],
    blocked: np.ndarray,
    threshold: float,
) -> List[int]:
    """One level-synchronous BFS expansion over the pool matrix."""
    known = blocked.copy()
    for index in visits:
        known[index] = True

    discovered: Dict[int, List[Tuple[int, float]]] = {}
    chunk = 512
    for offset in range(0, len(frontier), chunk):
        rows = frontier[offset : offset + chunk]
        sims = pool.matrix[rows] @ pool.matrix.T
        local, targets = np.nonzero((sims >= threshold) & ~known)
        for row, target in zip(local.tolist(), targets.tolist()):
            source = rows[row]
            if _are_variants(pool.words[source], pool.words[target]):
                continue
            discovered.setdefault(target, []).append((source, float(sims[row, target])))

    visits.update(discovered)
    return sorted(discovered)


def _assemble_routes(
    start: str,
    end: str,
    threshold: float,
    pool: Pool,
    visits: Dict[int, _Visit],
    finishers: List[int],
    end_sims: np.ndarray,
    model,
    limit: int,
) -> List[Route]:
    """Enumerate shortest paths through recorded parents, deterministically
    ranked by total link cosine (desc) then alphabetical word order."""
    candidates: List[Tuple[Tuple[str, ...], float]] = []
    seen_paths = set()
    # Enumerate more than requested so ranking by total cosine is stable
    # before trimming to the limit.
    budget = max(limit * 8, 64)

    def backtrack(index: int, suffix: Tuple[int, ...], cosines: float):
        if len(candidates) >= budget:
            return
        for parent, cosine in sorted(visits[index], key=lambda offer: -offer[1]):
            if parent == -1:
                words = tuple(pool.words[i] for i in (index, *suffix))
                if words not in seen_paths:
                    seen_paths.add(words)
                    candidates.append((words, cosines + cosine))
            elif parent not in suffix and parent != index:
                backtrack(parent, (index, *suffix), cosines + cosine)

    for finisher in sorted(finishers, key=lambda i: -float(end_sims[i])):
        backtrack(finisher, (), float(end_sims[finisher]))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    routes = []
    for words, total in candidates[:limit]:
        chain = [start, *words, end]
        links = tuple(
            link(chain[i], chain[i + 1], threshold, model) for i in range(len(chain) - 1)
        )
        routes.append(Route(start, end, words, links, len(words), round(total, 6)))
    return routes
