#!/usr/bin/env python3
"""Generate Sequence puzzles from a small deterministic Word2Vec model."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DIMENSIONS = 24
WINDOW = 2
EPOCHS = 70
LEARNING_RATE = 0.035
NEGATIVE_SAMPLES = 2
MAX_TIER = 3
STOP_WORDS = {"a", "an", "and", "for", "from", "of", "or", "the", "to", "where"}


Vector = List[float]


@dataclass(frozen=True)
class SeedCorpus:
    domains: List[Mapping[str, object]]
    endpoint_pairs: List[Mapping[str, str]]
    sentences: List[List[str]]
    vocabulary: List[str]
    word_domains: Dict[str, str]


@dataclass(frozen=True)
class Word2VecModel:
    vectors: Dict[str, Vector]
    vocabulary: List[str]


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def tokenize_text(value: str) -> List[str]:
    clean = re.sub(r"[^a-z0-9 ]", " ", normalize(value))
    return [token for token in clean.split(" ") if token and token not in STOP_WORDS]


def unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def load_seed_corpus(source: object) -> SeedCorpus:
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = source

    domains = list(data.get("domains", []))
    endpoint_pairs = list(data.get("endpoint_pairs", []))
    sentences: List[List[str]] = []
    word_domains: Dict[str, str] = {}

    for domain in domains:
        domain_name = str(domain.get("name", "general"))
        for sentence in domain.get("sentences", []):
            clean_sentence = [normalize(word) for word in sentence if normalize(word)]
            if len(clean_sentence) < 2:
                continue
            sentences.append(clean_sentence)
            sentences.append(list(reversed(clean_sentence)))
            for word in clean_sentence:
                word_domains.setdefault(word, domain_name)

    for pair in endpoint_pairs:
        prompt_words = tokenize_text(pair.get("prompt", ""))
        sentence = [normalize(pair["start"]), *prompt_words, normalize(pair["end"])]
        if len(sentence) >= 2:
            sentences.append(sentence)
        word_domains.setdefault(normalize(pair["start"]), "endpoint")
        word_domains.setdefault(normalize(pair["end"]), "endpoint")

    vocabulary = sorted(unique(word for sentence in sentences for word in sentence))
    return SeedCorpus(domains, endpoint_pairs, sentences, vocabulary, word_domains)


def hash_word(value: str) -> int:
    hash_value = 2166136261
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def initial_vector(word: str, salt: str) -> Vector:
    return [
        (hash_word(f"{word}:{salt}:{index}") / 4294967295 - 0.5) * 0.18
        for index in range(DIMENSIONS)
    ]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-clamp(value, -8, 8)))


def normalize_vector(vector: Sequence[float]) -> Vector:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return list(vector)
    return [value / magnitude for value in vector]


def update_pair(input_vector: Vector, output_vector: Vector, label: int, rate: float) -> None:
    prediction = sigmoid(dot(input_vector, output_vector))
    gradient = (label - prediction) * rate

    for index, input_value in enumerate(list(input_vector)):
        output_value = output_vector[index]
        input_vector[index] += gradient * output_value
        output_vector[index] += gradient * input_value


def negative_word(
    vocabulary: Sequence[str],
    source: str,
    context: str,
    pair_index: int,
    epoch: int,
    negative_index: int,
) -> str:
    index = (pair_index * 37 + epoch * 101 + negative_index * 17) % len(vocabulary)
    for _ in range(len(vocabulary)):
        word = vocabulary[index]
        if word != source and word != context:
            return word
        index = (index + 1) % len(vocabulary)
    return vocabulary[index]


def train_model(corpus: SeedCorpus) -> Word2VecModel:
    input_vectors = {word: initial_vector(word, "input") for word in corpus.vocabulary}
    output_vectors = {word: initial_vector(word, "output") for word in corpus.vocabulary}
    pairs: List[Tuple[str, str]] = []

    for sentence in corpus.sentences:
        for source_index, source in enumerate(sentence):
            start = max(0, source_index - WINDOW)
            end = min(len(sentence) - 1, source_index + WINDOW)
            for context_index in range(start, end + 1):
                if context_index != source_index:
                    pairs.append((source, sentence[context_index]))

    for epoch in range(EPOCHS):
        rate = LEARNING_RATE * (1 - epoch / EPOCHS * 0.65)
        for pair_index, (source, context) in enumerate(pairs):
            update_pair(input_vectors[source], output_vectors[context], 1, rate)
            for negative_index in range(NEGATIVE_SAMPLES):
                negative = negative_word(corpus.vocabulary, source, context, pair_index, epoch, negative_index)
                update_pair(input_vectors[source], output_vectors[negative], 0, rate)

    vectors: Dict[str, Vector] = {}
    for word in corpus.vocabulary:
        vectors[word] = normalize_vector([
            input_vectors[word][index] + output_vectors[word][index]
            for index in range(DIMENSIONS)
        ])

    return Word2VecModel(vectors, corpus.vocabulary)


def average_vectors(weighted_vectors: Iterable[Tuple[Sequence[float], float]]) -> Vector:
    output = [0.0] * DIMENSIONS
    total_weight = 0.0

    for vector, weight in weighted_vectors:
        if weight <= 0:
            continue
        for index, value in enumerate(vector):
            output[index] += value * weight
        total_weight += weight

    if total_weight == 0:
        return output
    return normalize_vector([value / total_weight for value in output])


def vector_for(model: Word2VecModel, word: str) -> Vector:
    clean = normalize(word)
    if clean in model.vectors:
        return model.vectors[clean]

    token_vectors = [(model.vectors[token], 1.0) for token in tokenize_text(clean) if token in model.vectors]
    if token_vectors:
        return average_vectors(token_vectors)

    return [0.0] * DIMENSIONS


def similarity(model: Word2VecModel, left: str, right: str) -> float:
    return cosine(vector_for(model, left), vector_for(model, right))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0
    return max(0.0, dot(left, right) / (left_magnitude * right_magnitude))


def connectability(model: Word2VecModel, candidate: str, left: str, right: str) -> float:
    return math.sqrt(similarity(model, candidate, left) * similarity(model, candidate, right))


def interpolated_target(model: Word2VecModel, start: str, end: str, slot_index: int, tier: int) -> Vector:
    alpha = (slot_index + 1) / (tier + 1)
    start_vector = vector_for(model, start)
    end_vector = vector_for(model, end)
    return normalize_vector([
        start_vector[index] * (1 - alpha) + end_vector[index] * alpha
        for index in range(DIMENSIONS)
    ])


def candidate_score(
    model: Word2VecModel,
    candidate: str,
    left: str,
    right: str,
    target_vector: Sequence[float],
) -> float:
    bridge = connectability(model, candidate, left, right)
    target_similarity = cosine(vector_for(model, candidate), target_vector)
    return bridge * 0.72 + target_similarity * 0.28


def route_score(model: Word2VecModel, start: str, end: str, route: Sequence[str]) -> float:
    chain = [start, *route, end]
    scores = [
        connectability(model, chain[index], chain[index - 1], chain[index + 1])
        for index in range(1, len(chain) - 1)
    ]
    if not scores:
        return 0.0
    return similarity(model, start, end) * 0.25 + sum(scores) / len(scores) * 0.75


def ranked_routes(
    corpus: SeedCorpus,
    model: Word2VecModel,
    start: str,
    end: str,
    tier: int,
    limit: int,
) -> List[List[str]]:
    beams: List[Tuple[List[str], float]] = [([], 0.0)]

    for slot_index in range(tier):
        next_beams: List[Tuple[List[str], float]] = []
        for route, partial_score in beams:
            used = {start, end, *route}
            left = route[-1] if route else start
            right = end
            target_vector = interpolated_target(model, start, end, slot_index, tier)
            candidates = []
            for candidate in corpus.vocabulary:
                if candidate in used or candidate in STOP_WORDS:
                    continue
                score = candidate_score(model, candidate, left, right, target_vector)
                candidates.append((candidate, score))

            for candidate, score in sorted(candidates, key=lambda item: (-item[1], item[0]))[:12]:
                next_beams.append(([*route, candidate], partial_score + score))

        beams = sorted(next_beams, key=lambda item: (-item[1], item[0]))[:24]

    completed = [(route, route_score(model, start, end, route)) for route, _ in beams]
    deduped: List[List[str]] = []
    seen = set()
    for route, _ in sorted(completed, key=lambda item: (-item[1], item[0])):
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(route)
        if len(deduped) == limit:
            break
    return deduped


def clue_for_slot(corpus: SeedCorpus, model: Word2VecModel, routes: Sequence[Sequence[str]], slot_index: int) -> Mapping[str, object]:
    slot_words = unique(route[slot_index] for route in routes)
    domain_counts: Dict[str, int] = {}
    for word in slot_words:
        domain = corpus.word_domains.get(word, "semantic")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    domain = sorted(domain_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    related_scores = []
    for candidate in corpus.vocabulary:
        if candidate in STOP_WORDS:
            continue
        score = max(similarity(model, candidate, word) for word in slot_words)
        related_scores.append((candidate, score))
    related = [
        word for word, _ in sorted(related_scores, key=lambda item: (-item[1], item[0]))[:6]
    ]

    return {
        "category": f"{domain} connectability {slot_index + 1}",
        "related": related,
    }


def calculated_tier(model: Word2VecModel, start: str, end: str) -> int:
    score = similarity(model, start, end)
    if score >= 0.64:
        return 1
    if score >= 0.48:
        return 2
    return 3


def endpoint_candidates(corpus: SeedCorpus, model: Word2VecModel) -> List[Mapping[str, object]]:
    candidates = []
    for pair in corpus.endpoint_pairs:
        start = normalize(pair["start"])
        end = normalize(pair["end"])
        endpoint_similarity = similarity(model, start, end)
        candidates.append({
            "start": start,
            "end": end,
            "prompt": pair.get("prompt") or f"Connect {start} to {end}.",
            "similarity": endpoint_similarity,
            "calculatedTier": calculated_tier(model, start, end),
        })
    return sorted(candidates, key=lambda item: (-item["similarity"], item["start"], item["end"]))


def generate_puzzles(
    corpus: SeedCorpus,
    model: Word2VecModel,
    count_per_tier: int = 2,
    routes_per_puzzle: int = 3,
) -> List[Mapping[str, object]]:
    puzzles: List[Mapping[str, object]] = []
    endpoints = endpoint_candidates(corpus, model)
    used_pairs = set()

    for tier in range(1, MAX_TIER + 1):
        scored = []
        for endpoint in endpoints:
            pair_key = (endpoint["start"], endpoint["end"])
            routes = ranked_routes(corpus, model, endpoint["start"], endpoint["end"], tier, routes_per_puzzle)
            if len(routes) < routes_per_puzzle:
                continue
            score = max(route_score(model, endpoint["start"], endpoint["end"], route) for route in routes)
            if pair_key in used_pairs:
                score *= 0.97
            score -= abs(endpoint["calculatedTier"] - tier) * 0.04
            scored.append((score, endpoint, routes))

        for _, endpoint, routes in sorted(scored, key=lambda item: (-item[0], item[1]["start"]))[:count_per_tier]:
            used_pairs.add((endpoint["start"], endpoint["end"]))
            puzzles.append({
                "id": f"w2v-{tier}-{endpoint['start']}-{endpoint['end']}",
                "tier": tier,
                "start": endpoint["start"],
                "end": endpoint["end"],
                "prompt": endpoint["prompt"],
                "similarity": round(endpoint["similarity"], 4),
                "routes": routes,
                "clues": [
                    clue_for_slot(corpus, model, routes, slot_index)
                    for slot_index in range(tier)
                ],
            })

    return puzzles


def generate_browser_data(corpus: SeedCorpus, model: Word2VecModel) -> Mapping[str, object]:
    puzzles = generate_puzzles(corpus, model)
    vectors = {
        word: [round(value, 6) for value in model.vectors[word]]
        for word in model.vocabulary
    }
    poc_source = next((puzzle for puzzle in puzzles if puzzle["tier"] == 3), puzzles[-1])
    multi_route_trial = dict(poc_source)
    multi_route_trial.update({
        "id": "multi-route",
        "prompt": f"POC A: a forgiving verifier accepts several generated routes from {poc_source['start']} to {poc_source['end']}.",
        "verifierNote": "Multiple generated route trees accepted. This is the recommended playability model.",
    })
    single_route_trial = dict(poc_source)
    single_route_trial.update({
        "id": "single-route",
        "prompt": f"POC B: a strict verifier accepts one generated route from {poc_source['start']} to {poc_source['end']}.",
        "verifierNote": "Only one generated route accepted. Use this to feel how brittle a one-answer puzzle becomes.",
        "routes": [poc_source["routes"][0]],
        "clues": [
            clue_for_slot(corpus, model, [poc_source["routes"][0]], slot_index)
            for slot_index in range(poc_source["tier"])
        ],
    })
    return {
        "metadata": {
            "algorithm": "custom-deterministic-word2vec",
            "dimensions": DIMENSIONS,
            "similarity": "max(0, cosine(vector(a), vector(b)))",
            "connectability": "sqrt(similarity(candidate,left) * similarity(candidate,right))",
        },
        "puzzles": puzzles,
        "pocTrials": [multi_route_trial, single_route_trial],
        "scoring": {
            "vocabulary": model.vocabulary,
            "vectors": vectors,
        },
    }


def write_browser_data(data: Mapping[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    output_path.write_text(f"window.SEQUENCE_DATA = {payload};\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate static Sequence puzzle data.")
    parser.add_argument("--seed", type=Path, default=root / "algorithm" / "seed_words.json")
    parser.add_argument("--output", type=Path, default=root / "sequence-data.js")
    args = parser.parse_args()

    corpus = load_seed_corpus(args.seed)
    model = train_model(corpus)
    write_browser_data(generate_browser_data(corpus, model), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
