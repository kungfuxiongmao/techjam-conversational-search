from __future__ import annotations

import math
import zlib
from array import array
from collections import defaultdict

from starter.vocabulary import canonical_tokens


MASK_64 = (1 << 64) - 1
SEMANTIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}


def _feature_seed(feature: str) -> int:
    first = zlib.crc32(feature.encode("utf-8"))
    second = zlib.crc32(feature[::-1].encode("utf-8"), 0x9E3779B9)
    return ((first << 32) | second) & MASK_64


def _unique_features(value: object, limit: int = 32) -> tuple[str, ...]:
    tokens = canonical_tokens(value)
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) <= 1 or token in SEMANTIC_STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return tuple(result)


class LocalSemanticIndex:
    """Dependency-free dense random-indexing representation for local search.

    Each canonical concept has a deterministic dense vector. A document vector
    is the normalized sum of its concept vectors; no model download, API call,
    training data, or Python hash randomization is involved.
    """

    def __init__(self, dimensions: int = 32, posting_buckets: int = 4096) -> None:
        self.dimensions = dimensions
        self._vectors = array("f")
        self._identifiers: list[str] = []
        self._posting_buckets = posting_buckets
        self._postings = [array("I") for _ in range(posting_buckets)]

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) % self._posting_buckets

    def encode(self, value: object) -> array:
        features = _unique_features(value)
        vector = array("f", [0.0]) * self.dimensions
        for feature in features:
            state = _feature_seed(feature) or 0xA0761D6478BD642F
            # Xorshift emits a reproducible dense random index for every concept.
            for dimension in range(self.dimensions):
                state ^= (state << 13) & MASK_64
                state ^= state >> 7
                state ^= (state << 17) & MASK_64
                vector[dimension] += 1.0 if state & 1 else -1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm:
            for dimension in range(self.dimensions):
                vector[dimension] /= norm
        return vector

    def add(self, identifier: str, value: object) -> None:
        row_id = len(self._identifiers)
        self._identifiers.append(identifier)
        self._vectors.extend(self.encode(value))
        # Canonical postings cheaply narrow 50k dense vectors to a useful
        # neighborhood before cosine scoring. Hash buckets avoid retaining a
        # large Python dictionary and millions of duplicate token objects.
        for bucket in {self._bucket(feature) for feature in _unique_features(value)}:
            self._postings[bucket].append(row_id)

    def _cosine(self, query: array, row_id: int) -> float:
        offset = row_id * self.dimensions
        return sum(query[index] * self._vectors[offset + index] for index in range(self.dimensions))

    def search(self, value: object, limit: int = 240) -> dict[str, float]:
        query_features = _unique_features(value)
        if not query_features:
            return {}
        candidate_weights: defaultdict[int, float] = defaultdict(float)
        corpus_size = max(1, len(self._identifiers))
        buckets = sorted(
            {self._bucket(feature) for feature in query_features},
            key=lambda bucket: len(self._postings[bucket]),
        )
        useful_buckets = [
            bucket
            for bucket in buckets
            if len(self._postings[bucket]) <= max(4000, corpus_size // 12)
        ]
        for bucket in (useful_buckets or buckets[:1])[:8]:
            posting = self._postings[bucket]
            if not posting:
                continue
            inverse_frequency = math.log1p(corpus_size / len(posting))
            for row_id in posting:
                candidate_weights[row_id] += inverse_frequency
        if not candidate_weights:
            return {}
        # The lexical-concept overlap is only a prefilter; final ordering is by
        # the dense-vector cosine, with overlap as a stable secondary signal.
        shortlist = sorted(candidate_weights, key=lambda row: (candidate_weights[row], -row), reverse=True)
        shortlist = shortlist[: max(limit * 4, 320)]
        query = self.encode(value)
        ranked = sorted(
            shortlist,
            key=lambda row: (self._cosine(query, row), candidate_weights[row], -row),
            reverse=True,
        )[:limit]
        return {self._identifiers[row]: self._cosine(query, row) for row in ranked}

    def similarity(self, left: object, right: object) -> float:
        left_vector = self.encode(left)
        right_vector = self.encode(right)
        return sum(a * b for a, b in zip(left_vector, right_vector))

    @property
    def document_count(self) -> int:
        return len(self._identifiers)

    def document_frequency(self, feature: str) -> int:
        return len(self._postings[self._bucket(feature)])
