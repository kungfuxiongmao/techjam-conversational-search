from __future__ import annotations

import json
import math
import re
import sqlite3
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starter.semantic import LocalSemanticIndex
from starter.tracker import active_constraints
from starter.vocabulary import canonical_tokens, expanded_search_terms


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CONCEPT_MASK_BITS = 2048
DISPLAY_COLORS = {
    "beige", "black", "blue", "brown", "gold", "gray", "green", "navy",
    "orange", "pink", "purple", "red", "silver", "white", "yellow",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "matters", "key", "requirement", "still", "exploring", "have", "has",
}

# Centralized calibration values. Exact field evidence intentionally outweighs
# dense similarity; the latter is chiefly a recall signal for Browsing mode.
ATTRIBUTE_COVERAGE_WEIGHTS = {
    "brand": 20.0,
    "color": 14.0,
    "feature": 12.0,
    "material": 18.0,
    "size": 14.0,
    "style": 14.0,
    "use_case": 13.0,
}
ATTRIBUTE_PHRASE_WEIGHTS = {
    "brand": 10.0,
    "color": 7.0,
    "feature": 9.0,
    "material": 9.0,
    "size": 8.0,
    "style": 8.0,
    "use_case": 8.0,
}
SEMANTIC_ROUTE_WEIGHTS = {"browsing": 0.9, "buying": 0.8}
SEMANTIC_RERANK_WEIGHTS = {"browsing": 6.0, "buying": 4.0}
OVERRIDE_MULTIPLIER = 1.5
REDUNDANT_EVIDENCE_FLOOR = 0.75
GENERIC_EVIDENCE = {
    "all", "black", "blue", "brown", "button", "closure", "color", "cotton",
    "gray", "green", "hand", "heather", "imported", "leather", "linen", "made",
    "machine", "material", "nylon", "on", "polyester", "pull", "rayon", "red",
    "silk", "solid", "spandex", "tie", "usa", "wash", "white", "wool", "zipper",
}
HYBRID_CATEGORY_CONCEPTS = {
    "backpack", "loafer", "parka", "robe", "shacket", "slipon", "undershirt", "vest",
}


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normal_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(_flatten(value).casefold()))


def _terms(value: object, limit: int = 48) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(_flatten(value).casefold()):
        if len(token) <= 1 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _numeric_price(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        price = float(value)
        return price if math.isfinite(price) and price >= 0 else None
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    price = float(match.group(0))
    return price if math.isfinite(price) and price >= 0 else None


def _concept_mask(value: object) -> int:
    """Compact two-hash signature used for fast synonym-aware coverage."""

    result = 0
    for concept in set(canonical_tokens(value)):
        if len(concept) <= 1 or concept in STOPWORDS:
            continue
        encoded = concept.encode("utf-8")
        first = zlib.crc32(encoded) % CONCEPT_MASK_BITS
        second = zlib.crc32(encoded, 0x9E3779B9) % CONCEPT_MASK_BITS
        result |= (1 << first) | (1 << second)
    return result


def _concept_present(concept: str, mask: int) -> bool:
    encoded = concept.encode("utf-8")
    first = zlib.crc32(encoded) % CONCEPT_MASK_BITS
    second = zlib.crc32(encoded, 0x9E3779B9) % CONCEPT_MASK_BITS
    return bool(mask & (1 << first) and mask & (1 << second))


def _evidence_concepts(value: object) -> set[str]:
    return {
        token
        for token in canonical_tokens(value)
        if len(token) > 1 and not token.isdigit() and token not in STOPWORDS
    }


def _explicit_color_text(details: object) -> str:
    if not isinstance(details, dict):
        return ""
    return " ".join(
        str(value)
        for key, value in details.items()
        if "color" in str(key).casefold() and value not in (None, "")
    )


def _display_colors(value: object) -> tuple[str, ...]:
    return tuple(token for token in canonical_tokens(value) if token in DISPLAY_COLORS)


@dataclass(frozen=True)
class ProductDocument:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    combined: str
    category_concept_mask: int
    combined_concept_mask: int
    explicit_color_mask: int
    title_colors: tuple[str, ...]
    price: float | None
    average_rating: float
    rating_number: int


@dataclass(frozen=True)
class RankingContext:
    records: tuple[tuple[str, dict[str, Any]], ...]
    evidence_weights: tuple[float, ...]
    hybrid_category: bool


class ProductRetriever:
    """Offline FTS candidate generation followed by deterministic reranking."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.documents: dict[str, ProductDocument] = {}
        self.semantic_index = LocalSemanticIndex()
        self._popular: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE product_fts USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = _flatten(product.get("title"))
                categories = _flatten(product.get("categories"))
                features = _flatten(product.get("features"))
                details = _flatten(product.get("details"))
                store = _flatten(product.get("store"))
                description = _flatten(product.get("description"))
                combined = _normal_text(" ".join((title, categories, features, details, store, description)))
                semantic_source = " ".join((title, features, details, categories, store, description))
                rating = product.get("average_rating")
                rating_number = product.get("rating_number")
                document = ProductDocument(
                    parent_asin=parent_asin,
                    title=_normal_text(title),
                    categories=_normal_text(categories),
                    features=_normal_text(features),
                    details=_normal_text(details),
                    store=_normal_text(store),
                    description=_normal_text(description),
                    combined=combined,
                    category_concept_mask=_concept_mask(f"{categories} {title}"),
                    combined_concept_mask=_concept_mask(semantic_source),
                    explicit_color_mask=_concept_mask(_explicit_color_text(product.get("details"))),
                    title_colors=_display_colors(title),
                    price=_numeric_price(product.get("price")),
                    average_rating=float(rating) if isinstance(rating, (int, float)) else 0.0,
                    rating_number=int(rating_number) if isinstance(rating_number, (int, float)) else 0,
                )
                self.documents[parent_asin] = document
                self.semantic_index.add(parent_asin, semantic_source)
                batch.append((parent_asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO product_fts VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO product_fts VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self._popular = sorted(
            self.documents,
            key=lambda asin: (
                self.documents[asin].average_rating * math.log1p(self.documents[asin].rating_number),
                self.documents[asin].rating_number,
                asin,
            ),
            reverse=True,
        )

    @staticmethod
    def _expression(terms: Iterable[str], operator: str = "OR") -> str:
        unique = list(dict.fromkeys(term for term in terms if term))
        return f" {operator} ".join(f'"{term}"' for term in unique)

    def _search(self, terms: list[str], limit: int, operator: str = "OR") -> list[str]:
        expression = self._expression(terms, operator)
        if not expression:
            return []
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM product_fts WHERE product_fts MATCH ? "
                "ORDER BY bm25(product_fts, 0.0, 7.0, 5.0, 3.0, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    @staticmethod
    def _active_records(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        records: list[tuple[str, dict[str, Any]]] = []
        for attribute, values in state["constraints"].items():
            for record in values:
                if record.get("status") == "active":
                    records.append((attribute, record))
        return records

    def _candidate_scores(self, state: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        constraints = active_constraints(state)
        category_terms = _terms(constraints.get("category", []), 24)
        records = [
            (attribute, record)
            for attribute, record in self._active_records(state)
            if attribute not in {"category", "budget"}
        ]
        records.sort(key=lambda item: (int(item[1].get("turn", 0)), float(item[1].get("confidence", 0.0))))
        clue_terms = _terms([record["value"] for _, record in records], 48)
        latest_terms = _terms([record["value"] for _, record in records[-2:]], 32)
        expanded_terms = expanded_search_terms(
            [*constraints.get("category", []), *[record["value"] for _, record in records]],
            stopwords=STOPWORDS,
            limit=72,
        )

        routes: list[tuple[list[str], str, int, float]] = []
        if expanded_terms:
            routes.append((expanded_terms, "OR", 420, 0.9))
        if category_terms or clue_terms:
            routes.append((category_terms + clue_terms, "OR", 350, 1.0))
        if category_terms and clue_terms:
            # Simulator clues are copied from the target metadata, so the strict
            # category-plus-clue route is high precision even for generic facts.
            routes.append((category_terms + clue_terms[:32], "AND", 220, 2.2))
        if clue_terms:
            routes.append((clue_terms[:24], "AND", 180, 1.8))
        if latest_terms:
            routes.append((latest_terms, "AND", 180, 1.6))
            routes.append((latest_terms, "OR", 220, 1.1))
        if category_terms:
            routes.append((category_terms, "OR", 220, 0.55))
        for _, record in records[-4:]:
            individual = _terms(record["value"], 20)
            if individual:
                routes.append((individual, "AND", 100, 1.25))

        scores: defaultdict[str, float] = defaultdict(float)
        for terms, operator, limit, weight in routes:
            for rank, parent_asin in enumerate(self._search(terms, limit, operator), start=1):
                scores[parent_asin] += weight / (12.0 + rank)
        semantic_query = " ".join(
            str(value)
            for attribute, values in constraints.items()
            if attribute != "budget"
            for value in values
        )
        scenario = str(state.get("scenario_detected"))
        semantic_scores = self.semantic_index.search(semantic_query, 260) if scenario in SEMANTIC_ROUTE_WEIGHTS else {}
        semantic_route_weight = SEMANTIC_ROUTE_WEIGHTS.get(scenario, 0.0)
        for rank, parent_asin in enumerate(semantic_scores, start=1):
            scores[parent_asin] += semantic_route_weight / (12.0 + rank)
        return dict(scores), semantic_scores

    @staticmethod
    def _coverage(terms: Iterable[str], text: str) -> float:
        terms = tuple(terms)
        if not terms:
            return 0.0
        text_terms = set(text.split())
        return sum(term in text_terms for term in terms) / len(terms)

    def _weighted_coverage(self, terms: list[str], concept_mask: int) -> float:
        concepts = list(dict.fromkeys(
            token for token in canonical_tokens(terms) if len(token) > 1 and token not in STOPWORDS
        ))
        if not concepts:
            return 0.0
        weights = [
            math.log1p((self.semantic_index.document_count + 1) /
                       (self.semantic_index.document_frequency(term) + 1))
            for term in concepts
        ]
        denominator = sum(weights)
        if not denominator:
            return 0.0
        return sum(
            weight for term, weight in zip(concepts, weights) if _concept_present(term, concept_mask)
        ) / denominator

    @staticmethod
    def _attribute_field(document: ProductDocument, attribute: str) -> str:
        if attribute == "brand":
            return f"{document.store} {document.title}"
        if attribute == "color":
            return f"{document.details} {document.title} {document.features}"
        if attribute == "material":
            return f"{document.details} {document.features} {document.title} {document.description}"
        if attribute in {"size", "style"}:
            return f"{document.details} {document.features} {document.title}"
        return document.combined

    def _ranking_context(self, state: dict[str, Any]) -> RankingContext:
        records = tuple(self._active_records(state))
        seen: defaultdict[str, set[str]] = defaultdict(set)
        weights: list[float] = []
        for attribute, record in records:
            concepts = _evidence_concepts(record.get("value"))
            previous = seen[attribute]
            novelty = len(concepts - previous) / len(concepts) if concepts else 1.0
            weight = 1.0 if not previous else REDUNDANT_EVIDENCE_FLOOR + (
                1.0 - REDUNDANT_EVIDENCE_FLOOR
            ) * novelty
            previous.update(concepts)
            weights.append(weight)
        return RankingContext(
            records=records,
            evidence_weights=tuple(weights),
            hybrid_category=any(
                attribute == "category"
                and bool(_evidence_concepts(record.get("value")) & HYBRID_CATEGORY_CONCEPTS)
                for attribute, record in records
            ),
        )

    def _color_confidence(self, document: ProductDocument, value_terms: list[str]) -> float:
        requested = set(canonical_tokens(value_terms)) & DISPLAY_COLORS
        if not requested:
            return 1.0
        if document.explicit_color_mask:
            return 1.0 if any(
                _concept_present(color, document.explicit_color_mask) for color in requested
            ) else 0.0
        title_matches = [color for color in document.title_colors if color in requested]
        if title_matches:
            # A leading color plus a different terminal color commonly denotes a
            # named design or band (for example, "Red Hot ... T-Shirt Black").
            if len(set(document.title_colors)) > 1 and document.title_colors[-1] not in requested:
                return 0.75
            return 0.95
        if any(_concept_present(color, document.combined_concept_mask) for color in requested):
            # Feature-list colors often describe fabric blends or available
            # variants, so they are supporting rather than definitive evidence.
            return 0.9
        return 0.0

    def _rerank_score(
        self,
        document: ProductDocument,
        state: dict[str, Any],
        retrieval_score: float,
        semantic_score: float = 0.0,
        context: RankingContext | None = None,
    ) -> float:
        context = context or self._ranking_context(state)
        score = retrieval_score * 100.0
        semantic_weight = SEMANTIC_RERANK_WEIGHTS.get(str(state.get("scenario_detected")), 0.0)
        score += semantic_weight * max(0.0, semantic_score)
        for (attribute, record), evidence_weight in zip(context.records, context.evidence_weights):
            value_terms = _terms(record["value"], 40)
            if not value_terms:
                continue
            if attribute == "budget":
                continue
            if attribute == "category":
                field = f"{document.categories} {document.title}"
                coverage = self._weighted_coverage(value_terms, document.category_concept_mask)
                score += 11.0 * coverage
                if context.hybrid_category:
                    score += 6.0 * self._weighted_coverage(
                        value_terms,
                        _concept_mask(document.title),
                    )
                if " ".join(value_terms) in field:
                    score += 3.0
                continue

            field = self._attribute_field(document, attribute)
            coverage = self._weighted_coverage(value_terms, document.combined_concept_mask)
            field_coverage = self._coverage(value_terms, field)
            if attribute == "color":
                evidence_weight *= self._color_confidence(document, value_terms)
            source_weight = OVERRIDE_MULTIPLIER if record.get("source") == "override" else 1.0
            source_weight *= evidence_weight
            base_weight = ATTRIBUTE_COVERAGE_WEIGHTS.get(attribute, 12.0)
            score += source_weight * base_weight * (0.75 * coverage + 0.25 * field_coverage)
            normalized_phrase = " ".join(value_terms)
            if normalized_phrase and normalized_phrase in field:
                score += source_weight * ATTRIBUTE_PHRASE_WEIGHTS.get(attribute, 8.0)
            informative = _evidence_concepts(record.get("value")) - GENERIC_EVIDENCE
            if len(value_terms) >= 2 and informative and normalized_phrase in document.title:
                score += source_weight * 4.0
            if attribute in {"material", "color", "brand"} and coverage == 1.0:
                score += source_weight * 5.0

        for excluded in state.get("excluded_terms", []):
            if re.search(rf"\b{re.escape(str(excluded).casefold())}\b", document.combined):
                score -= 80.0

        for record in state["constraints"].get("budget", []):
            if record.get("status") != "active" or "numeric_value" not in record:
                continue
            target = float(record["numeric_value"])
            if document.price is None:
                continue
            if record.get("price_mode") == "maximum":
                score += 9.0 if document.price <= target else -min(24.0, 6.0 + document.price - target)
            else:
                scale = max(5.0, target * 0.25)
                score += 14.0 * max(0.0, 1.0 - abs(document.price - target) / scale)

        profile = state.get("user_profile") or {}
        for tag in profile.get("preference_tags", []) or []:
            tag_terms = _terms(tag, 4)
            if tag_terms and self._coverage(tag_terms, document.combined) == 1.0:
                score += 0.25
        if str(profile.get("rating_style", "")).casefold() == "critical":
            score += max(0.0, document.average_rating - 3.5) * 0.15
        score += min(0.25, math.log1p(document.rating_number) / 50.0)
        return score

    def _category_fallback(self, state: dict[str, Any], needed: int, seen: set[str]) -> list[str]:
        categories = active_constraints(state).get("category", [])
        category_terms = _terms(categories, 24)
        result: list[str] = []
        if category_terms:
            for parent_asin in self._search(category_terms, max(needed * 8, 80), "OR"):
                if parent_asin not in seen:
                    seen.add(parent_asin)
                    result.append(parent_asin)
                    if len(result) >= needed:
                        return result
        for parent_asin in self._popular:
            if parent_asin not in seen:
                seen.add(parent_asin)
                result.append(parent_asin)
                if len(result) >= needed:
                    break
        return result

    def recommend(self, state: dict[str, Any], top_k: int = 10) -> list[dict[str, str]]:
        limit = max(1, min(int(top_k), 10))
        retrieval_scores, semantic_scores = self._candidate_scores(state)
        context = self._ranking_context(state)
        ranked = sorted(
            retrieval_scores,
            key=lambda asin: (
                self._rerank_score(
                    self.documents[asin],
                    state,
                    retrieval_scores[asin],
                    semantic_scores.get(asin, 0.0),
                    context,
                ),
                asin,
            ),
            reverse=True,
        )
        selected = ranked[:limit]
        seen = set(selected)
        if len(selected) < limit:
            selected.extend(self._category_fallback(state, limit - len(selected), seen))
        return [{"parent_asin": parent_asin} for parent_asin in selected]
