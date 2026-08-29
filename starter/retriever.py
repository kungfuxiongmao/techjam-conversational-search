from __future__ import annotations

import functools
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starter.tracker import active_constraints


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "matters", "key", "requirement", "still", "exploring", "have", "has",
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


try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    _local_nltk_dir = Path(__file__).resolve().parent.parent / ".venv" / "nltk_data"
    if _local_nltk_dir.exists():
        nltk.data.path.append(str(_local_nltk_dir))

    _STEMMER: PorterStemmer | None = PorterStemmer()
    _LEMMATIZER: WordNetLemmatizer | None = WordNetLemmatizer()
except ImportError:
    _STEMMER = None
    _LEMMATIZER = None


@functools.lru_cache(maxsize=16384)
def _stem_word(word: str) -> str:
    """Uses NLTK WordNetLemmatizer and PorterStemmer with an algorithmic fallback."""
    w = word.casefold().strip()
    if len(w) <= 2:
        return w
    if _LEMMATIZER is not None:
        try:
            w_noun = _LEMMATIZER.lemmatize(w, pos="n")
            if w_noun != w:
                return w_noun
            w_verb = _LEMMATIZER.lemmatize(w, pos="v")
            if w_verb != w:
                return w_verb
        except Exception:
            pass
    if _STEMMER is not None:
        try:
            return _STEMMER.stem(w)
        except Exception:
            pass

    # Built-in algorithmic fallback when NLTK is not present
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and (w.endswith("ses") or w.endswith("xes") or w.endswith("zes") or w.endswith("ches") or w.endswith("shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing") and not w.endswith("thing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    # Footwear
    "sneaker": ("shoe", "running", "athletic", "trainer", "footwear"),
    "shoe": ("footwear", "sneaker"),
    "boot": ("footwear", "ankle", "hiking"),
    "sandal": ("slide", "open", "toe", "flip", "flop"),
    "heel": ("pump", "stiletto"),
    # Tops & Shirts
    "tee": ("t-shirt", "shirt", "top"),
    "t-shirt": ("tee", "shirt", "top"),
    "tshirt": ("t-shirt", "tee", "shirt", "top"),
    "shirt": ("top", "tee", "blouse", "button"),
    "hoodie": ("sweatshirt", "pullover", "hooded"),
    "sweatshirt": ("hoodie", "pullover", "fleece", "crewneck"),
    "sweater": ("knit", "pullover", "cardigan"),
    # Outerwear
    "jacket": ("coat", "outerwear", "windbreaker", "parka"),
    "coat": ("jacket", "outerwear", "parka", "overcoat"),
    # Bottoms
    "jeans": ("denim", "pants", "trousers"),
    "pants": ("trousers", "slacks", "jeans", "chinos"),
    "shorts": ("trunks", "bermuda"),
    "leggings": ("tights", "pants", "yoga"),
    # Jewelry
    "jewelry": ("ring", "necklace", "earring", "bracelet"),
    "earring": ("stud", "hoop", "dangle"),
    "necklace": ("pendant", "chain", "choker"),
    "bracelet": ("bangle", "cuff", "wristband"),
    # Materials
    "denim": ("jean", "cotton"),
    "leather": ("genuine", "faux", "pu"),
    # Styles & Use cases
    "workout": ("gym", "running", "athletic", "fitness"),
    "running": ("athletic", "workout", "training", "shoe"),
    "hiking": ("outdoor", "trail", "trekking", "boot"),
    "winter": ("warm", "thermal", "fleece"),
    "summer": ("beach", "lightweight", "breathable"),
}


def _expand_terms(terms: Iterable[str], limit: int = 36) -> list[str]:
    """Expands query terms with root-word stem matching and domain synonyms."""
    result: list[str] = []
    seen: set[str] = set()
    for raw_term in terms:
        if raw_term not in seen and raw_term not in STOPWORDS:
            seen.add(raw_term)
            result.append(raw_term)

        stemmed = _stem_word(raw_term)
        if stemmed not in seen and stemmed not in STOPWORDS:
            seen.add(stemmed)
            result.append(stemmed)

        synonyms = SYNONYM_MAP.get(raw_term) or SYNONYM_MAP.get(stemmed) or ()
        for syn in synonyms:
            if syn not in seen and syn not in STOPWORDS:
                seen.add(syn)
                result.append(syn)
                if len(result) >= limit:
                    return result
    return result


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
    price: float | None
    average_rating: float
    rating_number: int


class ProductRetriever:
    """Offline FTS candidate generation followed by deterministic reranking."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.documents: dict[str, ProductDocument] = {}
        self._popular: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE product_fts USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
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
                    price=_numeric_price(product.get("price")),
                    average_rating=float(rating) if isinstance(rating, (int, float)) else 0.0,
                    rating_number=int(rating_number) if isinstance(rating_number, (int, float)) else 0,
                )
                self.documents[parent_asin] = document
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

    def _candidate_scores(self, state: dict[str, Any]) -> dict[str, float]:
        """Generate candidate pool using dynamic Dual-Track Intent Routing (Pillar I)."""
        # Step 1: Extract active category and clue terms from dialogue state
        constraints = active_constraints(state)
        category_terms = _terms(constraints.get("category", []), 24)
        expanded_category_terms = _expand_terms(category_terms, 32)
        records = [
            (attribute, record)
            for attribute, record in self._active_records(state)
            if attribute not in {"category", "budget"}
        ]
        # Sort records chronologically by turn and confidence
        records.sort(key=lambda item: (int(item[1].get("turn", 0)), float(item[1].get("confidence", 0.0))))
        clue_terms = _terms([record["value"] for _, record in records], 48)
        latest_terms = _terms([record["value"] for _, record in records[-2:]], 32)

        scenario = state.get("scenario_detected", "unknown")
        turn = state.get("turn_count", 1)
        routes: list[tuple[list[str], str, int, float]] = []

        # Step 2: Configure search routes based on the detected buyer scenario
        # Track A: Intent override routing when preferences change
        if scenario == "intent_override":
            override_records = [r for _, r in records if r.get("source") == "override"]
            override_terms = _terms([r["value"] for r in override_records], 32)
            if override_terms:
                if category_terms:
                    routes.append((category_terms + override_terms, "AND", 180, 3.5))
                routes.append((override_terms, "AND", 150, 3.0))
                routes.append((override_terms, "OR", 220, 1.8))
            if category_terms:
                routes.append((category_terms, "OR", 200, 0.8))

        # Track B: High-precision routing for buying sessions or later turns
        elif scenario == "buying" or turn >= 3 or len(records) >= 2:
            if category_terms and clue_terms:
                routes.append((category_terms + clue_terms[:28], "AND", 180, 3.2))
            if latest_terms:
                routes.append((latest_terms, "AND", 140, 2.4))
                routes.append((latest_terms, "OR", 200, 1.2))
            if clue_terms:
                routes.append((clue_terms[:20], "AND", 140, 2.0))
            if expanded_category_terms or clue_terms:
                routes.append((expanded_category_terms + clue_terms, "OR", 300, 0.7))
            if expanded_category_terms:
                routes.append((expanded_category_terms, "OR", 200, 0.5))

        # Track C: Exploratory browsing routing for broad early-turn discovery
        else:
            if expanded_category_terms:
                routes.append((expanded_category_terms, "OR", 380, 2.5))
                routes.append((category_terms, "OR", 220, 1.8))
            if clue_terms:
                routes.append((clue_terms, "OR", 250, 1.5))
                if category_terms:
                    routes.append((category_terms + clue_terms[:12], "AND", 120, 1.4))

        # Track D: Exact attribute phrase routes for recent individual constraints
        for _, record in records[-3:]:
            individual = _terms(record["value"], 16)
            if individual:
                routes.append((individual, "AND", 100, 1.25))

        # Step 3: Execute FTS queries and combine candidate scores using Reciprocal Rank Fusion (RRF)
        scores: defaultdict[str, float] = defaultdict(float)
        for terms, operator, limit, weight in routes:
            for rank, parent_asin in enumerate(self._search(terms, limit, operator), start=1):
                scores[parent_asin] += weight / (12.0 + rank)
        return dict(scores)

    @staticmethod
    def _coverage(terms: list[str], text: str) -> float:
        if not terms:
            return 0.0
        text_terms = set(text.split())
        stemmed_text = {_stem_word(t) for t in text_terms}
        matches = sum(1 for term in terms if term in text_terms or _stem_word(term) in stemmed_text)
        return matches / len(terms)

    def _rerank_score(self, document: ProductDocument, state: dict[str, Any], retrieval_score: float) -> float:
        # Step 1: Base score from multi-route candidate retrieval stage
        score = retrieval_score * 80.0

        # Step 2: Attribute and category matching with hierarchical field weighting
        for attribute, record in self._active_records(state):
            value_terms = _terms(record["value"], 40)
            if not value_terms or attribute == "budget":
                continue

            source_weight = 2.0 if record.get("source") == "override" else 1.0
            normalized_phrase = " ".join(value_terms)

            # 2a. Category matching against categories and title fields
            if attribute == "category":
                cat_cov = self._coverage(value_terms, document.categories)
                title_cov = self._coverage(value_terms, document.title)
                combined_cat_cov = max(cat_cov * 1.0, title_cov * 0.9)
                if combined_cat_cov < 1.0:
                    expanded = _expand_terms(value_terms, 24)
                    expanded_cat_cov = max(
                        self._coverage(expanded, document.categories),
                        self._coverage(expanded, document.title) * 0.9,
                    )
                    combined_cat_cov = max(combined_cat_cov, expanded_cat_cov * 0.85)

                score += 10.0 * combined_cat_cov
                if normalized_phrase in document.categories:
                    score += 4.0
                elif normalized_phrase in document.title:
                    score += 3.0
                continue

            # 2b. Attribute matching with hierarchical field weighting (title > features/details > description)
            title_cov = self._coverage(value_terms, document.title)
            feat_cov = self._coverage(value_terms, f"{document.features} {document.details}")
            desc_cov = self._coverage(value_terms, document.description)

            if max(title_cov, feat_cov, desc_cov) < 1.0:
                expanded = _expand_terms(value_terms, 24)
                title_cov = max(title_cov, self._coverage(expanded, document.title) * 0.9)
                feat_cov = max(feat_cov, self._coverage(expanded, f"{document.features} {document.details}") * 0.85)
                desc_cov = max(desc_cov, self._coverage(expanded, document.description) * 0.75)

            field_weighted_coverage = (2.0 * title_cov + 1.5 * feat_cov + 0.8 * desc_cov) / 4.3
            score += source_weight * 14.0 * field_weighted_coverage

            # 2c. Consecutive exact phrase bonus
            if normalized_phrase:
                if normalized_phrase in document.title:
                    score += source_weight * 6.0
                elif normalized_phrase in document.features or normalized_phrase in document.details:
                    score += source_weight * 4.0
                elif normalized_phrase in document.description:
                    score += source_weight * 1.5

            # 2d. Key discriminating attribute bonus (material, color, brand, style)
            if attribute in {"material", "color", "brand", "style"} and max(title_cov, feat_cov) >= 1.0:
                score += source_weight * 4.0

        # Step 3: Heavy penalty for user-rejected / negated terms (e.g. "no wool")
        for excluded in state.get("excluded_terms", []):
            if re.search(rf"\b{re.escape(str(excluded).casefold())}\b", document.combined):
                score -= 80.0

        # Step 4: Calibrated budget filtering and price proximity scoring
        for record in state["constraints"].get("budget", []):
            if record.get("status") != "active" or "numeric_value" not in record:
                continue
            target = float(record["numeric_value"])
            if document.price is None:
                score += 1.5
                continue

            if record.get("price_mode") == "maximum":
                if document.price <= target:
                    # Under budget bonus + savings incentive
                    savings_ratio = max(0.0, (target - document.price) / max(1.0, target))
                    score += 8.0 + 2.0 * savings_ratio
                elif document.price <= target * 1.10:
                    # Mild tolerance margin (10% overage)
                    overage_ratio = (document.price - target) / (target * 0.10)
                    score -= 5.0 * overage_ratio
                else:
                    # Steep penalty for major budget violations
                    score -= 35.0 + min(40.0, (document.price - target) * 1.5)
            else:
                # Gaussian proximity curve for 'around' mode
                sigma = max(5.0, target * 0.25)
                diff = document.price - target
                score += 12.0 * math.exp(-0.5 * (diff / sigma) ** 2)

        # Step 5: Personalization tags and review quality priors
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
        """Safety fallback to guarantee returning 10 catalog-valid recommendations."""
        categories = active_constraints(state).get("category", [])
        category_terms = _terms(categories, 24)
        expanded_terms = _expand_terms(category_terms, 32)
        result: list[str] = []
        
        # Fallback Level 1: Broader category matches (with synonyms)
        if expanded_terms:
            for parent_asin in self._search(expanded_terms, max(needed * 8, 80), "OR"):
                if parent_asin not in seen:
                    seen.add(parent_asin)
                    result.append(parent_asin)
                    if len(result) >= needed:
                        return result
                        
        # Fallback Level 2: Top popular/high-rated catalog items
        for parent_asin in self._popular:
            if parent_asin not in seen:
                seen.add(parent_asin)
                result.append(parent_asin)
                if len(result) >= needed:
                    break
        return result

    def recommend(self, state: dict[str, Any], top_k: int = 10) -> list[dict[str, str]]:
        limit = max(1, min(int(top_k), 10))
        # Multi-route candidate generation
        retrieval_scores = self._candidate_scores(state)
        # Reranking based on context
        ranked = sorted(
            retrieval_scores,
            key=lambda asin: (self._rerank_score(self.documents[asin], state, retrieval_scores[asin]), asin),
            reverse=True,
        )
        selected = ranked[:limit]
        seen = set(selected)
        if len(selected) < limit:
            # Fallback if < 10 items
            selected.extend(self._category_fallback(state, limit - len(selected), seen))
        return [{"parent_asin": parent_asin} for parent_asin in selected]
