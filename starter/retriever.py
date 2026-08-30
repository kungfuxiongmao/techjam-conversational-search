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
HYPHEN_PREFIX_RE = re.compile(r"\b([a-z0-9]{1,4})-([a-z0-9]+)\b", re.IGNORECASE)
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


def _normalize_compounds(text: str) -> str:
    """Normalizes hyphenated compounds (e.g. 't-shirt' -> 'tshirt t-shirt shirt', 'v-neck' -> 'vneck v-neck neck')."""
    def repl(m: re.Match) -> str:
        p1, p2 = m.group(1), m.group(2)
        return f"{p1}{p2} {p1} {p2}"
    return HYPHEN_PREFIX_RE.sub(repl, text)


def _normal_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(_normalize_compounds(_flatten(value).casefold())))


def _terms(value: object, limit: int = 48) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    cleaned = _normalize_compounds(_flatten(value).casefold())
    for token in TOKEN_RE.findall(cleaned):
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
    from nltk.corpus import wordnet as wn
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    _local_nltk_dir = Path(__file__).resolve().parent.parent / ".venv" / "nltk_data"
    if _local_nltk_dir.exists():
        nltk.data.path.append(str(_local_nltk_dir))

    _STEMMER: PorterStemmer | None = PorterStemmer()
    _LEMMATIZER: WordNetLemmatizer | None = WordNetLemmatizer()
    _WN = wn
except ImportError:
    _STEMMER = None
    _LEMMATIZER = None
    _WN = None


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


class WordNetSynonymProvider:
    """Open-vocabulary synonym resolution using NLTK Princeton WordNet.
    
    Traverses WordNet's taxonomic synsets for noun artifact, substance, and attribute
    concepts to provide dynamic, domain-general synonym expansion without requiring
    hardcoded lexical lookup tables.
    """

    def __init__(self, wn_corpus: Any = _WN, custom_overlays: dict[str, tuple[str, ...]] | None = None) -> None:
        self._wn = wn_corpus
        self._overlays = custom_overlays or {}

    @functools.lru_cache(maxsize=8192)
    def synonyms(self, word: str, limit: int = 4) -> tuple[str, ...]:
        syns: list[str] = []
        seen: set[str] = {word}

        # 1. Dynamic open-vocabulary WordNet synsets
        if self._wn is not None:
            try:
                for syn in self._wn.synsets(word, pos="n"):
                    lexname = syn.lexname()
                    if lexname not in {"noun.artifact", "noun.substance", "noun.attribute"}:
                        continue
                    for lemma in syn.lemmas():
                        clean = lemma.name().replace("_", "-").casefold()
                        for part in clean.split("-"):
                            if part not in seen and len(part) > 2 and part not in STOPWORDS:
                                seen.add(part)
                                syns.append(part)
                                if len(syns) >= limit:
                                    return tuple(syns)
            except Exception:
                pass

        # 2. Optional domain overlays (if configured)
        for overlay in self._overlays.get(word, ()):
            if overlay not in seen and overlay not in STOPWORDS:
                seen.add(overlay)
                syns.append(overlay)
                if len(syns) >= limit:
                    break

        return tuple(syns)


_DEFAULT_SYNONYM_PROVIDER = WordNetSynonymProvider()


def _expand_terms(
    terms: Iterable[str],
    limit: int = 36,
    synonym_provider: WordNetSynonymProvider | None = None,
) -> list[str]:
    """Expands query terms with root-word stem matching and dynamic WordNet synsets."""
    provider = synonym_provider or _DEFAULT_SYNONYM_PROVIDER
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

        for syn in provider.synonyms(raw_term):
            if syn not in seen and syn not in STOPWORDS:
                seen.add(syn)
                result.append(syn)
                if len(result) >= limit:
                    return result
    return result


@dataclass(frozen=True)
class RetrieverConfig:
    """Hyperparameters governing candidate generation and reranking utility curves.
    
    Attributes:
        base_retrieval_multiplier: Scaling factor for candidate RRF scores into reranker base.
        rrf_k: Reciprocal Rank Fusion rank dampening constant (default: 12.0).
        category_match_scale: Maximum points awarded for full category field coverage.
        attribute_match_scale: Maximum base scale for field-weighted attribute coverage.
        phrase_bonus_scale: Multiplier for exact consecutive multi-word phrase matches.
        hard_budget_tolerance: Fractional overage allowed before steep budget penalties (0.10 = 10%).
        budget_penalty_slope: Linear slope applied to overage beyond the tolerance margin.
        negation_penalty: Fixed score penalty applied to items containing user-rejected terms.
    """
    base_retrieval_multiplier: float = 40.0
    rrf_k: float = 12.0
    category_match_scale: float = 12.0
    attribute_match_scale: float = 16.0
    phrase_bonus_scale: float = 3.5
    hard_budget_tolerance: float = 0.10
    budget_penalty_slope: float = 1.5
    negation_penalty: float = 80.0


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
    """Offline FTS candidate generation followed by deterministic reranking.
    
    Calculates dynamic BM25F field signal densities and vocabulary document frequencies (IDF)
    at catalog load time to replace hardcoded field and term importance weights with mathematically
    grounded information-theoretic values:
    
    1. Field Saliency (BM25F):
       W_field = 3.0 * (1 / sqrt(avg_len(field))) / sum_f (1 / sqrt(avg_len(f)))
       
    2. Term Specificity (IDF):
       IDF(t) = ln(1.0 + N / (DF(t) + 1))
    """

    def __init__(
        self,
        catalog_path: str | Path,
        config: RetrieverConfig | None = None,
        synonym_provider: WordNetSynonymProvider | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config or RetrieverConfig()
        self.synonym_provider = synonym_provider or _DEFAULT_SYNONYM_PROVIDER
        self.connection = sqlite3.connect(":memory:")
        self.documents: dict[str, ProductDocument] = {}
        self._popular: list[str] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.field_weights: dict[str, float] = {"title": 1.5, "features": 0.8, "description": 0.7}
        self._build_index()

    def _expand(self, terms: Iterable[str], limit: int = 36) -> list[str]:
        return _expand_terms(terms, limit=limit, synonym_provider=self.synonym_provider)

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE product_fts USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        doc_count = 0
        total_title_tokens = 0
        total_feat_tokens = 0
        total_desc_tokens = 0

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

                doc_count += 1
                t_terms = TOKEN_RE.findall(title.casefold())
                f_terms = TOKEN_RE.findall(features.casefold())
                d_terms = TOKEN_RE.findall(description.casefold())
                total_title_tokens += len(t_terms)
                total_feat_tokens += len(f_terms)
                total_desc_tokens += len(d_terms)

                # Track document frequency for vocabulary terms
                seen_in_doc = set(t_terms) | set(f_terms)
                for term in seen_in_doc:
                    if term not in STOPWORDS and len(term) > 1:
                        self.doc_freq[term] += 1

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

        # Compute dynamic field weights via BM25F field-length normalization:
        # W_field = 3.0 * (1 / sqrt(avg_len(field))) / sum_f (1 / sqrt(avg_len(f)))
        if doc_count > 0:
            avg_title = max(6.0, total_title_tokens / doc_count)
            avg_feat = max(18.0, total_feat_tokens / doc_count)
            avg_desc = max(24.0, total_desc_tokens / doc_count)

            raw_t = 1.0 / math.sqrt(avg_title)
            raw_f = 1.0 / math.sqrt(avg_feat)
            raw_d = 1.0 / math.sqrt(avg_desc)
            total_raw = raw_t + raw_f + raw_d

            self.field_weights = {
                "title": (raw_t / total_raw) * 3.0,
                "features": (raw_f / total_raw) * 3.0,
                "description": (raw_d / total_raw) * 3.0,
            }

        self._popular = sorted(
            self.documents,
            key=lambda asin: (
                self.documents[asin].average_rating * math.log1p(self.documents[asin].rating_number),
                self.documents[asin].rating_number,
                asin,
            ),
            reverse=True,
        )

    def _idf(self, term: str) -> float:
        """Calculates smoothed Inverse Document Frequency for term specificity:
        
        IDF(t) = ln(1.0 + N / (DF(t) + 1))
        """
        doc_count = max(1, len(self.documents))
        df = self.doc_freq.get(term, 1)
        return math.log(1.0 + doc_count / (df + 1))

    def _constraint_salience(self, record: dict[str, Any], terms: list[str]) -> float:
        """Dynamically computes constraint salience from term IDF, turn confidence, and override source:
        
        Salience = Override_Multiplier * Confidence * Specificity(IDF)
        """
        if not terms:
            return 1.0
        idfs = [self._idf(t) for t in terms]
        avg_idf = sum(idfs) / len(idfs)
        specificity = max(0.7, min(1.8, avg_idf / 2.5))
        confidence = float(record.get("confidence", 1.0))
        override_mult = 2.0 if record.get("source") == "override" else 1.0
        return override_mult * confidence * specificity

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
        expanded_category_terms = self._expand(category_terms, 32)
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
                scores[parent_asin] += weight / (self.config.rrf_k + rank)
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
        score = retrieval_score * self.config.base_retrieval_multiplier

        # Step 2: Attribute and category matching with hierarchical field weighting
        for attribute, record in self._active_records(state):
            value_terms = _terms(record["value"], 40)
            if not value_terms or attribute == "budget":
                continue

            salience = self._constraint_salience(record, value_terms)
            normalized_phrase = " ".join(value_terms)

            # 2a. Category matching against categories and title fields
            if attribute == "category":
                cat_cov = self._coverage(value_terms, document.categories)
                title_cov = self._coverage(value_terms, document.title)
                combined_cat_cov = max(cat_cov * 1.0, title_cov * 0.9)
                if combined_cat_cov < 1.0:
                    expanded = self._expand(value_terms, 24)
                    expanded_cat_cov = max(
                        self._coverage(expanded, document.categories),
                        self._coverage(expanded, document.title) * 0.9,
                    )
                    combined_cat_cov = max(combined_cat_cov, expanded_cat_cov * 0.85)

                score += self.config.category_match_scale * combined_cat_cov
                if normalized_phrase in document.categories:
                    score += 4.0
                elif normalized_phrase in document.title:
                    score += 3.0
                continue

            # 2b. Attribute matching with dynamic BM25F field weighting (Title > Features > Description)
            title_cov = self._coverage(value_terms, document.title)
            feat_cov = self._coverage(value_terms, f"{document.features} {document.details}")
            desc_cov = self._coverage(value_terms, document.description)

            if max(title_cov, feat_cov, desc_cov) < 1.0:
                expanded = self._expand(value_terms, 24)
                title_cov = max(title_cov, self._coverage(expanded, document.title) * 0.9)
                feat_cov = max(feat_cov, self._coverage(expanded, f"{document.features} {document.details}") * 0.85)
                desc_cov = max(desc_cov, self._coverage(expanded, document.description) * 0.75)

            # Normalized by the sum of dynamic field weights (3.0)
            field_weighted_coverage = (
                self.field_weights["title"] * title_cov +
                self.field_weights["features"] * feat_cov +
                self.field_weights["description"] * desc_cov
            ) / 3.0
            score += salience * self.config.attribute_match_scale * field_weighted_coverage

            # 2c. Consecutive exact phrase bonus
            if normalized_phrase:
                if normalized_phrase in document.title:
                    score += salience * self.config.phrase_bonus_scale * self.field_weights["title"]
                elif normalized_phrase in document.features or normalized_phrase in document.details:
                    score += salience * self.config.phrase_bonus_scale * self.field_weights["features"]
                elif normalized_phrase in document.description:
                    score += salience * self.config.phrase_bonus_scale * self.field_weights["description"]

            # 2d. Information-theoretic exact coverage bonus (self-calibrating via IDF)
            if max(title_cov, feat_cov) >= 1.0:
                max_idf = max((self._idf(t) for t in value_terms), default=2.5)
                specificity_bonus = min(4.0, max(1.5, max_idf * 0.8))
                score += salience * specificity_bonus

        # Step 3: Heavy penalty for user-rejected / negated terms (e.g. "no wool")
        for excluded in state.get("excluded_terms", []):
            if re.search(rf"\b{re.escape(str(excluded).casefold())}\b", document.combined):
                score -= self.config.negation_penalty

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
                elif document.price <= target * (1.0 + self.config.hard_budget_tolerance):
                    # Mild tolerance margin
                    overage_ratio = (document.price - target) / (target * self.config.hard_budget_tolerance)
                    score -= 5.0 * overage_ratio
                else:
                    # Steep penalty for major budget violations
                    score -= 35.0 + min(40.0, (document.price - target) * self.config.budget_penalty_slope)
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
        expanded_terms = self._expand(category_terms, 32)
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
