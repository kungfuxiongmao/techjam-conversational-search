from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Canonical concepts deliberately stay conservative: entries are interchangeable
# in a clothing search, rather than merely associated words that could add noise.
SINGLE_TOKEN_CONCEPTS = {
    "athleisure": "activewear",
    "activewear": "activewear",
    "breathability": "breathable",
    "cushion": "cushioned",
    "cushioning": "cushioned",
    "grey": "gray",
    "hoodies": "hoodie",
    "backpacks": "backpack",
    "bathrobe": "robe",
    "bathrobes": "robe",
    "daypack": "backpack",
    "daypacks": "backpack",
    "jog": "running",
    "jogging": "running",
    "kicks": "sneaker",
    "lightweight": "lightweight",
    "padded": "cushioned",
    "pullover": "hoodie",
    "purse": "handbag",
    "purses": "handbag",
    "loafers": "loafer",
    "parkas": "parka",
    "rainproof": "waterproof",
    "sneakers": "sneaker",
    "robes": "robe",
    "shackets": "shacket",
    "sweatshirt": "hoodie",
    "sweatshirts": "hoodie",
    "tee": "tshirt",
    "tees": "tshirt",
    "tshirt": "tshirt",
    "trainer": "sneaker",
    "trainers": "sneaker",
    "trouser": "pants",
    "trousers": "pants",
    "undershirts": "undershirt",
    "vests": "vest",
    "ultralight": "lightweight",
    "ventilated": "breathable",
    "waterproofing": "waterproof",
}

PHRASE_CONCEPTS = {
    ("athletic", "shoe"): "sneaker",
    ("athletic", "shoes"): "sneaker",
    ("crew", "neck"): "crewneck",
    ("dress", "shoe"): "formalshoe",
    ("dress", "shoes"): "formalshoe",
    ("gym", "wear"): "activewear",
    ("light", "weight"): "lightweight",
    ("rain", "resistant"): "waterproof",
    ("running", "footwear"): "sneaker",
    ("running", "shoe"): "sneaker",
    ("running", "shoes"): "sneaker",
    ("slip", "on"): "slipon",
    ("t", "shirt"): "tshirt",
    ("t", "shirts"): "tshirt",
    ("tee", "shirt"): "tshirt",
    ("tee", "shirts"): "tshirt",
    ("v", "neck"): "vneck",
    ("water", "resistant"): "waterproof",
    ("work", "out"): "workout",
}

# Tokens sent to FTS. They complement (rather than replace) the original query,
# which keeps exact product wording highly ranked.
CONCEPT_EXPANSIONS = {
    "activewear": ("activewear", "athleisure", "gym", "fitness", "workout"),
    "backpack": ("backpack", "backpacks", "daypack", "daypacks", "sling", "crossbody", "bag"),
    "breathable": ("breathable", "ventilated", "airflow", "mesh"),
    "crewneck": ("crew", "neck", "crewneck"),
    "cushioned": ("cushion", "cushioning", "cushioned", "padded"),
    "formalshoe": ("dress", "formal", "shoe", "shoes"),
    "gray": ("gray", "grey"),
    "handbag": ("handbag", "handbags", "purse", "purses"),
    "hoodie": ("hoodie", "hoodies", "sweatshirt", "sweatshirts", "pullover"),
    "loafer": ("loafer", "loafers", "slipon", "shoe"),
    "parka": ("parka", "parkas", "jacket", "coat"),
    "lightweight": ("lightweight", "light", "ultralight"),
    "pants": ("pants", "trouser", "trousers", "slacks"),
    "running": ("running", "jog", "jogging"),
    "robe": ("robe", "robes", "bathrobe", "bathrobes", "loungewear"),
    "shacket": ("shacket", "shackets", "shirt", "jacket"),
    "sneaker": ("sneaker", "sneakers", "trainer", "trainers", "kicks", "shoe", "shoes"),
    "tshirt": ("tshirt", "tee", "tees", "shirt", "shirts"),
    "undershirt": ("undershirt", "undershirts", "tshirt", "underwear"),
    "vneck": ("vneck", "neck"),
    "waterproof": ("waterproof", "rainproof", "resistant", "rain"),
    "vest": ("vest", "vests", "jacket"),
    "workout": ("workout", "gym", "fitness", "exercise", "training"),
}


def raw_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        text = " ".join(f"{key} {item}" for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return TOKEN_RE.findall(text.casefold())


def canonical_tokens(value: object) -> list[str]:
    """Return synonym-aware tokens, folding common two-word fashion phrases."""

    source = raw_tokens(value)
    result: list[str] = []
    index = 0
    while index < len(source):
        phrase = tuple(source[index:index + 2])
        concept = PHRASE_CONCEPTS.get(phrase)
        if concept:
            result.append(concept)
            index += 2
            continue
        token = source[index]
        concept = SINGLE_TOKEN_CONCEPTS.get(token, token)
        result.append(concept)
        index += 1
    return result


def expanded_search_terms(
    value: object,
    *,
    stopwords: Iterable[str] = (),
    limit: int = 64,
) -> list[str]:
    """Expand a query into raw terms, canonical concepts, and catalog variants."""

    stopped = set(stopwords)
    result: list[str] = []
    seen: set[str] = set()
    for token in [*raw_tokens(value), *canonical_tokens(value)]:
        variants = CONCEPT_EXPANSIONS.get(token, (token,))
        for variant in variants:
            if len(variant) <= 1 or variant in stopped or variant in seen:
                continue
            seen.add(variant)
            result.append(variant)
            if len(result) >= limit:
                return result
    return result


def vocabulary_categories() -> tuple[str, ...]:
    """User-facing category aliases recognized by the free-text tracker."""

    return (
        "activewear", "athleisure", "backpack", "backpacks", "bathrobe", "bathrobes",
        "daypack", "daypacks", "hoodie", "hoodies", "kicks", "loafer", "loafers",
        "parka", "parkas", "pullover", "purse", "purses", "robe", "robes", "shacket",
        "shackets", "sling", "sneaker", "sneakers", "sweatshirt", "sweatshirts", "tee",
        "tees", "trainer", "trainers", "trouser", "trousers", "tshirt", "undershirt",
        "undershirts", "vest", "vests",
    )
