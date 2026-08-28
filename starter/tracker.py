from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)

MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "denim",
    "linen",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
    "navy",
    "beige",
    "gold",
    "silver",
)
STYLE_WORDS = (
    "style",
    "fit",
    "fitted",
    "relaxed",
    "oversized",
    "slim",
    "casual",
    "formal",
    "sleeve",
    "sleeveless",
    "neck",
    "crewneck",
    "v-neck",
    "vintage",
)
USE_CASE_WORDS = (
    "hiking",
    "running",
    "walking",
    "gym",
    "workout",
    "winter",
    "outdoor",
    "work",
    "wedding",
    "travel",
)
CATEGORY_WORDS = (
    "t-shirt",
    "t-shirts",
    "shirt",
    "shirts",
    "dress",
    "dresses",
    "shoe",
    "shoes",
    "sneaker",
    "sneakers",
    "boot",
    "boots",
    "jacket",
    "jackets",
    "coat",
    "coats",
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "skirts",
    "sandal",
    "sandals",
    "hat",
    "hats",
    "watch",
    "watches",
    "earring",
    "earrings",
    "necklace",
    "necklaces",
    "bracelet",
    "bracelets",
    "costume",
    "costumes",
)

_LOOKING_FOR_RE = re.compile(
    r"\blooking\s+for\s+(?P<value>.+?)(?=\.\s|,\s*(?:but\b|and\b)|$)",
    re.IGNORECASE,
)
_KEY_REQUIREMENT_RE = re.compile(r"\bkey requirement is:\s*(?P<value>.+)$", re.IGNORECASE)
_MATTERS_RE = re.compile(r"\bwhat matters is:\s*(?P<value>.+)$", re.IGNORECASE)
_OVERRIDE_RE = re.compile(r"\bwhat i need is:\s*(?P<value>.+)$", re.IGNORECASE)
_NO_PREFERENCE_RE = re.compile(
    r"\b(?:don't|do not) have (?:an additional |a )?preference for\s+(?P<attribute>[a-z_]+)",
    re.IGNORECASE,
)
_DOLLAR_PRICE_RE = re.compile(
    r"(?P<mode>under|below|less than|up to|maximum|max|around|about|near)?\s*\$\s*(?P<amount>\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_PLAIN_PRICE_RE = re.compile(
    r"\b(?P<mode>under|below|less than|up to|maximum|max|around|about|near)\s+"
    r"(?P<amount>\d+(?:\.\d{1,2})?)\b",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:-")


def _normalized(value: str) -> str:
    return _clean(value).casefold()


def _price_parts(value: str) -> tuple[str, float, str] | None:
    match = _DOLLAR_PRICE_RE.search(value) or _PLAIN_PRICE_RE.search(value)
    if not match:
        return None
    mode = (match.group("mode") or "").casefold()
    price_mode = "maximum" if mode in {
        "under", "below", "less than", "up to", "maximum", "max"
    } else "around"
    return price_mode, float(match.group("amount")), match.group(0)


def classify_constraint(value: str) -> str:
    """Classify a simulator constraint using rules aligned with the evaluator."""

    lowered = value.casefold()
    if "budget" in lowered or _price_parts(lowered):
        return "budget"
    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in MATERIALS):
        return "material"
    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in COLORS):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in STYLE_WORDS):
        return "style"
    if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in USE_CASE_WORDS):
        return "use_case"
    if lowered.startswith("brand:") or lowered.startswith("manufacturer:"):
        return "brand"
    return "feature"


def new_buyer_state(session_id: str, user_profile: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "constraints": {attribute: [] for attribute in ATTRIBUTES},
        "excluded_terms": [],
        "no_preference": [],
        "user_profile": deepcopy(user_profile or {}),
        "turn_count": 0,
        "scenario_detected": "unknown",
        "asked_attributes": [],
        "last_asked_attribute": None,
        "raw_history": [],
    }


def active_constraints(state: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for attribute, records in state["constraints"].items():
        values = [record["value"] for record in records if record.get("status") == "active"]
        if values:
            result[attribute] = values
    return result


def _add_constraint(
    state: dict[str, Any],
    attribute: str,
    value: str,
    turn: int,
    source: str,
    confidence: float = 1.0,
) -> None:
    value = _clean(value)
    if not value or attribute not in state["constraints"]:
        return
    normalized = _normalized(value)
    records = state["constraints"][attribute]
    for record in records:
        if record["normalized"] == normalized and record.get("status") == "active":
            record["last_seen_turn"] = turn
            record["confidence"] = max(float(record.get("confidence", 0.0)), confidence)
            return
    record: dict[str, Any] = {
        "value": value,
        "normalized": normalized,
        "turn": turn,
        "last_seen_turn": turn,
        "source": source,
        "confidence": confidence,
        "status": "active",
    }
    price = _price_parts(value)
    if attribute == "budget" and price:
        record["price_mode"], record["numeric_value"], _ = price
    records.append(record)


def _supersede_initial_preference(state: dict[str, Any], turn: int) -> None:
    for attribute, records in state["constraints"].items():
        if attribute == "category":
            continue
        for record in records:
            if record.get("status") == "active" and record.get("source") == "initial_preference":
                record["status"] = "superseded"
                record["superseded_turn"] = turn


def _extract_negations(state: dict[str, Any], message: str) -> None:
    lowered = message.casefold()
    for term in (*MATERIALS, *COLORS, *STYLE_WORDS):
        pattern = (
            rf"\b(?:avoid|hate|without|not|no|don't want|do not want|"
            rf"anything except|except)\s+(?:any\s+)?{re.escape(term)}\b"
        )
        if not re.search(pattern, lowered):
            continue
        if term not in state["excluded_terms"]:
            state["excluded_terms"].append(term)
        for records in state["constraints"].values():
            for record in records:
                if record.get("status") == "active" and re.search(
                    rf"\b{re.escape(term)}\b", record["normalized"]
                ):
                    record["status"] = "rejected"


def _extract_free_text(state: dict[str, Any], message: str, turn: int) -> None:
    """Fallback extraction for natural messages outside the deterministic simulator."""

    lowered = message.casefold()
    for material in MATERIALS:
        if material not in state["excluded_terms"] and re.search(rf"\b{re.escape(material)}\b", lowered):
            _add_constraint(state, "material", material, turn, "keyword", 0.75)
    for color in COLORS:
        if color not in state["excluded_terms"] and re.search(rf"\b{re.escape(color)}\b", lowered):
            canonical = "gray" if color == "grey" else color
            _add_constraint(state, "color", canonical, turn, "keyword", 0.75)
    for category in CATEGORY_WORDS:
        if re.search(rf"\b{re.escape(category)}\b", lowered):
            _add_constraint(state, "category", category, turn, "keyword", 0.65)
    price = _price_parts(message)
    if price:
        _add_constraint(state, "budget", price[2], turn, "keyword", 0.8)


def update_buyer_state(state: dict[str, Any], user_message: str, turn: int) -> dict[str, Any]:
    message = str(user_message or "")
    lowered = message.casefold()
    state["turn_count"] = turn

    no_preference = _NO_PREFERENCE_RE.search(message)
    if no_preference:
        attribute = no_preference.group("attribute").casefold()
        if attribute not in state["no_preference"]:
            state["no_preference"].append(attribute)
        state["scenario_detected"] = "boundary"

    override = _OVERRIDE_RE.search(message)
    if override or ("actually" in lowered and "ignore" in lowered):
        state["scenario_detected"] = "intent_override"
        _supersede_initial_preference(state, turn)

    looking_for = _LOOKING_FOR_RE.search(message)
    if looking_for:
        _add_constraint(state, "category", looking_for.group("value"), turn, "initial_category")

    structured_values: list[tuple[str, str]] = []
    if override:
        structured_values.append((override.group("value"), "override"))
    else:
        key_requirement = _KEY_REQUIREMENT_RE.search(message)
        matters = _MATTERS_RE.search(message)
        if key_requirement:
            state["scenario_detected"] = "buying"
            structured_values.append((key_requirement.group("value"), "key_requirement"))
        elif matters:
            for value in matters.group("value").split(";"):
                structured_values.append((value, "clarification"))
        elif looking_for and "." in message and "still exploring" not in lowered:
            remainder = message[looking_for.end():].lstrip(". ")
            if remainder:
                structured_values.append((remainder, "initial_preference"))

    if "still exploring" in lowered and state["scenario_detected"] == "unknown":
        state["scenario_detected"] = "browsing"

    for value, source in structured_values:
        cleaned = _clean(value)
        if cleaned:
            _add_constraint(state, classify_constraint(cleaned), cleaned, turn, source)

    _extract_negations(state, message)
    if not structured_values:
        _extract_free_text(state, message, turn)

    state["raw_history"].append({"turn": turn, "user": message})
    return state


def choose_next_question(state: dict[str, Any]) -> tuple[str | None, str]:
    """Choose a high-yield clarification while respecting boundary responses."""

    asked = state["asked_attributes"]
    unavailable = set(state["no_preference"])
    broad_asks = sum(attribute == "other" for attribute in asked)
    fact_count = sum(len(values) for values in active_constraints(state).values())

    # A Boundary session consumes its one non-answer on the first broad ask.
    # Asking for concrete product features next avoids walking a long fixed list.
    if state.get("scenario_detected") == "boundary" and "feature" not in asked:
        return "feature", "Which concrete product features should I prioritize?"

    # In the released simulator, `other` returns up to two remaining intent facts.
    # Two early broad questions have the best expected information gain for vague sessions.
    if "other" not in unavailable and broad_asks < 2 and (state["turn_count"] <= 2 or fact_count < 3):
        return "other", "What details or features matter most to you?"

    constraints = active_constraints(state)
    priorities = ("material", "style", "color", "size", "use_case", "budget", "brand", "feature")
    for attribute in priorities:
        if attribute in unavailable or attribute in asked or constraints.get(attribute):
            continue
        label = attribute.replace("_", " ")
        return attribute, f"Do you have a preference for {label}?"

    return None, "I’ll refine the results using what you’ve told me."


class BuyerStateTracker:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def reset(self, session_id: str, user_profile: dict[str, Any] | None) -> dict[str, Any]:
        state = new_buyer_state(session_id, user_profile)
        self._states[session_id] = state
        return state

    def update(self, session_id: str, user_message: str, turn: int) -> dict[str, Any]:
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")
        return update_buyer_state(self._states[session_id], user_message, turn)

    def record_question(self, session_id: str, attribute: str | None, message: str) -> None:
        state = self._states[session_id]
        state["last_asked_attribute"] = attribute
        if attribute:
            state["asked_attributes"].append(attribute)
        if state["raw_history"]:
            state["raw_history"][-1]["agent_ask"] = attribute
            state["raw_history"][-1]["agent_message"] = message

    def get(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._states:
            raise RuntimeError("unknown session")
        return self._states[session_id]
