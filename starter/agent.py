from __future__ import annotations

from pathlib import Path

from starter.retriever import ProductRetriever
from starter.tracker import BuyerStateTracker, choose_next_question


class Agent:
    """Cotton Sheep (棉羊): Stateful, deterministic conversational search agent connecting the
    Dialogue State Tracker and Product Retriever & Ranker.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.tracker = BuyerStateTracker()
        self.retriever = ProductRetriever(self.catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.tracker.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self.tracker.update(session_id, user_message, turn)
        recommendations = self.retriever.recommend(state, top_k)
        ask_attribute, question = choose_next_question(state)
        self.tracker.record_question(session_id, ask_attribute, question)
        return {
            "message": question,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
