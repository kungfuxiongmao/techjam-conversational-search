from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.retriever import ProductRetriever
from starter.tracker import new_buyer_state, update_buyer_state


PRODUCTS = [
    {
        "parent_asin": "TARGET",
        "title": "Black Cotton Crew Neck T-Shirt",
        "features": ["100% cotton", "crew neck", "machine washable"],
        "description": ["comfortable casual shirt"],
        "price": 24.99,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts", "T-Shirts"],
        "details": {"Color": "Black", "Department": "mens"},
        "average_rating": 4.7,
        "rating_number": 500,
        "store": "Example",
    },
    {
        "parent_asin": "WHITE",
        "title": "White Cotton T-Shirt",
        "features": ["100% cotton", "v-neck"],
        "description": [],
        "price": 25.0,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts", "T-Shirts"],
        "details": {"Color": "White"},
        "average_rating": 4.8,
        "rating_number": 1000,
        "store": "Example",
    },
    {
        "parent_asin": "WOOL",
        "title": "Black Wool Crew Neck Shirt",
        "features": ["warm wool"],
        "description": [],
        "price": 70.0,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts"],
        "details": {"Color": "Black"},
        "average_rating": 5.0,
        "rating_number": 2000,
        "store": "Example",
    },
    {
        "parent_asin": "SHOE",
        "title": "Black Running Shoe",
        "features": ["lightweight mesh"],
        "description": [],
        "price": 55.0,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Shoes", "Running"],
        "details": {},
        "average_rating": 4.9,
        "rating_number": 800,
        "store": "Example",
    },
]


class RetrieverAndAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.catalog_path = Path(cls.temp_directory.name) / "catalog.jsonl"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in PRODUCTS), encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def test_retriever_ranks_cumulative_exact_matches_first(self) -> None:
        retriever = ProductRetriever(self.catalog_path)
        state = new_buyer_state("s", {})
        update_buyer_state(
            state,
            "I'm looking for Men Shirts T-Shirts. A key requirement is: 100% cotton.",
            1,
        )
        update_buyer_state(state, "For that, what matters is: color: black; crew neck.", 2)
        ranked = retriever.recommend(state, 4)
        self.assertEqual(ranked[0]["parent_asin"], "TARGET")
        self.assertEqual(len({item["parent_asin"] for item in ranked}), 4)

    def test_retriever_honors_exclusion_penalty(self) -> None:
        retriever = ProductRetriever(self.catalog_path)
        state = new_buyer_state("s", {})
        update_buyer_state(state, "I want a black shirt but no wool", 1)
        ranked = retriever.recommend(state, 4)
        self.assertNotEqual(ranked[0]["parent_asin"], "WOOL")

    def test_agent_returns_valid_contract_and_tracks_question(self) -> None:
        agent = Agent(self.catalog_path)
        agent.reset("session", {"preference_tags": ["comfort"]})
        response = agent.respond(
            "session", "I'm looking for Men Shirts T-Shirts, but I'm still exploring.", 1, 10
        )
        self.assertIsInstance(response["message"], str)
        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(len(response["recommendations"]), 4)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(agent.tracker.get("session")["asked_attributes"], ["other"])


if __name__ == "__main__":
    unittest.main()
