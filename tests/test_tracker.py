from __future__ import annotations

import unittest

from starter.tracker import (
    BuyerStateTracker,
    active_constraints,
    choose_next_question,
    new_buyer_state,
    update_buyer_state,
)


class BuyerStateTrackerTest(unittest.TestCase):
    def test_accumulates_structured_constraints_across_turns(self) -> None:
        state = new_buyer_state("s", {"preference_tags": ["comfort"]})
        update_buyer_state(
            state,
            "I'm looking for Men Clothing Shirts T-Shirts. A key requirement is: 100% cotton.",
            1,
        )
        update_buyer_state(state, "For that, what matters is: color: black; crew neck.", 2)

        constraints = active_constraints(state)
        self.assertEqual(constraints["category"], ["Men Clothing Shirts T-Shirts"])
        self.assertEqual(constraints["material"], ["100% cotton"])
        self.assertEqual(constraints["color"], ["color: black"])
        self.assertEqual(constraints["style"], ["crew neck"])
        self.assertEqual(state["scenario_detected"], "buying")

    def test_override_supersedes_only_initial_preference(self) -> None:
        state = new_buyer_state("s", {})
        update_buyer_state(state, "I'm looking for Women Jackets. soft cotton fabric", 1)
        update_buyer_state(state, "For that, what matters is: color: black.", 2)
        update_buyer_state(
            state,
            "Actually, ignore my earlier preference. What I need is: genuine leather.",
            3,
        )

        constraints = active_constraints(state)
        self.assertEqual(constraints["category"], ["Women Jackets"])
        self.assertEqual(constraints["color"], ["color: black"])
        self.assertEqual(constraints["material"], ["genuine leather"])
        old = state["constraints"]["material"][0]
        self.assertEqual(old["value"], "soft cotton fabric")
        self.assertEqual(old["status"], "superseded")
        self.assertEqual(state["scenario_detected"], "intent_override")

    def test_boundary_response_marks_attribute_unavailable(self) -> None:
        tracker = BuyerStateTracker()
        tracker.reset("s", {})
        state = tracker.update("s", "I'm looking for Women Dresses, but I'm still exploring.", 1)
        attribute, question = choose_next_question(state)
        self.assertEqual(attribute, "other")
        tracker.record_question("s", attribute, question)

        state = tracker.update(
            "s", "I don't have a preference for other; please use your judgment.", 2
        )
        next_attribute, _ = choose_next_question(state)
        self.assertIn("other", state["no_preference"])
        self.assertEqual(next_attribute, "feature")

    def test_budget_keeps_around_semantics(self) -> None:
        state = new_buyer_state("s", {})
        update_buyer_state(state, "For that, what matters is: budget around $24.99.", 1)
        record = state["constraints"]["budget"][0]
        self.assertEqual(record["numeric_value"], 24.99)
        self.assertEqual(record["price_mode"], "around")

    def test_budget_under_without_currency_symbol_is_a_maximum(self) -> None:
        state = new_buyer_state("s", {})
        update_buyer_state(state, "I need a shirt under 25", 1)
        record = state["constraints"]["budget"][0]
        self.assertEqual(record["numeric_value"], 25.0)
        self.assertEqual(record["price_mode"], "maximum")

    def test_negated_material_is_not_an_active_constraint(self) -> None:
        state = new_buyer_state("s", {})
        update_buyer_state(state, "I want a black shirt but I don't want wool", 1)
        constraints = active_constraints(state)
        self.assertIn("wool", state["excluded_terms"])
        self.assertNotIn("wool", constraints.get("material", []))

    def test_states_are_isolated_by_session(self) -> None:
        tracker = BuyerStateTracker()
        tracker.reset("one", {})
        tracker.reset("two", {})
        tracker.update("one", "I want a black shirt", 1)
        self.assertIn("black", active_constraints(tracker.get("one"))["color"])
        self.assertEqual(active_constraints(tracker.get("two")), {})


if __name__ == "__main__":
    unittest.main()
