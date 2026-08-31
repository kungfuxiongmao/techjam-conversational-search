from __future__ import annotations

import unittest

from starter.semantic import LocalSemanticIndex
from starter.vocabulary import canonical_tokens, expanded_search_terms


class VocabularyAndSemanticTest(unittest.TestCase):
    def test_common_catalog_variants_share_canonical_concepts(self) -> None:
        self.assertEqual(canonical_tokens("grey hoodie"), canonical_tokens("gray sweatshirt"))
        self.assertEqual(canonical_tokens("running shoe"), canonical_tokens("sneakers"))
        self.assertEqual(canonical_tokens("tee"), canonical_tokens("t-shirt"))

    def test_kicks_expands_to_catalog_footwear_terms(self) -> None:
        expanded = expanded_search_terms("kicks")
        self.assertIn("sneaker", expanded)
        self.assertIn("sneakers", expanded)
        self.assertIn("shoe", expanded)

    def test_hybrid_categories_expand_across_catalog_taxonomies(self) -> None:
        backpack_terms = expanded_search_terms("casual daypacks")
        self.assertIn("backpack", backpack_terms)
        self.assertIn("sling", backpack_terms)
        self.assertIn("crossbody", backpack_terms)
        self.assertEqual(canonical_tokens("bathrobes"), canonical_tokens("robe"))

    def test_dense_similarity_understands_synonyms(self) -> None:
        index = LocalSemanticIndex()
        synonym_similarity = index.similarity("lightweight running shoe", "light sneakers")
        unrelated_similarity = index.similarity("lightweight running shoe", "formal wool coat")
        self.assertGreater(synonym_similarity, unrelated_similarity)

    def test_dense_index_retrieves_a_synonym_only_match(self) -> None:
        index = LocalSemanticIndex()
        index.add("RUN", "mesh trail sneakers with cushioned soles")
        index.add("COAT", "formal wool overcoat for winter")
        ranked = index.search("running shoe", limit=2)
        self.assertEqual(next(iter(ranked)), "RUN")


if __name__ == "__main__":
    unittest.main()
