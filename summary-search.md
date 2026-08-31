# Conversational Search Improvements — Session Summary

## Objective

This session improved the local conversational clothing-search agent in three
initial areas:

1. Synonym and vocabulary expansion.
2. Lightweight local dense/semantic search for Browsing mode.
3. Ranking-weight calibration to move target products toward rank #1 and improve
   Mean Reciprocal Rank (MRR).

After evaluating the first implementation, the remaining failed sessions were
analyzed and four additional improvements were implemented:

1. Discount duplicate evidence.
2. Handle ambiguous or misleading colors more safely.
3. Strengthen informative title and brand evidence.
4. Normalize hybrid product categories.

## Baseline at the Start of the Session

The existing FTS5 reranker produced the following results on the released
200-session public set:

| Metric | Starting value |
|---|---:|
| Passed sessions | 182/200 |
| Hit Rate@10 | 0.910 |
| MRR | 0.616841 |
| Mean Turns to Conversion (MTTC) | 3.225 |
| Technical score | 0.795552 |
| Targets at rank #1 | 100/200 |

## 1. Synonym and Vocabulary Expansion

Created `starter/vocabulary.py` as the shared normalization layer for the
tracker, lexical retriever, and semantic index.

The vocabulary supports common catalog variations such as:

- `kicks`, `running shoes`, `trainers`, and `sneakers`
- `tee`, `t-shirt`, and `tshirt`
- `hoodie`, `sweatshirt`, and `pullover`
- `grey` and `gray`
- `purse` and `handbag`
- `trousers`, `slacks`, and `pants`
- `jogging` and `running`
- `water resistant`, `rainproof`, and `waterproof`
- `light weight`, `lightweight`, and `ultralight`

The implementation preserves original query tokens while adding conservative
canonical concepts and FTS variants. This keeps exact matches strong while
allowing catalog wording to differ from customer wording.

`starter/tracker.py` was also updated so aliases such as `kicks`, `hoodie`,
`tee`, and hybrid product names can be recognized as product categories in
natural free text.

## 2. Local Dense/Semantic Search

The environment did not contain `sentence-transformers`, PyTorch, NumPy, or a
cached MiniLM model. To keep the submission offline and deterministic,
`starter/semantic.py` implements a standard-library local semantic index.

Its main properties are:

- 32-dimensional deterministic dense vectors.
- Canonical fashion concepts as semantic features.
- Dense random indexing using reproducible xorshift vectors.
- Normalized document vectors and cosine similarity.
- Compact hashed posting lists to shortlist candidates before dense scoring.
- No model download, network call, API key, or third-party dependency.
- Stable results independent of Python hash randomization.

Semantic candidates are combined with FTS candidates. Their ranking influence
is deliberately lower than exact field evidence and is strongest in Browsing
mode.

## 3. Retrieval and Ranking Changes

`starter/retriever.py` was extended with the following behavior:

### Candidate generation

- Original multi-route SQLite FTS5 search.
- Expanded synonym-aware FTS route.
- Local semantic candidate route.
- Strict AND routes remain available for high-precision copied catalog clues.
- Category and popularity fallbacks still guarantee valid recommendations.

### Synonym-aware coverage

- Catalog documents receive compact two-hash concept signatures.
- Query constraints are canonicalized before coverage scoring.
- Inverse document frequency gives rare evidence more influence than common
  evidence.

### Field-specific scoring

- Material evidence is weighted more heavily than generic features.
- Explicit brand evidence receives stronger coverage and phrase weights.
- Color, size, style, feature, and use-case evidence have separately calibrated
  values.
- Overrides receive a stronger multiplier so superseded preferences do not
  dominate the new intent.
- Exact informative multiword phrases receive an additional title boost.
- Brand and title matches are evaluated in their appropriate fields rather than
  only against the complete document text.

### Price scoring

- Maximum budgets reward products at or under the limit and penalize products
  above it.
- “Around” budgets use continuous distance from the requested price rather than
  a binary match.

### Evidence deduplication

Repeated facts no longer receive full independent scores. For example,
`polyester` followed by `100% Polyester` retains the more specific evidence but
the second record receives a reduced multiplier. New concepts in a composition,
such as spandex alongside polyester, still contribute additional evidence.

### Confidence-aware colors

Color evidence is now prioritized as follows:

1. An explicit color in structured product details.
2. An unambiguous title color.
3. A color mentioned only in features, fabric blends, or available variants.

Titles containing multiple colors are treated cautiously. For example, in
`Red Hot Chili Peppers ... T-Shirt Black`, `red` is likely part of the named
entity while `black` is the product color. This prevents every color-looking
token from being treated as equally reliable product-color evidence.

### Hybrid-category normalization

The final vocabulary adds targeted cross-taxonomy expansions for products whose
names and catalog categories do not align cleanly:

- shacket → shirt/jacket
- daypack → backpack/sling/crossbody bag
- bathrobe → robe/loungewear
- parka → jacket/coat
- vest → jacket
- undershirt → T-shirt/underwear
- loafer → slip-on/shoe

The associated score boost is restricted to known hybrid concepts. A broader
category-to-title boost was tested but removed because it displaced valid exact
matches.

## 4. Evaluation and Calibration Process

Several complete public-set evaluations were run. The first combined synonym,
semantic, and scoring implementation improved the result to:

| Metric | First implementation |
|---|---:|
| Hit Rate@10 | 0.925 |
| MRR | 0.653020 |
| MTTC | 3.045 |
| Technical score | 0.817506 |
| Rank #1 | 108/200 |

The 15 remaining misses were then inspected by scenario, target metadata,
disclosed constraints, and target position in the candidate pool.

### Main failure patterns found

- Generic clues such as cotton, polyester, `Imported`, and closure type matched
  many products.
- Repeated material facts were being counted independently.
- Colors were sometimes extracted from a band name or fabric-composition text
  rather than the actual product color.
- Hybrid products were stored under surprising catalog branches.
- Boundary sessions withheld or delayed useful information.
- Intent overrides sometimes left only generic material evidence.

Most failed targets appeared in the candidate pool but ranked below position
10, showing that the dominant problem was disambiguation and reranking rather
than hard catalog recall.

### Calibration decisions

An early aggressive version added broad category/title boosts and global
category associations. It recovered some failed cases but reduced MRR, consumed
more memory, and displaced existing hits. That experiment was not retained.

The final version instead uses:

- A moderate duplicate-evidence floor.
- Confidence adjustments rather than hard color rejection.
- Multiword, non-generic title bonuses only.
- Higher explicit brand weights.
- Hybrid-category boosts restricted to known hybrid concepts.
- State-only ranking calculations computed once per recommendation call rather
  than once per candidate.

## 5. Final Public-Set Results

| Metric | Start of session | Final result | Change |
|---|---:|---:|---:|
| Passed sessions | 182/200 | **187/200** | +5 |
| Hit Rate@10 | 0.910 | **0.935** | +0.025 |
| MRR | 0.616841 | **0.653841** | +0.037000 |
| MTTC | 3.225 | **2.970** | -0.255 |
| Technical score | 0.795552 | **0.824252** | +0.028700 |
| Targets at rank #1 | 100/200 | **108/200** | +8 |

Compared with the first improved version, the second calibration recovered two
additional failures without turning any previously passing session into a miss:

- `public_0149`: black sling/crossbody backpack, recovered at rank 9.
- `public_0183`: women’s long vest jacket, recovered at rank 9.

The final scenario metrics are:

| Scenario | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Buying | 0.9250 | 0.658294 | 2.6375 |
| Browsing | 0.9875 | 0.626766 | 2.3250 |
| Intent Override | 0.866667 | 0.708783 | 4.966667 |
| Boundary | 0.8000 | 0.670000 | 4.8000 |

## 6. Tests and Verification

The test suite grew from 13 to 22 tests. New coverage includes:

- `kicks` retrieving a running shoe.
- `tee` retrieving a T-shirt.
- Equivalent vocabulary concepts for grey/gray, hoodie/sweatshirt, and running
  shoe/sneakers.
- Dense synonym similarity exceeding unrelated-product similarity.
- Semantic retrieval of a synonym-only match.
- Hybrid category expansion for daypacks, backpacks, sling bags, and bathrobes.
- Duplicate material evidence receiving a lower second weight.
- Named or ambiguous colors receiving less confidence than an unambiguous
  product color.

Final verification completed successfully:

```bash
python3 -m unittest discover -v
python3 -m compileall -q starter tests evaluator
git diff --check
python3 -m evaluator.local_evaluator
```

- All 22 tests pass.
- Python compilation succeeds.
- The diff check reports no patch errors.
- The final 200-session evaluator run reproduces the documented metrics.

## 7. Files Added or Updated

### Added

- `starter/vocabulary.py` — canonical vocabulary and FTS expansion.
- `starter/semantic.py` — offline dense semantic index.
- `tests/test_vocabulary_semantic.py` — vocabulary and semantic regression tests.
- `summary-search.md` — this session summary.

### Updated

- `starter/retriever.py` — combined retrieval, semantic scoring, calibrated
  reranking, duplicate handling, color confidence, and hybrid normalization.
- `starter/tracker.py` — synonym-aware category recognition.
- `tests/test_agent.py` — retrieval, duplicate-evidence, and color-confidence
  tests.
- `README.md` — architecture description and final benchmark.

The pre-existing untracked `.idea/` directory was not modified.

## Remaining Difficult Cases

Thirteen public sessions remain misses. They are concentrated in generic
material-only intents, crowded apparel/footwear categories, and Boundary or
Intent Override sessions with limited distinguishing information. Further gains
would likely require either better clarification-question selection, a stronger
pretrained local embedding model, or additional structured catalog attributes
rather than larger unconditional ranking boosts.
