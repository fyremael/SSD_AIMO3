# TROPICAL_EXTENSION.md
## Tropical Reranking and Constraint-Guided Aggregation for AIMO3 Validation

This extension is an **auxiliary layer** for the AIMO3 SSD validation pack. It does **not** replace the SSD-style adaptation core. Its role is narrower and more surgical:

1. preserve the base validation question,
2. add a structured reranking surface for sampled math traces,
3. allow cheap, auditable constraint checks to influence final-answer selection,
4. test whether semiring-style pruning and dominance logic improve exact final-answer accuracy.

The guiding idea is simple. SSD-style training is dense and probabilistic. Tropicalization is sparse and combinatorial. For olympiad-style math, that means they naturally belong in different parts of the stack:

- **SSD** for domain adaptation,
- **tropical reranking** for inference-time search, pruning, and answer aggregation.

---

## 1. Why this extension exists

In AIMO3-style evaluation, a fluent solution is not enough. The competition only rewards the final integer answer. A model can generate a polished argument and still fail because of one hidden algebraic or combinatorial error.

That makes inference-time structure disproportionately important. Tropical methods are useful here because they let us reason in terms of:

- dominance,
- path costs,
- pruning,
- hard penalties,
- semiring-style aggregation.

This is a good fit for settings where many sampled traces are verbose, partially redundant, or only locally coherent.

---

## 2. Intended role in the validation ladder

The tropical extension should be tested **after** the base ladder has already been made honest.

Recommended placement:

- `A0`: frozen baseline
- `A1`: naïve SSD
- `A2`: SSD + ordinary selection
- `A3`: SSD + weak filtering
- `A4`: SSD + lightweight verifier
- `A5`: SSD + tropical reranking and constraint-guided aggregation

The purpose of `A5` is not to claim paper-faithful reproduction. It is to test whether our earlier tropical / semiring / DP intuition contributes useful incremental lift in olympiad-style exact-answer evaluation.

---

## 3. Conceptual design

### 3.1 Object of scoring

Each sampled reasoning trace is treated as a candidate path through a latent proof-state graph.

We do **not** assume the path is formally verified. We only assume it can be assigned a structured score built from:

- final-answer extraction validity,
- internal arithmetic consistency,
- simple contradiction penalties,
- branch complexity penalties,
- heuristic signals of proof-state coherence.

### 3.2 Tropical viewpoint

We use a min-plus / max-plus flavored scoring language.

- A **candidate trace** accumulates penalties.
- Lower penalty is better under a min-plus view.
- Equivalent candidates with the same extracted answer can be aggregated into an answer-level score.
- The final prediction is the answer with the best dominant score after tie-breaking.

This is useful when we do not trust probabilities to be well calibrated, but we do trust some invariants enough to penalize obvious failure modes.

### 3.3 What this is not

This is **not** a proof assistant.

It will not certify correctness in the formal sense. It is a structured reranker that tries to separate:

- internally cleaner traces,
- from traces that are self-contradictory, arithmetically sloppy, or extraction-fragile.

---

## 4. Data flow

Recommended data flow for the tropical stage:

1. Run generation with multiple samples per problem.
2. Run answer extraction.
3. Normalize each trace into a lightweight proof-state object.
4. Score each trace with cheap constraints.
5. Aggregate by extracted answer.
6. Select the answer with the best tropical aggregate.

```text
problem -> sampled traces -> extracted answers -> proof-state normalization
       -> constraint penalties -> answer-level aggregation -> final prediction
```

---

## 5. Scoring model

Each candidate trace receives a score decomposition:

```text
total_penalty =
    extraction_penalty
  + contradiction_penalty
  + arithmetic_penalty
  + structure_penalty
  + complexity_penalty
  + optional_length_penalty
```

Lower is better.

### 5.1 Extraction penalty

- `0` if a valid integer answer is extracted in range.
- large fixed penalty if no valid answer is extracted.

### 5.2 Contradiction penalty

Cheap contradiction checks may include:

- multiple incompatible “Final Answer” declarations,
- inconsistent parity claims on the same variable,
- contradictory sign or equality claims when detected by simple patterns.

### 5.3 Arithmetic penalty

Cheap arithmetic checks may include:

- malformed explicit integer equations,
- incorrect arithmetic in patterns like `a+b=c`, `a*b=c`, `a-b=c`, `a/b=c` with integer literals,
- residue inconsistencies for small-modulus statements when explicitly asserted.

### 5.4 Structure penalty

Reward or penalize surface signals such as:

- explicit final-answer declaration,
- boxed answer,
- step structure,
- degenerate extremely short traces that jump straight to an answer,
- traces with no mathematical operators or no intermediate reasoning markers.

### 5.5 Complexity penalty

This is a guardrail rather than a theorem. Excessively long or branchy traces may be penalized slightly when they do not buy consistency.

---

## 6. Answer-level aggregation

After scoring traces, aggregate by final extracted answer.

Let `S(a)` be the set of traces that end in answer `a`.

Useful answer-level summaries:

- best-trace penalty for `a`,
- mean penalty for `a`,
- count of traces supporting `a`,
- vote entropy,
- support diversity across prompt templates or sampling temperatures.

A practical decision rule is:

```text
answer_score(a) = best_penalty(a) - lambda_count * log(1 + support_count(a))
```

Choose the answer with minimum `answer_score(a)`.

This keeps the tropical spirit: the dominant clean path matters most, but repeated support also matters.

---

## 7. Minimal proof-state schema

The extension uses a lightweight schema rather than a full symbolic proof object. The intent is auditability, not grandeur.

Each record may contain:

- `problem_id`
- `sample_index`
- `trace_text`
- `extracted_answer`
- `steps`: array of simple parsed step objects
- `features`: cheap surface-level reasoning features
- `constraint_checks`: results of arithmetic / contradiction / modular checks
- `tropical_score`: final scalar penalty

The starter JSON schema lives at:

- `prompts/proof_state_schema.json`

---

## 8. Constraint library philosophy

The constraint library should stay **cheap**, **auditable**, and **domain-safe**.

Good examples:

- range checks,
- parity checks,
- divisibility claims with explicit integers,
- arithmetic equalities over explicit integers,
- repeated final-answer conflict detection,
- simple bound consistency.

Bad examples for the first pass:

- opaque neural verifiers,
- theorem-prover dependence,
- expensive symbolic algebra with hidden heuristics,
- silently problem-specific handcrafted rules that would contaminate the scientific claim.

---

## 9. Experimental posture

### Positive result

Call the tropical extension useful only if it improves exact final-answer accuracy over the matched non-tropical comparison under the same base model, same sampled generations, and same extraction logic.

### Weak result

If it mostly reduces invalid answers or improves answer concentration but not exact-answer accuracy, record that as operationally interesting but not yet a scientific win.

### Negative result

If it only adds complexity, over-penalizes creative but correct traces, or produces brittle prompt-dependent behavior, remove it from the default path.

---

## 10. Immediate implementation pieces

This extension adds:

- `scripts/constraint_library.py`
- `scripts/tropical_rerank.py`
- `prompts/proof_state_schema.json`
- `tests/test_tropical_rerank.py`
- `configs/a5_tropical_rerank.yaml`

These are starter surfaces, not final production machinery.

---

## 11. Recommended first experiment

Do the simplest clean test first.

1. Take an existing A2 or A4 evaluation run.
2. Reuse the same sampled traces.
3. Apply the tropical reranker offline.
4. Compare:
   - ordinary majority vote,
   - logprob tie-break,
   - tropical rerank.

This isolates the value of reranking itself.

Only after that should we test tropical signals inside a larger online search loop.

---

## 12. Sober verdict

Tropicalization is plausibly useful **here**, but mainly as:

- a structured reranker,
- a pruning logic,
- a cheap verifier scaffold,
- an answer-level aggregation device.

It is **not** the first thing to swap into the SSD training loss.

That separation is intentional and should remain part of the Grand Challenge team’s framing.
