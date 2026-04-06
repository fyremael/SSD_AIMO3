# TASKS.md
## Grand Challenge Technologies
## SSD-Style Validation for AIMO3
**Status:** ready for execution  
**Date:** 2026-04-03  
**Parent spec:** `SPECS_SSD_Validation_AIMO3_Grand_Challenge.md`  
**Program posture:** validate the described approach; do **not** chase exact replication

---

## 0. Mission

We are not trying to reproduce Apple’s exact table values or public artifacts. We are trying to answer a narrower and more useful question:

> Does an SSD-style self-distillation pass, applied as domain adaptation for olympiad-style math, improve exact-integer answer accuracy under a disciplined AIMO3-style evaluation stack?

Success means **directional validation**, not paper cosplay.

---

## 1. Ground truth question

The single highest-priority decision question is:

> After tuning prompts, decoding, answer extraction, and selection fairly on both sides, does SSD-style adaptation beat the strongest frozen-model baseline on exact final-answer accuracy?

Everything else is subordinate to that.

---

## 2. Program structure

We execute five experiment tiers:

- **A0** — frozen baseline
- **A1** — naïve SSD math adaptation
- **A2** — SSD + inference-time selection
- **A3** — SSD + weak filtering of obviously bad self-generated traces
- **A4** — SSD + lightweight verifier signals

We advance only if the prior tier produces a real positive signal.

---

## 3. Hard success criteria

A tier is considered positive only if all of the following hold:

1. It improves **exact final-answer accuracy** over the best fair lower tier.
2. It does **not** depend on a broken answer extractor or looser parsing.
3. It remains positive across at least **2 random seeds** or **2 independent decode batches**.
4. It survives at least **2 prompt templates** for the same evaluation set.
5. It does not merely increase verbosity while keeping answers wrong.

A tier is considered strong if it also improves:
- vote concentration or self-consistency,
- final-answer extraction reliability,
- or calibrated abstention/uncertainty handling.

---

## 4. Stop conditions

We stop or redesign the branch if any of the following appears:

- no gain versus the best decode-only frozen baseline after fair tuning
- answer extraction failure rate rises materially
- reasoning traces get longer without exact-answer improvement
- gains appear only on a single prompt template
- public-LB-style gains do not survive hidden holdout or shadow split tests

---

## 5. Required shared assets

The following must exist before any result is considered reportable:

- versioned problem manifest
- versioned unlabeled prompt corpus manifest
- prompt-template manifest
- decoding config manifest
- answer-extraction ruleset
- run metadata JSON for every experiment
- sample transcripts for error analysis
- evaluation notebook or script for paired comparisons

---

## 6. Workstreams and owners

### 6.1 Coordination Agent
Own the board, the dependency graph, and the daily state summary.

**Deliverables**
- `docs/program_board.md`
- `docs/run_registry.md`
- `docs/decision_log.md`

**Checklist**
- [ ] Define run naming convention.
- [ ] Define artifact retention policy.
- [ ] Confirm compute budget per tier.
- [ ] Confirm freeze/go rules.
- [ ] Merge daily notes from all agents.

---

### 6.2 Research Agents
Own protocol integrity and critique.

**Deliverables**
- `docs/hypotheses.md`
- `docs/ablation_matrix.md`
- `docs/mechanism_notes.md`
- `docs/critique.md`

**Checklist**
- [ ] Translate A0–A4 into falsifiable hypotheses.
- [ ] Define the minimal ablation matrix.
- [ ] Specify what counts as a real positive result.
- [ ] Pre-register expected failure modes.
- [ ] Write interpretation notes after each tier.

**Questions to answer**
- Does SSD help exact answers or only reasoning style?
- Does it change vote entropy in a useful way?
- Does weak filtering help because the traces are better, or just because junk is removed?
- Does verifier-lite materially outperform paper-faithful SSD?

---

### 6.3 Data Agents
Own corpus assembly and manifests.

**Deliverables**
- `docs/corpus_manifest.md`
- `docs/data_lineage.md`
- `docs/dedup_report.md`

**Checklist**
- [ ] Assemble unlabeled olympiad-style prompt pool.
- [ ] Remove direct overlap with eval/holdout sets.
- [ ] Deduplicate near-duplicate prompts.
- [ ] Tag prompts by topic if feasible: algebra, number theory, combinatorics, geometry.
- [ ] Produce stable train/dev/analysis partitions for unlabeled prompts.
- [ ] Export JSONL prompt pack with hashes.

**Hard rules**
- No contamination from held-out evaluation problems.
- No silent corpus edits.
- Every prompt record must carry source, hash, and split.

---

### 6.4 Coder Agents
Own generation, training, and inference code.

**Deliverables**
- `generate_self_samples.py`
- `train_ssd_math.py`
- `run_eval_math.py`
- `extract_answer.py`
- `aggregate_votes.py`
- `tests/`

**Checklist**
- [ ] Implement self-sample generation from a frozen base model.
- [ ] Implement weak filtering hooks.
- [ ] Implement plain SFT on self-generated traces.
- [ ] Implement AIMO-style answer extraction.
- [ ] Implement self-consistency voting.
- [ ] Emit run metadata and sample transcripts.
- [ ] Add tests for extraction edge cases.

**Hard rules**
- One config file = one run.
- No hidden defaults.
- Every saved model must have an attached config snapshot.

---

### 6.5 Evaluation Agents
Own benchmark fidelity.

**Deliverables**
- `docs/eval_protocol.md`
- `docs/metric_audit.md`
- `docs/holdout_policy.md`

**Checklist**
- [ ] Define exact final-answer accuracy metric.
- [ ] Define prompt-template comparison protocol.
- [ ] Define paired evaluation script.
- [ ] Audit extraction success/failure.
- [ ] Track accuracy by topic and difficulty if available.
- [ ] Track vote entropy and answer agreement.

**Hard rules**
- Never compare an untuned baseline against a tuned student.
- Always report best frozen baseline under the same inference budget family.
- Keep evaluation budgets comparable.

---

### 6.6 Infra Agents
Own environments and reproducibility.

**Deliverables**
- `docs/env.md`
- `docs/compute_matrix.md`
- `docker/` or equivalent env lockfiles

**Checklist**
- [ ] Pin transformer/vLLM/train stack versions.
- [ ] Define GPU memory envelopes per run class.
- [ ] Define artifact storage layout.
- [ ] Define resume/restart behavior for interrupted runs.
- [ ] Verify inference and training code produce deterministic metadata.

---

### 6.7 Analysis Agents
Own qualitative and quantitative error review.

**Deliverables**
- `docs/error_taxonomy.md`
- `docs/casebook.md`

**Checklist**
- [ ] Classify errors: extraction, arithmetic slip, combinatoric miss, geometry misread, premature collapse, hallucinated lemma.
- [ ] Compare wrong-but-elegant traces vs wrong-and-chaotic traces.
- [ ] Compare vote spread before and after SSD.
- [ ] Surface cases where SSD improves structure but not truth.

---

## 7. Dependency graph

### Phase 0 — before any training
- [ ] Finalize corpus manifest
- [ ] Finalize evaluation manifest
- [ ] Finalize answer extractor
- [ ] Finalize prompt templates
- [ ] Finalize frozen-baseline decode sweep plan

### Phase 1 — A0 frozen baseline
- [ ] Run decode sweep on frozen model(s)
- [ ] Tune prompt template(s)
- [ ] Measure exact accuracy, extraction reliability, vote entropy
- [ ] Select strongest fair frozen baseline

**Gate to Phase 2:** A0 package is complete and reproducible.

### Phase 2 — A1 naïve SSD
- [ ] Generate one self-sampled trace per unlabeled prompt
- [ ] Apply minimal filtering only
- [ ] Train student with plain SFT
- [ ] Evaluate under same or matched inference family
- [ ] Compare against best A0 baseline

**Gate to Phase 3:** positive directional signal or a well-diagnosed negative result.

### Phase 3 — A2 inference-time selection
- [ ] Add majority voting
- [ ] Add answer canonicalization
- [ ] Add entropy / agreement logging
- [ ] Evaluate cost/benefit

### Phase 4 — A3 weak filtering
- [ ] Reject null traces
- [ ] Reject traces with missing final answer
- [ ] Reject malformed or extraction-broken traces
- [ ] Retrain and compare

### Phase 5 — A4 verifier-lite
- [ ] Add lightweight checker signals
- [ ] Keep only low-cost, low-fragility filters
- [ ] Retrain and compare against A3

---

## 8. Minimal experiment matrix

We do **not** explode the grid at the start.

### Required variables
- base model identity
- prompt template
- training-time sampling temperature
- evaluation-time temperature
- number of inference samples
- filtering policy
- selection policy
- seed

### First-pass matrix
- 1 primary base model
- 1 backup base model
- 2 prompt templates
- 2 training temperatures
- 2 evaluation temperatures
- 2 seeds for A0/A1
- 1 small weak-filter policy
- 1 lightweight verifier policy

This is enough to validate the method without drowning in combinatorics.

---

## 9. Metrics to log every time

### Primary
- exact final-answer accuracy

### Secondary
- answer extraction success rate
- invalid-output rate
- average reasoning length
- vote entropy
- majority-vote agreement
- unique-answer count per problem
- calibration proxy if available
- latency and token cost

### Slice metrics
- topic slice
- problem length slice
- prompt-template slice
- confidence slice

---

## 10. Required plots

- A0 vs A1 exact-accuracy bar chart
- accuracy vs evaluation temperature
- vote entropy histogram before/after SSD
- answer extraction success before/after SSD
- reasoning length vs accuracy scatter
- per-topic delta chart
- cost-vs-accuracy plot for A0–A4

---

## 11. Error taxonomy

Every failed run review should classify at least 20 examples into:

- correct reasoning, correct answer
- coherent reasoning, wrong integer
- plausible reasoning, extraction failure
- early branch collapse
- arithmetic slip late in trace
- wrong symbolic setup
- geometry/combinatorics misparse
- multiple candidate answers with weak selection
- malformed final response

---

## 12. Red-team questions

- Are we accidentally rewarding better formatting instead of better solving?
- Did the student gain only because the baseline was under-tuned?
- Are we contaminating evaluation through corpus overlap?
- Are verifier-lite gains really SSD gains, or just filtering gains?
- Does SSD help only the easiest style-consistent subcases?
- Are public-LB-style gains misleading us?

---

## 13. Deliverable package for management

At the end of the first full ladder, deliver:

1. `EXEC_SUMMARY.md`
2. `RESULTS_TABLE.csv`
3. `RUN_MANIFEST.json`
4. `CRITIQUE.md`
5. `RECOMMENDATION.md`

The recommendation must answer only one of three things:

- **Proceed** — SSD-style adaptation is a useful building block
- **Proceed conditionally** — only useful with filtering/selection/verifier-lite
- **Do not proceed** — no meaningful validation signal

---

## 14. Decision thresholds

### Proceed
Use if A1 or later beats the best A0 frozen baseline on exact-answer accuracy with acceptable extraction behavior and survives basic robustness checks.

### Proceed conditionally
Use if naïve SSD is weak, but A3/A4 shows consistent value inside a realistic inference stack.

### Do not proceed
Use if improvements vanish after fair baseline tuning or fail to survive prompt / seed variation.

---

## 15. Immediate next actions

### Today
- [ ] Lock corpus and holdout manifests.
- [ ] Lock prompt templates.
- [ ] Lock answer extractor.
- [ ] Run A0 sweep.

### Next
- [ ] Launch A1 generation job.
- [ ] Train first SSD student.
- [ ] Compare A1 vs strongest A0.
- [ ] Only then branch into A2/A3/A4.

---

## 16. Final orientation

This program is not about proving that SSD is magical.

It is about determining whether a very cheap and elegant self-adaptation step can move a strong math model into a more favorable distribution for AIMO3-style exact-answer solving, once rigorous inference-time selection and extraction are in place.
