# PATCH_PLAN.md — Prioritized minimal fixes

> **Status 2026-09-10:** Fixes 1–4 (F-01..F-04) IMPLEMENTED per approved
> patches, with one deviation: the F-01 gate returns `done(result)` instead of
> bare `result` so `execution_time_ms` is stamped like every other exit, and
> the F-04 regression test was rewritten from the proposed tautological form
> into a source-parity + truth-table test (importing `evaluate_end_to_end`
> offline is impossible — it imports `datasets` at module top). Regression
> locks added to `tests/smoke_offline.py::TestAuditSecurityFixes` (6 tests).
> Validation gate: suite 39/39 OK, `audit/repro_findings.py` exit 0.
> Fixes 5–12 remain unimplemented, pending direction.

Nothing else below has been implemented — the remainder stays a plan.
Each fix lists affected symbols, risk, the regression test that establishes it
(IDs from `TEST_RESULTS.md` §6), and a validation command. Fixes are ordered by
security impact, then measurement integrity, then hygiene. No redesigns: every
item is local to the cited function.

---

## P1 — security logic (fix first)

### Fix 1 · F-01 — fail-closed resolution of pending dual-key flags
**Severity:** High · **Evidence:** [EXEC] · **Root cause:** the L2 section can be
skipped (`escalate` false) while `l1_dual_key_pending` is set; nothing re-checks
the flag before generation.

- **Where:** `pipeline.py`, `InjectionDetectionPipeline.run()` — insert a guard
  immediately after the L2 block (after ~line 1194), before the generation section:
  if `result.metadata.get("l1_dual_key_pending")` is present and
  `l1_dual_key_resolved` is absent ⇒ block with reason
  `"Layer 1 ambiguous band could not be confirmed by L2 (fail-closed)"`
  (honor `fail_closed=False` by passing through with a metadata warning if the
  operator explicitly opted out of fail-closed).
- **Why this shape:** a single choke point; independent of *why* L2 didn't run
  (gate config, missing probe at a later refactor, future early-return).
- **Risk:** none at default config (pending is always resolved when gate=0.0);
  under non-default configs some ambiguous traffic moves from silent-allow to
  block — which is the stated security posture.
- **Test:** T-01 (+ T-08 row). **Validation:**
  `/home/user/.venv/bin/python tests/smoke_offline.py`.

### Fix 2 · F-02 — raise on total L1 guard failure
**Severity:** High · **Evidence:** [EXEC] · **Root cause:** `_injection_prob`
(`pipeline.py:354–379`) converts every guard exception into a 0.0 vote and logs;
`score()` never distinguishes "all guards failed" from "all guards saw benign text".

- **Where:** `pipeline.py`, `TextClassifierLayer._injection_prob` — track
  per-chunk guard errors; if **every** classifier raised for a chunk, raise
  `RuntimeError("all L1 guards failed on chunk …")`. `run()`'s existing L1
  exception handler then applies the fail-closed policy unchanged.
  Partial failure (≥1 guard healthy) keeps today's max-vote behavior.
- **Risk:** transient single-guard errors no longer degrade silently — intended.
  Ensemble-with-one-down still works because one healthy vote suffices.
- **Test:** T-02. **Validation:** smoke suite + harness repro showing the
  F-02 scenario now raises into the fail-closed path.

### Fix 3 · F-03 — complete the probe fingerprint check
**Severity:** Medium · **Evidence:** [EXEC] · **Root cause:** `load()`
(`pipeline.py:780–822`) validates only model name and layers, while `save()`
records `max_length`, `compute_dtype`, `quantized_4bit` too.

- **Where:** `pipeline.py`, `HiddenStateProbeLayer.load()`:
  - `max_length`: if artifact value exists and differs from runtime, raise
    `ValueError` (feature dimensionality depends on truncation window).
  - `quantized_4bit`: raise if artifact says quantized and runtime model is not
    quantized, or vice-versa (hidden-state distributions differ — the README's
    own retrain doctrine, now enforced).
  - `compute_dtype`: log a warning on mismatch (dtype affects activations less
    drastically; hard-fail would break legitimate fp16↔bf16 re-hosting — or
    hard-fail if the owner prefers strictness: decision D-5b).
- **Risk:** stricter loads may reject currently-"working" mismatched artifacts —
  that is the point; message must say how to fix (retrain / align flags).
- **Test:** T-03. **Validation:** smoke suite + real-hardware step 5 in
  `TEST_RESULTS.md` §6.

## P2 — measurement integrity

### Fix 4 · F-04 — align `evaluate_end_to_end.py` with the production rule
**Severity:** Medium · **Evidence:** [EXEC] · **Root cause:** evaluator's
`hard[i]` (line 96) hard-codes `n_ch == 1` as unambiguous, predating dual-key.

- **Where:** `evaluate_end_to_end.py:96` — replace with the production
  predicate: `hard = p1 >= block and ((p1 >= 0.95) or (n_ch > 1 and n_al >= 2))`,
  and add a CLI flag `--no_dual_key` if the legacy attribution is ever wanted.
  Re-run on Kaggle and refresh README/PROJECT_STATUS gate tables if numbers move.
- **Risk:** published per-layer attribution numbers may shift slightly; that is
  a correction, not a regression.
- **Test:** unit predicate test mirroring `audit/repro_findings.py` §F-04
  (assert evaluator == production truth table). **Validation:** `python
  evaluate_end_to_end.py --help` + Kaggle re-run (hardware plan step 3).

### Fix 5 · F-05 — remove or correctly model the degenerate dual-key config
**Severity:** Low · **Evidence:** [INSPECT + algebra] · **Where:**
`benchmark.py:202`. Options: (a) drop the row (it duplicates always-on), or
(b) model what actually distinguishes dual-key from legacy-single-key:
ambiguous-band rows blocked when `p2 < thr` under single-key vs allowed under
dual-key — i.e., report *legacy* (`hard | (p1>=gate) & (p2>=thr)` with
ambiguous-band forced-block) vs dual-key vs always-on. Recommendation: (a) plus
a comment, minimal churn.
- **Test:** algebraic assertion in smoke suite if kept; else none needed.

### Fix 6 · F-06 — correct the threshold-selection docstring
**Severity:** Low · **Where:** `pipeline.py:644` — replace "LARGEST value whose
OOF FPR stays ≤ budget" with "the recall-maximizing (lowest) threshold whose
out-of-fold FPR stays ≤ budget". Behavior unchanged; alternatively change code
to pick the strictest qualifying threshold — not recommended (loses recall at
the budget edge). **Test:** none (docs), or T-08 calibration assertion.

## P3 — hygiene / residual risk

### Fix 7 · F-12 — validate threshold ranges at construction
**Where:** `InjectionDetectionPipeline.__init__` — assert
`0 <= l1_block_threshold <= 1`, `0 <= l1_escalate_threshold <= l1_block_threshold`
(or warn when `escalate > block`, the F-01 enabler), and
`probe_threshold is None or 0 <= probe_threshold <= 1`. **Test:** T-07.

### Fix 8 · F-08/D-2 — decide the missing-probe policy
Owner decision needed: keep documented downgrade (then log at WARNING once per
pipeline, not per request) or fail-closed construction when `probe_path` was
expected. **No code change until decided.**

### Fix 9 · F-07 — document or shrink the L2 truncation gap
`HiddenStateProbeLayer.max_length=1024` vs generation's 4096. Options: raise L2
`max_length` (VRAM/latency cost on T4 — measure first, hardware plan), or
document the residual risk explicitly in README "Security posture". **Decision
needed before any code change.**

### Fix 10 · F-11 — make `verify.py` verify-then-write
Small reordering in the generated template (`build_verify.py`): decode +
checksum-check in memory, write files only if all digests match; then re-run
the suite. Removes the overwrite-before-verify wart. Regenerate `verify.py`
afterwards and update the pinned SHA in `INSTALL.md` (two-step commit as before).

### Fix 11 · F-09 — recalibrate the deployed threshold off-test
Per README caveat: re-run threshold selection on a validation slice of TRAIN,
keep the test split report-only. Requires a Kaggle training run (hardware plan
step 2) — not executable in this sandbox.

### Fix 12 · D-1 — pin model/dataset revisions
Add `revision=` kwargs (or a `MODEL_REVISIONS` constant) for the HF models and
dataset configs used in training/eval, so runs are reproducible across months.

---

## Suggested implementation order & blast radius

| Order | Fixes | Files touched | New tests | Est. risk |
|---|---|---|---|---|
| 1 | Fix 1, Fix 2 | `pipeline.py` only | T-01, T-02 | low (both add fail-closed paths) |
| 2 | Fix 3 | `pipeline.py` (`load`) | T-03 | low-medium (stricter artifact loads) |
| 3 | Fix 4, Fix 5 | `evaluate_end_to_end.py`, `benchmark.py` | predicate test | low |
| 4 | Fix 6, Fix 7, Fix 10 | docs + init validation + `build_verify.py` | T-07 | low |
| 5 | Fix 8, Fix 9, Fix 11, Fix 12 | pending owner decisions / Kaggle runs | — | n/a |

**Post-fix validation gate (all offline, in this sandbox):**
```
/home/user/.venv/bin/python -m py_compile pipeline.py evaluate_end_to_end.py benchmark.py
/home/user/.venv/bin/python tests/smoke_offline.py            # expect: Ran 33+N tests, OK, 0 skipped
/home/user/.venv/bin/python audit/repro_findings.py           # F-01..F-04 verdicts must flip to 'no gap'
python build_verify.py && python verify.py (in /tmp)          # bundle re-sync + suite green
```
Then, with approval, the Kaggle hardware plan in `TEST_RESULTS.md` §6.
