# LOGIC_MATRIX.md — Decision tables, predicates, boundaries, config interactions

All line references are to the repository at HEAD `34c4d52`.
Predicates are quoted verbatim from source; `[EXEC]` marks rows reproduced by
`audit/repro_findings.py` (see `TEST_RESULTS.md` §5).

---

## 1. L1 verdict logic — `TextClassifierLayer.score` + `run()`

Per-chunk probability (`pipeline.py:354–379`): for each guard, injection-label
score from the HF pipeline; guard exception ⇒ that guard's vote is **0.0** and
the error is only logged. Chunk score = `max(votes)` across guards.

Production verdict (`pipeline.py:1109–1144`), with `block = l1_block_threshold`
(default **0.85**, comparisons are **`>=`** throughout):

| # | Condition (exact) | Route | Verdict | Test coverage |
|---|---|---|---|---|
| 1 | `p1 >= block and (p1 >= 0.95 or (chunks>1 and alerts>=2))` | early return | **BLOCK (L1)** | `test_l1_hard_block_and_fail_closed`, `test_extreme_peak_blocks_without_l2` |
| 2 | `p1 >= block` and condition 1 false, `l1_dual_key and probe is not None` | set `l1_dual_key_pending`, continue | **defer to L2** | `test_ambiguous_band_requires_l2_confirm`, `test_ambiguous_band_vetoed_by_l2` |
| 3 | `p1 >= block`, condition 1 false, `¬dual_key ∨ probe is None` | early return | **BLOCK (ambiguous, no confirmation)** | `test_ambiguous_band_blocks_when_no_probe`, `test_dual_key_disabled_restores_legacy` |
| 4 | `p1 < block` | continue | pass L1 | implicit |
| 5 | `layer1.score()` raises, `fail_closed` | early return | **BLOCK (L1 Fault)** | `test_l1_hard_block_and_fail_closed` |
| 6 | `layer1.score()` raises, `¬fail_closed` | `l1_prob = 0.0`, continue | pass L1 | same test |
| 7 | **every guard raises on every chunk** | `score()` returns `("SAFE", 0.0)` — no exception surfaces | **pass L1 silently** | **none — F-02 [EXEC]** |

Boundary notes:
- `p1 = 0.85` exactly ⇒ row 2/3 (block band inclusive) [INSPECT].
- `p1 = 0.95` single chunk ⇒ row 1 (inclusive `>=`) [INSPECT; covered by `test_extreme_peak_blocks_without_l2` at 0.97, not at exact 0.95 — boundary untested].
- Multi-chunk hard block needs `alerts >= 2` — one quoted attack in a long doc escalates instead of blocking (intended, README-documented).
- NaN cannot currently reach these comparisons from the HF pipeline [HYPOTHESIS — depends on guard model outputs]; `NaN >= 0.85` is `False`, so a NaN score would route as SAFE-at-L1 — same silent-pass class as F-02.

## 2. Escalation + L2 logic (`pipeline.py:1156–1194`)

```
escalate = (l1_prob >= l1_escalate_threshold) or bool(ctx_raw)
```

| # | Condition | Route | Verdict | Coverage |
|---|---|---|---|---|
| 1 | `escalate ∧ probe is not None ∧ p2 >= active_thr` | early return | **BLOCK (L2)** (`active_thr = probe_threshold ?? artifact threshold`, `>=`) | `test_l2_escalation_block` |
| 2 | `escalate ∧ probe ∧ p2 < active_thr ∧ pending` | `resolved = "L2_PASS"`, continue | dual-key veto of L1 | `test_ambiguous_band_vetoed_by_l2` |
| 3 | `escalate ∧ probe ∧ p2 < active_thr ∧ ¬pending` | continue | pass | implicit |
| 4 | `escalate ∧ probe is None` | log info only, continue | **L2 skipped silently** (F-08) | none |
| 5 | `¬escalate` | L2 skipped | pass; **if `pending` was set, it stays unresolved (F-01)** | **none [EXEC]** |
| 6 | L2 raises, `fail_closed` | early return | **BLOCK (L2 Fault)** | none direct (fail-closed pattern covered for L1) |
| 7 | L2 raises, `¬fail_closed` | continue to generation | pass | none |

Reachability of row 5 with `pending`: needs `l1_escalate_threshold > l1_prob >=
l1_block_threshold` and empty `untrusted_context`. Impossible at the default
gate 0.0; reachable with e.g. `l1_escalate_threshold=0.90, l1_prob=0.88`.
**[EXEC]: request reached `generate_fn` with `pending=True, resolved=None`.**

## 3. Generation + L3 logic (`pipeline.py:1198–1265`)

| # | Condition | Verdict | Coverage |
|---|---|---|---|
| 1 | `generate_fn` is None | return unblocked, reason "no generation routine" (input-gate-only mode) | `test_pass_no_generate_fn` |
| 2 | generation raises, `fail_closed` | **BLOCK (inference fault)** | none (proposed T-04) |
| 3 | canary token appears in response | **BLOCK (exfiltration)** | `test_canary_exfiltration_block` |
| 4 | regex heuristics flag + judge `COMPROMISED` | **BLOCK (self-judge)** | none (proposed T-05) |
| 5 | heuristics flag + judge `ERROR`, `fail_closed` | **BLOCK (judge fault)** | none |
| 6 | heuristics flag + judge `SKIPPED` | pass with warning — unreachable via full pipeline (judge always attached) | `test_judge_aliases` (SKIPPED only) |
| 7 | heuristics flag + judge `SAFE` | pass; judge score recorded | `test_heuristics_pass_through_without_judge` (SKIPPED variant) |
| 8 | no flag | pass; `response` attached **after** all checks | `test_clean_output_passes` |

Output ordering confirmed: `result.response` is assigned only at
`pipeline.py:1263` after canary/heuristics/judge checks — no release-before-audit
path exists [INSPECT].

Judge parsing: verdict read from `raw.upper()[:80]`; `COMPROMISED` checked
before `SAFE` (`pipeline.py:943–951`) — a response containing both words in the
first 80 chars is treated as COMPROMISED (security-conservative) [INSPECT].

## 4. Threshold calibration — `HiddenStateProbeLayer.train` (`pipeline.py:712–750`)

- CV: `StratifiedGroupKFold` when `groups` given, else `StratifiedKFold` (with
  logged leakage warning) [INSPECT + tested `TestLeakFreeCV`].
- Calibration source: out-of-fold probabilities when CV feasible, else
  in-sample (logged as optimistic) [INSPECT].
- Selection: `ok = np.where(fpr_rates <= fpr_budget)[0]; idx = ok[-1]`
  (`pipeline.py:727–728`). ROC arrays are ordered by increasing FPR / decreasing
  threshold, so `ok[-1]` is the **lowest threshold** whose OOF FPR ≤ budget
  (recall-maximizing at the budget edge). Docstring `pipeline.py:644` says
  "LARGEST value" — wording contradicts code; behavior is the standard
  budget-edge operating point (F-06) [INSPECT].
- No operating point reaches budget ⇒ fallback `threshold = 0.90` with warning
  (`pipeline.py:735–740`) [INSPECT, untested].

## 5. Artifact save/load contract (`pipeline.py:758–822`)

| Field | Saved | Checked on load | Restored on load |
|---|---|---|---|
| `probe`, `scaler` | ✓ | — | ✓ |
| `model_name` | ✓ | **✓ raises ValueError** | — |
| `layers`/`layer` | ✓ | **✓ raises ValueError** | ✓ |
| `pooling` | ✓ | ✗ (artifact wins) | ✓ |
| `threshold`, `fpr_budget` | ✓ | ✗ | ✓ |
| `max_length` | ✓ | **✗ ignored** | **✗** (runtime default kept) |
| `compute_dtype` | ✓ | **✗ ignored** | ✗ |
| `quantized_4bit` | ✓ | **✗ ignored** | ✗ |

F-03 [EXEC]: artifact with `max_length=512, quantized_4bit=True` loads into a
`max_length=1024`, non-quantized runtime without error. README instructs manual
retraining after quantization switches, but nothing enforces it.

## 6. Configuration interactions (cross-table)

| Config combination | Result | Class |
|---|---|---|
| defaults (gate 0.0, dual_key on, probe attached) | all rows behave per tables 1–3 | OK [INSPECT+EXEC] |
| `l1_escalate_threshold ∈ (block, 1.0]`, no ctx, L1 in ambiguous band | **F-01**: pending unresolved → generation | **defect [EXEC]** |
| `probe_path=None` | L2 skipped everywhere; dual-key ambiguous band blocks (row L1-3); `escalate` log-spams per request | documented downgrade (F-08) |
| `fail_closed=False` | L1/L2/generation/judge faults all become pass-throughs | explicit flag, but no warning at construction |
| `probe_threshold > 1.0` | L2 can never block; pending dual-key rows are "vetoed" by an impossible bar | **no range validation (F-12)** [INSPECT] |
| `l1_block_threshold > 1.0` | L1 never blocks; L2 + L3 remain active | no range validation (F-12) |
| `load_in_4bit=True` + fp16-trained artifact | load() succeeds; scores invalid (hidden states changed) | **F-03 consequence** [INSPECT] |
| L1 ensemble, one guard down | remaining guard's vote still taken (max) — graceful | OK by design |
| L1 ensemble, **all** guards down | SAFE 0.0 (F-02), then L2-only defense if probe present | **defect [EXEC]** |

## 7. Boundary & degenerate-value inventory

| Input/state | Behavior | Evidence |
|---|---|---|
| `p1` exactly 0.85 / 0.95 | inclusive `>=` ⇒ block band / unambiguous | [INSPECT], exact-boundary tests missing |
| `p2` exactly `threshold` | `>=` ⇒ block | [INSPECT] (`score()` line 848) |
| empty `user_input`, non-empty ctx | `prepared` = `"\n\n<untrusted_context>…"` (leading newlines, no strip of composed result); scored normally | [INSPECT], cosmetic |
| empty everything | L1 scores `""`; L2 runs (gate 0.0); generation gets empty user turn | [NOT-TESTED] |
| >4096-token chat prompt | truncated at generation (`pipeline.py:1052`) | [INSPECT] |
| >1024-token input to L2 | truncated, warning logged (`pipeline.py:487–494`) — **F-07**: L2 blind beyond token 1024 while LLM sees 4096 | [INSPECT] |
| >`max_chunks` windows at L1 | first N−1 windows + tail, middle skipped, warning logged (`pipeline.py:316–333`) | [INSPECT + tested `TestSlidingChunkGuards`] |
| NaN/inf scores | would compare as SAFE/pass; unreachable from HF pipeline in practice | [HYPOTHESIS] |
| constant training feature | `StandardScaler` scale_=0 ⇒ NaN/inf in probe scores | [HYPOTHESIS — F-14] |

## 8. Evaluator-vs-production rule comparison (F-04)

Production (`pipeline.py:1116`): `unambiguous = (p1 >= 0.95) or (multi and alerts >= 2)`
Evaluator (`evaluate_end_to_end.py:96`): `hard = p1 >= block and (n_ch == 1 or n_al >= 2 or p1 >= 0.95)`
Benchmark (`benchmark.py:189–190`): matches production.

[EXEC] truth table (block=0.85):

| p1 | chunks | alerts | production hard-blocks? | evaluator hard-blocks? |
|---|---|---|---|---|
| 0.87 | 1 | 1 | **No** (dual-key defer) | **Yes** ← diverges |
| 0.90 | 1 | 0 | **No** | **Yes** ← diverges |
| 0.97 | 1 | 1 | Yes | Yes |
| 0.87 | 5 | 2 | Yes | Yes |
| 0.87 | 5 | 1 | No | No |

Impact: `evaluate_end_to_end.py`'s "L1 hard-blocks N/60" and Policy A/B counts
treat single-chunk ambiguous-band rows as blocked at L1, while production defers
them to L2. End-to-end TPR/FPR may still coincide (L2 confirms/vetoes), but the
per-layer attribution and latency accounting published from this script do not
mirror serving. `benchmark.py:202` additionally defines
`dual_key = hard | ((p1 >= block) & ~hard & (p2 >= thr)) | (p2 >= thr)`, which
reduces to `hard | (p2 >= thr)` — algebraically identical to `always_on` (F-05).

## 9. Untested security-critical paths (candidate list for PATCH_PLAN)

1. F-01 config interaction (pending + high gate + no ctx).
2. F-02 total/partial L1 guard failure.
3. Generation fault under `fail_closed=True/False` (table 3 row 2).
4. Judge `COMPROMISED` and `ERROR` verdicts end-to-end through `run()`.
5. L2 fault (`score()` raising) fail-closed path.
6. `probe_threshold` override honored by `run()` (currently only exercised via evaluator CLIs).
7. Exact boundary values 0.85 / 0.95 / threshold equality.
8. Artifact with conflicting `max_length`/dtype/quant fields (F-03).
9. Empty-string and whitespace-only inputs through full `run()`.
10. NaN score injection at L1/L2 boundaries.
