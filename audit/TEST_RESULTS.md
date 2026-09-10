# TEST_RESULTS.md — Executed commands, results, limitations, proposed tests

Interpreter for all executions: `/home/user/.venv/bin/python` (Python 3.11.2;
numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1, joblib 1.6.0; **no torch /
transformers / datasets**). Working directory `/home/user/Ens_v2` unless noted.
No source, test, dependency, or config file was modified by any command below.

---

## 1. Commands actually executed (this audit)

| # | Command | Exit | Result summary |
|---|---|---|---|
| E1 | `git rev-parse HEAD && git status --porcelain && git log --oneline -5` | 0 | HEAD `34c4d52`, clean tree, 3 commits |
| E2 | `sha256sum <15 repo files>` | 0 | hashes recorded in `AUDIT.md` §2 |
| E3 | `nvidia-smi; ls /dev/nvidia*; grep nvidia /proc/modules` | ≠0 | **no GPU hardware present** |
| E4 | `find /home/user -name pipeline.py -not -path "*/.venv/*" -not -path "*/.git/*"` | 0 | exactly one copy (repo root) |
| E5 | `python -m py_compile <all 10 .py files>` | 0 | all compile clean |
| E6 | `python tests/smoke_offline.py` | 0 | `Ran 33 tests … OK` (0 skipped, 0 failed) |
| E7 | `cp verify.py /tmp/audit_bundle && python verify.py` (cwd=/tmp/audit_bundle) | 0 | 4 files extracted, `all checksums match`, `smoke suite PASSED` |
| E8 | `sha256sum` comparison of the 4 extracted files vs repo copies | 0 | **all four IDENTICAL** — bundle in sync with reviewed sources |
| E9 | `python audit/repro_findings.py` | 0 | findings F-01..F-04 + Phase E contract reproduced (§5) |

Smoke-suite tail (E6):

```
Ran 33 tests in 0.019s
OK
```

(The interleaved `[dataset] bad/x ... FAILED: no column(s) ['label']` line is
expected stdout from `test_missing_label_column_raises`, which asserts that
`build_training_set` hard-stops when a spec names a nonexistent column.)

## 2. Environment limitations (what these results do NOT prove)

- **No GPU, no torch:** every torch/transformers interaction ran against the
  suite's stubs. VRAM policy, dtype selection, device placement, chunked HF
  classification, hidden-state extraction, and generation are verified only at
  the control-flow level.
- **No model/dataset downloads permitted by audit rules:** L1 guard behavior,
  probe training, label mappings, and dataset schemas were inspected in loader
  code only.
- Passing offline tests therefore **do not establish production readiness** —
  they establish policy/plumbing correctness under stubs.

## 3. Phase E — the reported `test_cpu_fallback` failure

**Reported artifact:** `tests/smoke_offline.py:104`,
`assert P.t4_dtype() == torch.float32`, AssertionError, in a 26-test run on a
GPU-enabled Kaggle host.

**Reproduction attempt:** NOT reproducible against current HEAD [EXEC E6].

Evidence chain:

1. **Function under test** — `t4_dtype()` (`pipeline.py:145`) is an alias of
   `VRAMManager.get_optimal_dtype()` (`pipeline.py:86–97`):
   `is_available() == False ⇒ float32` is the FIRST branch; `CC >= 8 ∧
   bf16-supported ⇒ bfloat16`; else `float16`. No module-level caching exists
   (scan of `vars(pipeline)` found no dtype-valued module attributes [EXEC E9]).
2. **Current test** (`tests/smoke_offline.py:124–135`) mocks
   `pipeline.torch.cuda.is_available → False` and `device_count → 0` — i.e., it
   patches exactly what the function reads, in the module where it is looked
   up. A companion `test_gpu_host_reports_its_own_dtype` guards the inverse.
3. **Executed contract check** [EXEC E9, mocked hardware]:
   `CPU → float32`, `Turing CC 7.5 → float16`, `Ampere CC 8.0 + bf16 → bfloat16`.
4. **Historical root cause** (from repo provenance, [INSPECT]): the pre-fix
   test asserted float32 while *assuming* the host had no GPU; on an
   accelerator-enabled Kaggle session the real `is_available()` returned True
   and fp16 was (correctly) returned. The defect was in **test setup /
   environment assumption**, not production behavior. The fix (mock-based test)
   is merged — `patch_test_cpu_fallback.py` documents it and is superseded.
5. The 26-test count in the reported run identifies it as a **stale bundle**
   (current suite: 33 tests), consistent with the INSTALL.md staleness doctrine.

Conclusion: production contract correct; former failure = test-environment
assumption; no assertion was weakened to achieve green (the fix strengthened
the suite by adding the GPU-mirror test).

## 4. Full smoke-suite result [EXEC E6]

`Ran 33 tests … OK` — classes: `TestVRAMPolicy` (5), `TestSharedLogic` (5),
`TestOrchestrator` (6), `TestSlidingChunkGuards` (3), `TestMultiLayerProbe`
(2), `TestDualKeyPolicy` (5), `TestTrainingSetAssembly` (3), `TestLeakFreeCV`
(3), `TestAugmentation`-related cases within `TestSharedLogic`. 0 skipped —
the previous silent skip of the CV tests (caused by the stub lacking
`torch.Tensor`, fixed in commit `5ac5e43`) no longer occurs.

## 5. Defect reproduction log (audit harness `audit/repro_findings.py`) [EXEC E9]

The harness imports the **unmodified** `pipeline.py` and drives real
`InjectionDetectionPipeline.run` / `TextClassifierLayer.score` /
`HiddenStateProbeLayer.load` with stub layers. Verbatim key outputs:

**F-01 — unresolved dual-key pending:**
```
=== F-01: unresolved dual-key pending (config interaction) ===
blocked=False  reason='Securely passed all active protection layers.'
metadata: pending=True resolved=None
VERDICT: REACHED GENERATION despite unresolved ambiguous-band flag
control (escalate=0.0): blocked=False resolved='L2_PASS'
```
(Config used: `l1_dual_key=True, l1_block_threshold=0.85,
l1_escalate_threshold=0.90`, L1 returns p=0.88 on a single chunk, no
untrusted context, probe attached.)

**F-02 — total L1 guard failure reads as SAFE:**
```
=== F-02: total L1 guard failure degrades to SAFE ===
decision='SAFE' prob=0.0  (guard raised on every chunk)
VERDICT: silent SAFE on total guard failure — run()'s fail-closed L1 handler never sees an exception
```

**F-03 — incomplete artifact fingerprint:**
```
=== F-03: probe load() ignores max_length/dtype/quant fingerprint ===
after load(): pooling='last' (artifact restored) | max_length=1024 (artifact recorded 512 — IGNORED)
VERDICT: load() raised no error for quantized_4bit=True / max_length=512 mismatch
```

**F-04 — evaluator/production predicate divergence:**
```
p1=0.87 chunks=1 alert=1: production_hard=False evaluator_hard=True  <-- DIVERGES
p1=0.97 chunks=1 alert=1: production_hard=True evaluator_hard=True
p1=0.87 chunks=5 alert=2: production_hard=True evaluator_hard=True
p1=0.87 chunks=5 alert=1: production_hard=False evaluator_hard=False
p1=0.90 chunks=1 alert=0: production_hard=False evaluator_hard=True  <-- DIVERGES
```

## 6. Proposed tests (NOT executed — for approval under PATCH_PLAN)

Regression tests for confirmed defects (all offline, stub-based, mirroring
`tests/smoke_offline.py` style):

| ID | Test | Establishes |
|---|---|---|
| T-01 | `run()` with `l1_dual_key=True, l1_escalate_threshold=0.90`, L1 p=0.88 single chunk, no ctx: expect **blocked** (pending must never reach generation unresolved) | F-01 fix |
| T-02 | L1 whose every guard raises on every chunk: expect `run()` to **block** with an L1-fault reason when `fail_closed=True`; expect pass only when `fail_closed=False` | F-02 fix |
| T-03 | `load()` of an artifact whose `max_length`/`compute_dtype`/`quantized_4bit` conflict with the runtime: expect `ValueError` (or explicit warning-level contract, if chosen) | F-03 fix |
| T-04 | Generation callback raising: expect block iff `fail_closed` | LOGIC_MATRIX §3 row 2 (currently untested) |
| T-05 | Judge returning `COMPROMISED` / `ERROR` through `run()`: expect block paths | §3 rows 4–5 (currently untested) |
| T-06 | Boundary table: p1 ∈ {0.849, 0.85, 0.949, 0.95}, p2 == threshold exactly | inclusive-`>=` semantics pinned |
| T-07 | `probe_threshold=1.5` and `l1_block_threshold=2.0` at construction: expect validation error (or explicit documented no-op) | F-12 |

Table-driven decision-logic test:

| ID | Test | Shape |
|---|---|---|
| T-08 | One parameterized table over `(p1, chunks, alerts, p2, gate, dual_key, probe, ctx, fail_closed) → expected(blocked, blocking_layer)` covering every row of LOGIC_MATRIX tables 1–3 (≈40 cases), driven against the real `run()` with stub layers | replaces scattered ad-hoc cases; catches future predicate drift like F-01/F-04 |

Property-based tests:

| ID | Test | Property |
|---|---|---|
| T-09 | Hypothesis over random L1/L2 score pairs at default config: `block(p1', p2')` is monotone non-decreasing in each score once dual-key state is fixed | detects threshold-band holes |
| T-10 | Random chunk/alert counts: L1 verdict invariant to trailing benign windows once 2 alert windows exist | multi-chunk policy |

Tiny end-to-end smoke (still offline):

| ID | Test | Shape |
|---|---|---|
| T-11 | `run()` with stub layers + a canned `generate_fn` returning a canary echo / suspicious text / clean text, asserting full `PipelineResult` field consistency (`blocked`, `reason`, `layer_scores` keys, `response` presence iff allowed) | end-to-end field contract |

Separate real-hardware validation plan (Kaggle 2×T4 — cannot run here):

1. Fresh session: `verify.py` from pinned commit `34c4d52`; confirm 33/33 suite.
2. Train the multi-layer probe (`--layers 12,16,20,24 --notinject`); record
   OOF AUC, threshold-vs-in-sample gap, `[balance]` composition.
3. `evaluate_probe.py` + `evaluate_end_to_end.py` + `benchmark.py`
   (post-F-04 fix) on held-out splits; compare per-layer attribution against
   serving behavior on the 4 demo scenarios.
4. Live OOM/VRAM checks: `nvidia-smi` per-GPU residency after one pipeline
   build; dtype actually fp16 (`next(model.parameters()).dtype`); confirm no
   bf16 on CC 7.5.
5. Fingerprint negative test on hardware: load the fp16 artifact into a
   `load_in_4bit=True` pipeline → expect refusal (post-F-03 fix).

**Explicit disclaimer:** none of the above proposed tests has been executed;
they await approval per the audit rules.
