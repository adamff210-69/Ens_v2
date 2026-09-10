# AUDIT.md — Independent Audit: 3-Layer Prompt Injection Detection Pipeline

**Audit date:** 2026-09-10 · **Auditor:** independent code/ML-security review (Arena session)
**Repository:** `/home/user/Ens_v2` · **Branch:** `arena/01a08bcc-ens-v2` · **HEAD:** `34c4d5287f59818c62604fc6c74b327d7f94728f`
**Working tree:** clean (`git status --porcelain` empty) · **Remote:** `https://github.com/adamff210-69/Ens_v2.git`
**Rules honored:** no source/test/config modification during this audit; no model downloads, training, or remote scripts executed; new files only under `audit/`.

**Evidence tags used throughout:** `[EXEC]` confirmed by execution · `[INSPECT]` confirmed by code inspection · `[HYPOTHESIS]` needs verification · `[NOT-TESTED]` not exercised.

**Specification caveat (Rule 9):** the audit brief's specification field was left as a template placeholder. The stated design intent is taken from `README.md` + `PROJECT_STATUS.md` (the project's own design documents). Requirements not pinned down by those documents are marked **"Cannot determine — unspecified"** in the alignment matrix rather than assumed.

---

## 1. Executive summary

The repository implements, in one 1388-line module (`pipeline.py` v2.0.0), a
three-layer fail-closed prompt-injection defense around a target 7B LLM
(Qwen2.5-7B-Instruct), sized for Kaggle 2× Tesla T4: an L1 sliding-window
DeBERTa guard ensemble, an L2 logistic probe on LLM hidden states with
FPR-budget calibration, and an L3 output auditor (canary + regex + self-judge).
Surrounding CLIs train, evaluate, and benchmark the stack, and a stdlib-only
self-extracting installer (`verify.py`) ships the files to Kaggle.

**Overall verdict: the architecture matches its advertised design, the headline
routing policy (always-on L2, dual-key ambiguity handling, fail-closed faults)
is implemented and mostly correct — but the audit found one reachable
security-relevant logic gap, one silent-degradation path, an incomplete
artifact-fingerprint check, and an evaluator whose decision rule diverges from
production.** None require a redesign; all are small, localized fixes
(see `PATCH_PLAN.md`).

Headline results:

| # | Severity | Finding (one line) | Evidence |
|---|---|---|---|
| F-01 | **High** | Dual-key "pending" flag can be left unresolved → request reaches generation with an ambiguous-band L1 flag never confirmed (non-default config) | [EXEC] |
| F-02 | **High** | Total L1 guard failure degrades to `SAFE 0.0` instead of tripping the fail-closed handler | [EXEC] |
| F-03 | **Medium** | Probe `load()` ignores `max_length` / `compute_dtype` / `quantized_4bit` fingerprint fields it saves | [EXEC] |
| F-04 | **Medium** | `evaluate_end_to_end.py` hard-block predicate ≠ production `run()` rule → published gate metrics don't mirror serving | [EXEC] |
| F-05 | Low | `benchmark.py` "stack(dual-key)" config is algebraically identical to "always-on" | [INSPECT] |
| F-06 | Low | `train()` docstring misstates threshold selection ("LARGEST" vs code's lowest-qualifying) | [INSPECT] |
| F-07 | Medium | L2 sees first 1024 tokens; generation sees 4096 → L2-blind zone in long inputs (warning-only) | [INSPECT] |
| F-08 | Low | Missing probe ⇒ silent downgrade to L1+L3 (log line only); spec silent on whether this should block | [INSPECT] |
| F-09 | Low | Deployed probe threshold 0.90 was selected on the test split (self-disclosed in README) | [INSPECT] |
| F-10 | Low | No timeout anywhere; a hung generation hangs `run()` indefinitely | [INSPECT] |
| F-11 | Info | `verify.py` overwrites its target files in CWD *before* checksum verification (currently in sync ⇒ idempotent) | [EXEC] |
| F-12 | Low | No range validation on thresholds (`probe_threshold=1.5` silently disables L2 blocking) | [INSPECT] |
| F-13 | — | L1 label allowlist vs PIGuard label vocabulary | [HYPOTHESIS] |
| F-14 | — | Constant training feature ⇒ `StandardScaler` divide-by-zero ⇒ NaN probabilities | [HYPOTHESIS] |
| F-16 | Low | `system_prompt` input is never audited by any layer (trusted-by-construction assumption, unspecified) | [INSPECT] |

The previously reported failure (`tests/smoke_offline.py:104 test_cpu_fallback`,
26-test run) **does not reproduce**: that artifact predates the merged test
fix; the current suite (33 tests) passes with the CPU-fallback contract
verified under mocks — see `TEST_RESULTS.md` §4 and Phase E details there.

---

## 2. Reproducible project state [EXEC unless noted]

| Item | Value |
|---|---|
| Repo root / audit cwd | `/home/user/Ens_v2` |
| Git | branch `arena/01a08bcc-ens-v2`, HEAD `34c4d52`, base commit `c3b742a`; clean tree |
| Python | 3.11.2 (system, externally-managed) |
| Audit interpreter | `/home/user/.venv/bin/python` (created in a prior session, *outside* the repo): numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1, joblib 1.6.0. **No torch, transformers, datasets, accelerate, bitsandbytes** — model-dependent code paths cannot execute here |
| GPUs | none: `nvidia-smi` absent, no `/dev/nvidia*`, no nvidia kernel modules. All hardware policy verified only via mocks |
| Secrets | no `.env`, `*.key`, token files, or probe artifacts (`*.joblib`) present in the repo |
| Duplicate `pipeline.py` | none inside the repo (scratch extractions exist only under `/tmp/audit_bundle`, `/tmp/verify_e2e`, `/tmp/final_check` — outside version control) |
| Stale imports / shadowing | none: single module per role at repo root; `__pycache__` now gitignored |

Key file hashes (sha256, full values recorded in `TEST_RESULTS.md`):
`pipeline.py fcbf59e6…`, `train_probe.py f7b8622e…`, `benchmark.py 10672b7d…`,
`tests/smoke_offline.py 0d405a7a…`, `verify.py a6bc414f…`.

**Provenance of executed code:** everything executed in this audit ran from the
repository checkout itself. `verify.py` embeds gzip+base85 copies of
`pipeline.py`, `train_probe.py`, `benchmark.py`, `tests/smoke_offline.py` with
pinned SHA-256s; extraction into `/tmp/audit_bundle` produced files
**byte-identical to the repo copies** (hash-compared, exit 0), so the bundle
and the reviewed sources are the same version. The historical Kaggle runs
(referenced in the audit brief) executed bundle copies fetched from paste.rs
links; those links were removed from `INSTALL.md` in commit `34c4d52` in favor
of a pinned-commit `git clone`.

**Bootstrap side-effect (F-11):** `verify.py` writes its four files into the
CWD *unconditionally*, then checks hashes and exits 2 on mismatch. Re-running
it in the repo root today is content-neutral (in sync), but after any future
local edit it would silently revert the edit before reporting the mismatch.
Understood; not re-run inside the repo during this audit.

**Intended launch commands** (from README/INSTALL):
`python train_probe.py --model "Qwen/Qwen2.5-7B-Instruct" --layers 12,16,20,24 …` →
`python benchmark.py …` → serve via `InjectionDetectionPipeline(...)` +
`pipe.make_generate_fn()` inside one notebook session. Required external
resources: HF Hub models `Qwen/Qwen2.5-7B-Instruct`,
`ProtectAI/deberta-v3-base-prompt-injection-v2` (optionally `leolee99/PIGuard`),
datasets `deepset/prompt-injections`, `jackhhao/jailbreak-classification`,
`leolee99/NotInject`, `rubend18/ChatGPT-Jailbreak-Prompts`. No revisions are
pinned anywhere in code [INSPECT] — model/dataset versions drift silently
between sessions (design question D-1).

---

## 3. What the project actually builds

A request-time security wrapper: given `(user_input, system_prompt,
untrusted_context, generate_fn)`, `InjectionDetectionPipeline.run()`
(`pipeline.py:1071–1265`) returns a `PipelineResult(blocked, reason, response,
layer_scores, metadata, execution_time_ms)`. Nothing is generated until the
input gates pass; the generated output is then audited before being attached
to the result [INSPECT].

**Training flow** (`train_probe.py`):

```
--datasets specs ─┐                        ┌─> HiddenStateProbeLayer(model, layers)
NotInject ────────┼─> build_training_set ─>│     └─ _init_base_llm: load 7B (fp16/4bit)
(hard negatives)  ┘   (validate columns,   │
                      balance 1:1, protect │   extract_activations(texts)
                      hard negatives,      │     └─ forward pass, pool hidden states
                      stratify sources)    │        per layer, concatenate
                                           │   train(): StandardScaler + LogReg(C=0.5)
augment_injection_data ───────────────────>│     StratifiedGroupKFold CV (groups=source row)
(delimiter/base64/7 evasion families,      │     OOF-prob FPR-budget threshold
 returns group ids)                        └─> save(): joblib {probe, scaler, layers,
                                                                   threshold, fingerprints…}
```

**Inference/control flow** (`run()`):

```
wrap_untrusted(user, ctx) ──> L1.score (sliding windows, per-guard max)
  ├─ p1≥block & unambiguous(p1≥0.95 ∨ multi∧alerts≥2) ──────────> BLOCK(L1)
  ├─ p1≥block & ambiguous & (dual_key ∧ probe) ──> pending ──┐
  ├─ p1≥block & ambiguous & ¬(dual_key ∧ probe) ────────────> BLOCK(ambiguous)
  ├─ L1 raised & fail_closed ────────────────────────────────> BLOCK(L1 fault)
  └─ pass ──┐
escalate = (p1 ≥ l1_escalate_threshold) ∨ ctx ──────────────────┐
  ├─ escalate ∧ probe: L2.score ≥ active_thr ──────────────> BLOCK(L2)
  │                    L2.score <  thr ∧ pending ──> resolved=L2_PASS
  │                    L2 raised & fail_closed ───> BLOCK(L2 fault)
  ├─ escalate ∧ ¬probe: log only (F-08) ─────────────────────┐│
  └─ ¬escalate: skip L2 (F-01 if pending) ───────────────────┼┘
generate_fn(canary-augmented system prompt) ── except ──> BLOCK(fail_closed)
  └─> L3: canary echo? ──> BLOCK(exfiltration)
      regex heuristics flag? ──> self-judge on same 7B
         COMPROMISED ──> BLOCK · ERROR∧fail_closed ──> BLOCK · SKIPPED ──> pass(warn)
      ──> response released
```

### Module inventory

| File (lines) | Responsibility | Status |
|---|---|---|
| `pipeline.py` (1388) | All 3 layers, `VRAMManager`, augmentation, `PipelineResult`, orchestrator | Implemented; core |
| `train_probe.py` (338) | Multi-dataset load/validate/balance → augment → probe train+calibrate → save | Implemented |
| `evaluate_probe.py` (160) | L2-only held-out metrics, attack buckets, threshold sweep | Implemented |
| `evaluate_end_to_end.py` (192) | L1+L2 gate-policy comparison on held-out split | Implemented; **predicate diverges from production (F-04)** |
| `benchmark.py` (252) | One-load multi-dataset benchmark, bootstrap CIs, CSV dump | Implemented; dual-key config degenerate (F-05) |
| `kaggle_notebook.py` (179) | Demo cells: build pipeline, 4 scenarios | Implemented (demo) |
| `verify.py` (51 KB) | Self-extracting SHA-pinned installer | Implemented; overwrite-before-verify caveat (F-11) |
| `build_verify.py` (150) | Regenerates `verify.py` from current sources | Implemented |
| `tests/smoke_offline.py` (617) | 33 offline tests (stubs for torch/transformers) | Implemented |
| `patch_test_cpu_fallback.py` (40) | Historical one-off patch — **already merged**, superseded | Dead artifact (kept for provenance) |

**Implemented vs optional vs stubbed vs unreachable:**

- *Optional (flag-gated):* L1 ensemble (`l1_models`), multi-layer probe (`probe_layers`), `load_in_4bit`, `l1_dual_key`, `probe_threshold` override, `--notinject`, `--pos_variants`.
- *Rejected-by-design:* a separate judge model — `judge_model_name ≠ llm_model_name` only logs a warning and still reuses the target LLM (`pipeline.py:1019–1025`) [INSPECT].
- *Legacy-compat aliases (reachable, exercised by tests):* `t4_dtype`, `gpu_max_memory`, `n_gpus`, `empty_cache`, `wrap_untrusted_context`, `extract_all`, `probe_classifier`, `fpr_calibrated_threshold`, `create_canary_token`, `is_canary_exfiltrated`, `matches_suspicious_patterns`, `execute_self_judge`, `execute_security_augmentation`, `target_fpr` kwarg.
- *Unreachable inside the full pipeline:* `OutputCheckLayer.judge_response` returning `SKIPPED` (judge is always attached when built via `InjectionDetectionPipeline`); the `_FALLBACKS` L1 chain fires only if the primary spec fails to load.
- *Not implemented:* generation timeouts, threshold range validation, artifact dtype/quant fingerprint enforcement, system-prompt auditing, multi-turn/session state.

### Per-layer facts demanded by the brief

| Layer | Class/function | Callers | Runs | Changes decision? | Requires | On failure |
|---|---|---|---|---|---|---|
| L1 | `TextClassifierLayer.score` (`pipeline.py:381`) | `run()` only | Always (first stage) | Yes — hard block | HF classifier(s); GPU optional | Whole-`score()` exception ⇒ fail-closed block; **per-guard exception ⇒ vote 0.0, no error (F-02)** |
| L2 | `HiddenStateProbeLayer.score` (`pipeline.py:824`) | `run()` | Only if `escalate ∧ probe is not None` | Yes — block ≥ thr | Trained artifact + resident 7B | Exception ⇒ fail-closed block; **missing probe ⇒ skip + log (F-08)** |
| L3 | `OutputCheckLayer.canary_leaked/is_suspicious/judge_response` | `run()` after generation | Always after generation | Yes — block on canary/COMPROMISED/ERROR∧fail_closed | Same 7B | Judge exception ⇒ `ERROR` ⇒ fail-closed block |
| Generation | `make_generate_fn` closure (`pipeline.py:1032`) | `run()` | Only if caller supplies it | N/A (audited after) | Resident 7B | Exception ⇒ fail-closed block |

---

## 4. Alignment matrix (vs README/PROJECT_STATUS stated design)

| Requirement (as stated) | Code evidence | Actual behavior | Status | Required change |
|---|---|---|---|---|
| L1 sliding-window DeBERTa guard, GPU 1, multi-chunk escalation logic | `TextClassifierLayer` (`pipeline.py:215–433`) | As described; window cap 64, overlap clamp, tail coverage | **Aligned** | — |
| L2 probe on intermediate layer, threshold by FPR budget ≤1%, fingerprint-checked `load()` | `HiddenStateProbeLayer` (`pipeline.py:439–851`) | Model+layer fingerprints enforced; **max_length/dtype/quant saved but not checked** | **Partially aligned** | F-03 |
| L3 canary + regex + self-judge reusing same LLM, hardened prompt | `OutputCheckLayer` (`pipeline.py:857–972`) | As described; verdict from first 80 chars | **Aligned** | — |
| Fail-closed everywhere (L1 crash, L2 crash, generation fault, judge ERROR/empty) | `run()` handlers (`pipeline.py:1145–1150, 1186–1192, 1210–1214, 1236–1246`) | True for raised exceptions; **guard-level faults inside L1 become SAFE scores instead of exceptions (F-02)** | **Partially aligned** | F-02 |
| Always-on L2 default (`l1_escalate_threshold=0.0`) | `pipeline.py:1156–1159`, test `test_always_on_l2_default` | Default 0.0 confirmed in code and test | **Aligned** | — |
| Dual-key: ambiguous L1 band requires L2 confirmation | `pipeline.py:1116–1144`, `TestDualKeyPolicy` | Works at default config; **pending flag can go unresolved under `escalate > l1_prob` (F-01)** | **Partially aligned** | F-01 |
| Probe artifact refuses model/layer mismatch | `load()` (`pipeline.py:780–822`) + round-trip test | Confirmed | **Aligned** | — |
| VRAM policy: fp16 on T4, bf16 on CC≥8, fp32 CPU; per-GPU caps `{0:'13.5GiB',1:'12.0GiB'}` | `VRAMManager` (`pipeline.py:84–136`) | Verified under mocks [EXEC]; **real hardware NOT-TESTED** | **Aligned (mock-verified only)** | real-HW validation plan |
| Honest held-out evaluation, never train on test | `train_probe.py` defaults use `:train:` splits; evals use `:test:` | Split discipline holds in code; **deployed threshold 0.90 was picked on the test split (self-disclosed)** | **Partially aligned** | F-09 (recalibrate on train-val slice) |
| Same `<untrusted_context>` wrapping at train/score/generation | `wrap_untrusted` used by augmentation templates, `run()`, generation | Consistent | **Aligned** | — |
| One pipeline instance per session / OOM doctrine | README troubleshooting; `max_memory` caps | Code supports it; operational only | **Aligned** | — |
| L2 contribution measured end-to-end | `evaluate_end_to_end.py` | Exists, but its hard-block rule ≠ production (F-04) | **Partially aligned** | F-04 |
| Bootstrap CIs in benchmarking | `bootstrap_ci` (`benchmark.py:110`) | Implemented, seeded | **Aligned** | — |
| Number of layers = 3 | three layer classes, one orchestrator | Confirmed: 3 detection layers + generation + canary instrumentation; docs' "4-stage" phrasing in the brief is not used by the repo itself | **Aligned** | — |
| Citation-grounded explanations | — | Not claimed anywhere in code or docs | **Cannot determine — unspecified** (not a feature of this design) | — |
| Model/dataset revision pinning | — | No pins in code or requirements | **Cannot determine — unspecified** (D-1) | recommend pins |

Unanswered design questions (Rule 9): **D-1** model/dataset revisions unpinned;
**D-2** should a missing probe block instead of downgrade (F-08)?; **D-3** is a
caller-supplied `system_prompt` inside the threat model (F-16)?; **D-4** acceptable
FPR for the deployed operating point (artifact metadata says budget 0.01 while
0.90 operates at ~1.8% — self-noted in PROJECT_STATUS); **D-5** expected behavior
when L1 and L2 *disagree at the extremes* (L1 ≥0.95 blocks even if L2 would pass
— intended pre-emption, confirmed by `test_extreme_peak_blocks_without_l2`).

---

## 5. Closing assessment

1. **What the agent actually built:** a coherent, genuinely functional
   three-layer fail-closed injection-defense wrapper with training, evaluation,
   benchmarking, offline regression tests, and a self-verifying Kaggle delivery
   mechanism. The advertised architecture is real, not aspirational.
2. **What is definitely broken:** F-01 (unresolved dual-key pending reaches
   generation under non-default config), F-02 (total L1 guard failure reads as
   SAFE), F-03 (incomplete artifact fingerprint), F-04 (evaluator/production
   predicate divergence). All four reproduced by execution in `TEST_RESULTS.md` §5.
3. **What may be broken but needs verification:** F-13 (PIGuard label
   vocabulary vs the L1 label allowlist — needs one model download), F-14
   (NaN propagation from degenerate training features), F-07's practical impact
   (needs long-document attack eval on GPU), and everything hardware-dependent
   (fp16 paths, device placement, VRAM caps) — unverifiable without T4s.
4. **What does not match the core idea:** nothing structural. The deviations
   are *contract violations inside an otherwise aligned implementation*
   (fail-closed not holding for guard faults; fingerprint check partial) plus
   measurement tooling that doesn't mirror production (F-04/F-05).
5. **First three fixes:** ① fail-closed resolution of pending dual-key flags
   before generation (F-01); ② raise when every L1 guard fails on an input
   (F-02); ③ enforce `max_length`/dtype/quant fingerprints in `load()` (F-03).
   Details + regression tests in `PATCH_PLAN.md`.
6. **Not reviewed / not testable here:** real-model behavior of all four HF
   checkpoints; dataset content and label quality (no downloads permitted);
   Kaggle runtime behavior; bitsandbytes 4-bit path; network/proxy behavior of
   the delivery routes; any prior notebook state outside this repo.

See `LOGIC_MATRIX.md` for decision tables, `TEST_RESULTS.md` for executed
commands/logs, `PATCH_PLAN.md` for prioritized fixes. **No source code was
modified during this audit; awaiting approval before implementing fixes.**

---

## Addendum — 2026-09-10 (post-audit fixes approved and implemented)

Fixes 1–4 of `PATCH_PLAN.md` (findings F-01, F-02, F-03, F-04) were approved
and implemented in commit `b87ac4c5bc2600db7138fa5abd4f6996e960a1fe`, with
regression locks in `tests/smoke_offline.py::TestAuditSecurityFixes`
(suite: 39/39 OK). `audit/repro_findings.py` now exits 0 with all four
verdicts FIXED and the default-path control unchanged. Two deliberate
deviations from the approved patch text are recorded in the `PATCH_PLAN.md`
status note (timing-stamp-consistent `done(result)` in the F-01 gate;
F-04 regression test rewritten as source-parity + truth-table because the
proposed version was a tautology and the evaluator module cannot be imported
offline). `verify.py` was regenerated and re-validated end-to-end from an
extracted bundle. Findings F-05..F-14 and design questions D-1..D-5 remain
open per `PATCH_PLAN.md`.
