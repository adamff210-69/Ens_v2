# RELEASE_NOTES.md — R5 verification & release closure

Recorded 2026-09-11 per the reviewer's release-handover checklist (review §7).
Every number below was measured in this session, not carried over.

## Commits & branch

| Item | Value |
|---|---|
| Branch | `arena/01a08bcc-ens-v2` |
| Full artifact commit (code) | `b7f36c2f36df0adb87d4b2251f140e255aa3451c` |
| Documentation/release commit | the commit containing this file (INSTALL.md pin + AUDIT.md addendum; SHA via `git log -1`) |
| INSTALL.md pin | `b7f36c2f36df0adb87d4b2251f140e255aa3451c` |
| Remote verification | `git ls-remote origin arena/01a08bcc-ens-v2` must end in the documentation commit SHA; the push output shows `.. -> arena/01a08bcc-ens-v2` |

Historical pins (superseded — do not use as the current release):
`a0e7cad` (F-04 shared predicate), `21a1fdc` (Batch B/C), `0ad186e` (its doc pin).

## Artifact file hashes (SHA-256)

| File | sha256 |
|---|---|
| `pipeline.py` | `a19d118323c035ce3373f739499677144701a360074b554f33a5e136296a9aaa` |
| `train_probe.py` | `ad87878b51cf317783814d39c3c7b867b503b32b74fe8520b0968947f813f90c` |
| `benchmark.py` | `67b923f9a8c1a930957a9a3d8d9f8ce9dd9a4fdfed1e60dafcb3632e3a51b688` |
| `tests/smoke_offline.py` | `0a15b16b82594acc791f31c6ff51d13787746805301115cd17d7aa9e682a8531` |
| `verify.py` (installer) | `e155ce9c7997921e710457dc3954fcf574c5eceeeee8d7ddab9e37bde8ca0eed` |

The four runtime-file digests are also embedded in `verify.py`'s `EXPECTED`
table; the installer re-verifies them before writing anything.

## Environment (validation venv, recreated this session)

| Component | Version |
|---|---|
| Python | 3.11.2 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |

No GPU and no torch/transformers in the audit sandbox: all results below are
offline policy/plumbing validation (mocked hardware), not model runs.

## Test summaries (exact counts)

Repository (full tree, `/home/user/Ens_v2`):

```text
Ran 67 tests in ~3.2s
OK            (0 failed, 0 skipped)
```

Extracted bundle (clean directory, `verify.py` from the artifact commit):

```text
Ran 67 tests in ~3.4s
OK (skipped=2)   (0 failed)
```

The two bundle skips are packaging limitations, not policy gaps:
`test_f04_evaluators_call_shared_predicate` (evaluator files intentionally
not shipped) and `test_f07_readme_documents_prefix_probe_gap` (README not
shipped). The underlying F-04 truth table and all policy tests still run in
the bundle. The full-repo run keeps every check mandatory.

Repro harness:

```text
python audit/repro_findings.py  -> exit 0, ALL AUDIT FINDINGS F-01..F-04 VERIFIED FIXED
```

## Reproducibility evidence

Fresh checkout (local clone of the repo at the artifact commit, clean tree):

```text
git clone ... && git checkout b7f36c2f36df0adb87d4b2251f140e255aa3451c
python tests/smoke_offline.py   -> Ran 67 tests, OK (0 skipped)
```

Fresh bundle extraction (separate clean directory):

```text
python verify.py                -> exit 0; payloads verified, extracted,
                                   suite: Ran 67 tests, OK (skipped=2)
```

Installer matrix (measured manually, exit codes in parentheses):
valid bundle + absent files → install (0); valid + identical → no-op (0);
valid + locally edited destination → refuse, destination bytes preserved,
nothing written (4); `--force` → explicit overwrite (0); corrupted digest →
nothing written (2); unknown flag → usage error (64).

## Working-tree status at recording time

`git status --short` empty (clean tracked tree) on the artifact commit,
before the documentation commit.

## What this release does NOT yet prove

- No real-model / GPU results: Kaggle 2×T4 training run (F-09
  recalibration), real-hardware checklist `audit/TEST_RESULTS.md` §6.
- No live end-to-end TPR/FPR at the pinned revisions; gate-table numbers in
  README predate the F-04 attribution change and must be recomputed before
  publishing (see the gate-table caveat).
- F-10 generation timeouts and deferred F-13/F-14 remain open.
