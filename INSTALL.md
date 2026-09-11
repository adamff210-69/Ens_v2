# Getting the files onto Kaggle

The pipeline files live in this Git repository, **not** on your Kaggle box. A
fresh Kaggle session starts with an empty `/kaggle/working`, so
`python3 train_probe.py` fails with `No such file or directory`. Pick one
route below — route A is the fastest and the only one that stays reproducible.

> **Why not paste links?** Earlier versions of this doc shipped the files via
> anonymous `paste.rs` URLs. Don't do that: paste links expire (your notebook
> becomes unreproducible within weeks), the code is mutable and unattributed,
> and checksums printed by a downloaded script verify only *its own* embedded
> copies — circular trust. Everything below fetches from the pinned Git repo
> or from files you uploaded yourself.

---

## Route A — git clone at a pinned commit (recommended)

Requires **Internet: ON** in the notebook (right sidebar -> Settings ->
Internet). Pin the commit SHA so a future you (or a reviewer) gets exactly
the audited code:

    %cd /kaggle/working
    !git clone https://github.com/adamff210-69/Ens_v2.git
    %cd Ens_v2
    !git checkout 21a1fdc29ef52caff2f07e92b8ecead1b82370ed   # audited artifact
                                                            # (audit fixes
                                                            # F-01..F-08, F-11,
                                                            # F-12, D-1 incl.)
    !python verify.py

Which verifies each embedded payload against a pinned SHA-256 **before
writing anything**, extracts the four runtime files only if all digests
match (then re-hashes them from disk), and runs the 48-test offline suite.
A mismatch exits non-zero and leaves your local files untouched.

If `git clone` is blocked by Kaggle's proxy, fall back to Route B.

## Route B — download `verify.py` from GitHub, upload it, run it

`verify.py` is a single stdlib-only file with every source file embedded
(gzip + base85, SHA-256 pinned per file). No pip, no network, no git.
Because it is fetched from the pinned repository — not an anonymous paste —
you can audit it before uploading.

1. Download **`verify.py`** (~63 KB) from the repo at the pinned commit:
   `https://github.com/adamff210-69/Ens_v2/blob/<commit>/verify.py`
   (raw link -> right-click -> Save), or `git clone` locally first.
2. Kaggle notebook → right sidebar → **Data → Upload** → drop it in (it
   becomes `/kaggle/input/<your-dataset>/verify.py`), *or* use the
   notebook's own file upload widget which lands it in `/kaggle/working`.
3. Run one cell:

```python
# if you uploaded it as a Kaggle dataset (read-only /kaggle/input):
!cp /kaggle/input/*/verify.py /kaggle/working/ 2>/dev/null
%cd /kaggle/working
!python verify.py
```

Expected output:

```
verifying 4 embedded payloads (in memory, nothing written yet)
  [OK ] pipeline.py                  65612 bytes  sha256 fe0ff0903eb71ae0
  [OK ] train_probe.py               14109 bytes  sha256 f7b8622edf6147a4
  [OK ] benchmark.py                 11355 bytes  sha256 e3a90fe35e3ec82a
  [OK ] tests/smoke_offline.py       38804 bytes  sha256 71acb03fa1ea5453
all checksums match - extracting into /kaggle/working
extraction verified from disk
running offline test suite...
smoke suite PASSED
```

That one cell both installs the files **and** proves they work. Since the
installer verifies *before* writing, re-running it in a directory with local
edits reports the mismatch and exits without reverting your changes.

---

## Route C — they already exist somewhere in this session

If you ran the earlier notebook in *this same session*, the files may still
be around. Find them before re-uploading:

```bash
!find /kaggle -name "train_probe.py" -o -name "benchmark.py" -o -name "pipeline.py" 2>/dev/null | grep -v site-packages
```

Then copy what you find into the working directory:

```bash
!mkdir -p /kaggle/working && cp <path-from-above> /kaggle/working/
```

> Note: files under `/kaggle/input/` are read-only. Always copy into
> `/kaggle/working/` before running anything.

## Route D — upload the loose files

If you prefer separate uploads: download `pipeline.py`, `train_probe.py`,
`benchmark.py` from the pinned repo commit, then in Kaggle either upload
them as one dataset or paste them into notebook cells:

```python
%%writefile /kaggle/working/pipeline.py
# ...paste the file contents here...
```

Same result, more clicking.

---

## Always finish with this cell

Whatever route you used, confirm you are not running a stale copy from an
earlier session — this is the check that would have caught the immediate
problem:

```python
import sys; sys.path.insert(0, "/kaggle/working")
import pipeline
print("pipeline version:", pipeline.__version__)      # expect 2.0.0
print("has L1 ensemble:", "models" in pipeline.TextClassifierLayer.__init__.__code__.co_varnames)
print("has multi-layer probe:", "layers" in pipeline.HiddenStateProbeLayer.__init__.__code__.co_varnames)
print("has max_chunks guard:", "max_chunks" in pipeline.TextClassifierLayer.__init__.__code__.co_varnames)
```

All four lines printing `2.0.0 / True / True / True` means you have the v2 code.
Any `False` means an old copy is shadowing the new one — check `pipeline.__file__`.

---

## Order of operations

The GPU rules from earlier still hold; do not skip the session boundaries.

| Step | Session | Model resident? |
|---|---|---|
| `verify.py` install + smoke suite | any (CPU fine) | no |
| `train_probe.py` v2 | fresh session | Qwen loads here |
| `benchmark.py` | *same* session as training (reuses the load) | yes |
| serve `InjectionDetectionPipeline` | **fresh session** | yes |

Never run `!python evaluate_*.py` / `benchmark.py` with a live pipeline in the
same session — that is the OOM path.

---

## Historical note: the `test_cpu_fallback` patch

Earlier sessions carried a downloaded one-off patch for
`tests/smoke_offline.py::test_cpu_fallback`. That test used to assert a
float32 dtype while *assuming* the host had no GPU; on a Kaggle session with
the accelerator ON, `t4_dtype()` correctly returned fp16 and the assertion
failed. The fix — mock a CPU host instead of assuming one, plus a companion
`test_gpu_host_reports_its_own_dtype` for GPU boxes — is **merged into the
repo**. No patch download is needed at or after the pinned commit above; if a
session ever asks you to fetch a patch script, stop and diff it instead.
