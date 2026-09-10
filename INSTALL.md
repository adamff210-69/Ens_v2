# Getting the files onto Kaggle

The pipeline files live in the Arena workspace, **not** on your Kaggle box. A
fresh Kaggle session starts with an empty `/kaggle/working`, so
`python3 train_probe.py` fails with `No such file or directory`. Pick one route
below — route A is the fastest.

---

## Route A — one cell, no upload (recommended)

`verify.py` is hosted at a temporary public link. Requires **Internet: ON** in
the notebook (right sidebar -> Settings -> Internet).

    %cd /kaggle/working
    !curl -sSL -o verify.py https://paste.rs/aljgi
    !sha256sum verify.py

`sha256sum` must print
`fef18afa029211e35746bbb669277ab57d961a16cbab0e0ad5893170aeeb0b15`. Then:

    !python verify.py

Which extracts the four files, verifies their hashes, runs the 26-test offline
suite, and prints the training commands.

If `curl` is blocked or returns an empty file, use Python's own HTTP client:

    import urllib.request
    urllib.request.urlretrieve("https://paste.rs/aljgi", "/kaggle/working/verify.py")

If both fail, Kaggle's proxy is blocking the host - fall back to Route B.

## Route A2 — patch only the test file (no internet needed)

If the four files are already installed and verified (they were), only
`tests/smoke_offline.py` needs fixing. One cell:

    p = "/kaggle/working/tests/smoke_offline.py"
    s = open(p).read()
    start = s.index("    def test_cpu_fallback(self):")
    end = s.index("    def test_turing_is_fp16(self):")
    ...

(full cell: `patch_test_cpu_fallback.py`, also hosted at
https://paste.rs/RWoOo, then run `!python tests/smoke_offline.py`)

Why: `test_cpu_fallback` asserted a float32 dtype while *assuming* the host had
no GPU. Kaggle sessions with the accelerator ON have CUDA, so `t4_dtype()`
correctly returned fp16 and the assertion failed. The test now mocks a CPU host
instead of assuming one, and a companion `test_gpu_host_reports_its_own_dtype`
covers the GPU case.

## Route B — download `verify.py`, upload it, run it

`verify.py` is a single stdlib-only file with every source file embedded
(base64 + SHA-256 pinned). No pip, no network, no git.

1. Download **`verify.py`** (138 KB) from this workspace.
2. Kaggle notebook → right sidebar → **Data → Upload** → drop it in (it becomes
   `/kaggle/input/<your-dataset>/verify.py`), *or* use the notebook's own file
   upload widget which lands it in `/kaggle/working`.
3. Run one cell:

```python
# if you uploaded it as a Kaggle dataset (read-only /kaggle/input):
!cp /kaggle/input/*/verify.py /kaggle/working/ 2>/dev/null
%cd /kaggle/working
!python verify.py
```

Expected output:

```
extracting 4 files into /kaggle/working
  [OK ] pipeline.py                   54537 bytes  sha256 1b6e9200b85e3095
  [OK ] train_probe.py                 9580 bytes  sha256 6a5d5225fcde4cd7
  [OK ] benchmark.py                  10370 bytes  sha256 d60b8a4bd1246830
  [OK ] tests/smoke_offline.py         9580 bytes  sha256 e2672aa4822784ba
all checksums match
running offline test suite...
smoke suite PASSED
```

That one cell both installs the files **and** proves they work.

---

## Route C — they already exist somewhere in this session

If you ran the earlier notebook in *this same session*, the files may still be
around. Find them before re-uploading:

```python
!find /kaggle -name "train_probe.py" -o -name "benchmark.py" -o -name "pipeline.py" 2>/dev/null | grep -v site-packages
```

Then copy what you find into the working directory:

```python
!mkdir -p /kaggle/working && cp <path-from-above> /kaggle/working/
```

> Note: files under `/kaggle/input/` are read-only. Always copy into
> `/kaggle/working/` before running anything.

---

## Route D — upload the loose files

If you prefer separate uploads: download `pipeline.py`, `train_probe.py`,
`benchmark.py` from this workspace, then in Kaggle either upload them as one
dataset or paste them into notebook cells:

```python
%%writefile /kaggle/working/pipeline.py
# ...paste the file contents here...
```

Repeat per file. Same result, more clicking.

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
