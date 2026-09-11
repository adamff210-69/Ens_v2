#!/usr/bin/env python3
"""Build a stdlib-only, self-extracting installer for the pipeline stack.

Output: verify.py, written next to this script

Payload is gzip-compressed then base85-encoded (3.4x smaller than raw base64),
so the single file is small enough to upload quickly or paste into a cell.
SHA-256 of the *uncompressed* bytes is pinned per file and checked on extract.
"""
import base64
import gzip
import hashlib
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "pipeline.py",
    "train_probe.py",
    "benchmark.py",
    "tests/smoke_offline.py",
]
# base85 alphabet: 0-9 A-Z a-z !#$%&()*+-;<=>?@^_`{|}~  (no quote, no backslash)
FORBIDDEN = set('"\'\\ \n\r\t')

blobs, digests = {}, {}
for rel in FILES:
    with open(os.path.join(HERE, rel), "rb") as fh:
        raw = fh.read()
    payload = base64.b85encode(gzip.compress(raw, 9)).decode("ascii")
    assert not (set(payload) & FORBIDDEN), f"{rel}: payload needs escaping"
    blobs[rel] = payload
    digests[rel] = hashlib.sha256(raw).hexdigest()

entries = []
for rel, payload in blobs.items():
    wrapped = "\n".join('        "%s"' % line for line in textwrap.wrap(payload, 88))
    entries.append('    "%s": (\n%s\n    ),' % (rel, wrapped))

expected = "\n".join('    "%s": "%s",' % (k, v) for k, v in digests.items())

TEMPLATE = '''#!/usr/bin/env python3
"""Self-extracting installer + verifier for the 3-layer injection pipeline.

Usage, in the directory you want the files written to (e.g. /kaggle/working):

    python3 verify.py

Verifies each embedded payload (pipeline.py, train_probe.py, benchmark.py,
tests/smoke_offline.py) against a pinned SHA-256 BEFORE writing anything,
then scans the destination: files that already exist and DIFFER from the
bundle are refused by default (local edits are preserved; re-run with
--force to overwrite them explicitly). Identical files are a no-op. Only
if every payload matches AND there are no unresolved conflicts does it
extract, re-hash the written files from disk, and run the offline test
suite — so neither a bad transfer nor a silent overwrite can go unnoticed.

Exit codes: 0 ok · 2 corrupt bundle · 3 post-write mismatch · 4 local
conflict (nothing written) · 64 bad CLI usage.

Stdlib only: no pip installs, no network, Python 3.8+.

    python3 verify.py            # refuse to overwrite differing local files
    python3 verify.py --force    # explicitly overwrite them
"""
import base64
import gzip
import hashlib
import os
import subprocess
import sys

# gzip + base85 payloads, keyed by destination path
FILES = {
__ENTRIES__
}

# sha256 of the DECOMPRESSED bytes
EXPECTED = {
__EXPECTED__
}


def main() -> int:
    force = "--force" in sys.argv[1:]
    unknown = [a for a in sys.argv[1:] if a != "--force"]
    if unknown:
        print("usage: python3 verify.py [--force]")
        print("  --force  overwrite local files that differ from the bundle")
        return 64

    # Phase 1 - verify FIRST, write SECOND (audit finding F-11). The old
    # installer overwrote files in the CWD before checking digests, so a
    # corrupt transfer clobbered local work before reporting itself. Now:
    # decode + hash in memory, compare all digests, continue only if every
    # payload matches.
    print("verifying %d embedded payloads (in memory, nothing written yet)"
          % len(FILES))
    decoded = {}
    bad = []
    for name, payload in FILES.items():
        raw = gzip.decompress(base64.b85decode("".join(payload)))
        digest = hashlib.sha256(raw).hexdigest()
        ok = digest == EXPECTED[name]
        print("  [%s] %-26s %7d bytes  sha256 %s"
              % ("OK " if ok else "BAD", name, len(raw), digest[:16]))
        decoded[name] = raw
        if not ok:
            bad.append(name)

    if bad:
        print("\\nCHECKSUM MISMATCH: %s" % ", ".join(bad))
        print("The embedded payload is corrupt - re-copy verify.py in full.")
        print("No files were written.")
        return 2

    # Phase 2 - local-conflict scan BEFORE the first write (reviewer R2).
    # Payload verification alone does not protect local edits: a VALID
    # bundle would otherwise silently overwrite a locally edited file.
    # Any destination that exists and differs is refused by default;
    # identical files are a no-op; --force makes overwriting explicit.
    conflicts, up_to_date, to_write = [], [], []
    for name, raw in decoded.items():
        target = os.path.join(os.getcwd(), name)
        if not os.path.exists(target):
            to_write.append(name)
            continue
        with open(target, "rb") as fh:
            existing = fh.read()
        if hashlib.sha256(existing).hexdigest() == EXPECTED[name]:
            up_to_date.append(name)
        else:
            conflicts.append(name)

    if conflicts and not force:
        print("\\nLOCAL CONFLICT: %d destination file(s) differ from the bundle:"
              % len(conflicts))
        for name in conflicts:
            print("  - %s" % name)
        print("Refusing to overwrite local edits. Review them, then keep")
        print("them, or re-run with --force to replace them explicitly.")
        print("No files were written.")
        return 4

    for name in up_to_date:
        print("  [== ] %-26s already up to date (no write)" % name)
    for name in conflicts:
        print("  [!! ] %-26s differs locally - OVERWRITING (--force)" % name)

    print("\\nall checksums match - extracting into %s" % os.getcwd())
    post_write_bad = []
    for name in sorted(set(to_write) | set(conflicts)):
        raw = decoded[name]
        target = os.path.join(os.getcwd(), name)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(raw)
        # Belt and suspenders: re-hash what actually landed on disk.
        with open(target, "rb") as fh:
            on_disk = hashlib.sha256(fh.read()).hexdigest()
        if on_disk != EXPECTED[name]:
            post_write_bad.append(name)
            print("  [BAD] %s differs on disk after writing" % name)

    if post_write_bad:
        print("\\nPOST-WRITE MISMATCH: %s" % ", ".join(post_write_bad))
        return 3

    print("extraction verified from disk")
    if os.path.exists("tests/smoke_offline.py"):
        print("\\nrunning offline test suite...\\n")
        rc = subprocess.call([sys.executable, "tests/smoke_offline.py"])
        if rc != 0:
            print("\\nsmoke suite FAILED (exit %d) - do not start training." % rc)
            return rc
        print("\\nsmoke suite PASSED")

    print(r"""
confirm the import (catches a stale copy from an earlier session):
    python3 -c "import pipeline; print(pipeline.__version__)"     # expect 2.0.0

next steps, in a FRESH Kaggle session (no model resident):

  export your HF token first (Add-ons -> Secrets) to avoid Hub rate limits;
  every CLI below falls back to the HF_TOKEN env var automatically:
    import os; from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

  1) train the probe on the TRAIN split (out-of-fold calibration,
     ~25-45 min on 2x T4):
    !python train_probe.py --model "Qwen/Qwen2.5-7B-Instruct" --layer 20 \\
        --fpr_limit 0.01 --n_splits 5 --notinject \\
        --output probe_qwen_layer20.joblib

  2) evaluate + benchmark with the SAME probe layer(s) you trained.
     A mismatched --layers is refused by the probe fingerprint check
     (that is F-03 working, not a bug):
    !python evaluate_probe.py --probe probe_qwen_layer20.joblib --layer 20
    !python evaluate_end_to_end.py --probe probe_qwen_layer20.joblib \\
        --layer 20 --l1_escalate 0.0 --dump_scores e2e_scores.csv
    !python benchmark.py --probe probe_qwen_layer20.joblib --layer 20 \\
        --per_dataset 250 --dump_scores bench_scores.csv

  multi-layer variant: train with --layers 12,16,20,24 --output
  probe_qwen_ml.joblib, then pass --layers 12,16,20,24 to EVERY
  eval/benchmark command as well.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

text = (TEMPLATE
        .replace("__ENTRIES__", "\n".join(entries))
        .replace("__EXPECTED__", expected))

out = os.path.join(HERE, "verify.py")
with open(out, "w") as fh:
    fh.write(text)

size = os.path.getsize(out)
print("wrote %s (%.1f KB)" % (out, size / 1024))
print("payload chars: %d" % sum(len(v) for v in blobs.values()))
for k, v in digests.items():
    print("  %-26s %s" % (k, v))
