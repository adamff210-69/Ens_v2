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
extracts only if all digests match, re-hashes the written files, and then
runs the offline test suite so a bad transfer cannot go unnoticed.

Stdlib only: no pip installs, no network, Python 3.8+.
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
    # Verify FIRST, write SECOND (audit finding F-11). The old installer
    # overwrote files in the CWD before checking digests, so re-running it
    # in a tree with local edits silently reverted those edits before
    # reporting the mismatch. Now: decode + hash in memory, compare all
    # digests, write only if every payload matches, then re-hash from disk.
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

    print("\\nall checksums match - extracting into %s" % os.getcwd())
    post_write_bad = []
    for name, raw in decoded.items():
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

  train the v2 probe:
    !python train_probe.py --model "Qwen/Qwen2.5-7B-Instruct" \\
        --layers 12,16,20,24 --fpr_limit 0.01 --n_splits 5 \\
        --output probe_qwen_ml.joblib \\
        --datasets "deepset/prompt-injections:train:text:label:1,jackhhao/jailbreak-classification:train:prompt:type:jailbreak" \\
        --notinject --pos_variants 3

  then benchmark the stack (same session, one model load):
    !python benchmark.py --model "Qwen/Qwen2.5-7B-Instruct" \\
        --probe probe_qwen_ml.joblib --layers 12,16,20,24 \\
        --l1_models "ProtectAI/deberta-v3-base-prompt-injection-v2,leolee99/PIGuard" \\
        --per_dataset 250 --dump_scores bench_scores.csv
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
