#!/usr/bin/env python3
"""Build a stdlib-only, self-extracting installer for the pipeline stack.

Output: /home/user/injection_pipeline/verify.py

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

Writes pipeline.py, train_probe.py, benchmark.py and tests/smoke_offline.py,
verifies each against a pinned SHA-256, then runs the offline test suite so a
bad transfer cannot go unnoticed.

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
    print("extracting %d files into %s" % (len(FILES), os.getcwd()))
    bad = []
    for name, payload in FILES.items():
        raw = gzip.decompress(base64.b85decode("".join(payload)))
        digest = hashlib.sha256(raw).hexdigest()
        target = os.path.join(os.getcwd(), name)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(raw)
        ok = digest == EXPECTED[name]
        if not ok:
            bad.append(name)
        print("  [%s] %-26s %7d bytes  sha256 %s"
              % ("OK " if ok else "BAD", name, len(raw), digest[:16]))

    if bad:
        print("\\nCHECKSUM MISMATCH: %s" % ", ".join(bad))
        print("The embedded payload is corrupt - re-copy verify.py in full.")
        return 2

    print("\\nall checksums match")
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
