"""
End-to-end input-gate evaluation (L1 -> L2), mirroring pipeline.run()'s gating
logic WITHOUT generation.

Why this exists
---------------
evaluate_probe.py measures L2 in ISOLATION (probe sees every row). The live
pipeline gates L2 behind "L1 score >= l1_escalate_threshold". On
deepset/prompt-injections' test split, that gate proved catastrophic:

    L1 hard-blocks 21/60 injections
    only 1/60 injections land in the escalation band [0.30, 0.85)
    38/60 injections score < 0.30 at L1 -> the probe NEVER sees them

i.e. a probe with 0.78 standalone recall contributes ~nothing because it is
starved of traffic. This script measures BOTH gate policies in one pass and
sweeps the probe threshold, so the operating point can be chosen from data:

  Policy A (legacy gate): block = L1_hard OR (p1 >= gate AND p2 >= thr)
  Policy B (always-on)  : block = L1_hard OR (p2 >= thr)

Every non-hard-blocked row gets scored by L2 once, so both policies and the
full threshold sweep come out of a single 7B load. Optionally dumps per-row
scores to CSV for offline analysis.

Kaggle:
    !python evaluate_end_to_end.py --model "Qwen/Qwen2.5-7B-Instruct" \
        --probe probe_qwen_layer20.joblib --layer 20 --dump_scores l1l2_scores.csv
"""

import argparse
import csv
import os

import numpy as np

from pipeline import (DATASET_REVISIONS, InjectionDetectionPipeline,
                      VRAMManager, l1_hard_block, pinned_revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble (L1+L2) held-out evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--probe", default="probe_qwen_layer20.joblib", type=str)
    parser.add_argument("--layer", default=20, type=int)
    parser.add_argument("--probe_threshold", default=None, type=float,
                        help="Override probe decision threshold (default: artifact value)")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), type=str,
                        help="HF token; falls back to the HF_TOKEN env var")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--l1_block", default=0.85, type=float,
                        help="L1 hard-block threshold (matches pipeline default)")
    parser.add_argument("--l1_escalate", default=0.30, type=float,
                        help="Legacy gate threshold (Policy A). 0.0 = always run L2 = Policy B")
    parser.add_argument("--max_examples", default=250, type=int)
    parser.add_argument("--dump_scores", default=None, type=str,
                        help="Write per-row L1/L2 scores + labels to this CSV path")
    args = parser.parse_args()

    print("Building pipeline (L1 + LLM + probe)...")
    pipe = InjectionDetectionPipeline(
        llm_model_name=args.model,
        hf_token=args.hf_token,
        probe_layer=args.layer,
        probe_path=args.probe,
        load_in_4bit=args.load_in_4bit,
        l1_block_threshold=args.l1_block,
        l1_escalate_threshold=args.l1_escalate,
        probe_threshold=args.probe_threshold,
        fail_closed=True,
    )
    artifact_thr = pipe.layer2.threshold
    active_thr = (args.probe_threshold if args.probe_threshold is not None
                  else artifact_thr)
    print(f"Artifact probe threshold: {artifact_thr:.4f} | active: {active_thr:.4f}")
    print(f"L1 block >= {args.l1_block}")

    # Heavy dataset import stays in the CLI execution path (audit review §3).
    from datasets import load_dataset

    # ------------------------------------------------------------------
    test_split = load_dataset(
        "deepset/prompt-injections",
        revision=pinned_revision(DATASET_REVISIONS,
                                 "deepset/prompt-injections"))["test"]
    texts = [ex["text"] for ex in test_split][: args.max_examples]
    y = np.array([int(ex["label"]) for ex in test_split])[: args.max_examples]
    if not texts:
        print("No test rows — nothing to evaluate.")
        return

    print(f"\nAuditing {len(texts)} held-out samples "
          f"(positives={int(y.sum())}); L2 runs on every non-hard-blocked row...\n")

    p1s = np.zeros(len(texts))
    p2s = np.full(len(texts), np.nan)   # NaN = L2 not evaluated (L1 hard-blocked)
    hard = np.zeros(len(texts), dtype=bool)

    for i, text in enumerate(texts):
        # ---- Layer 1 (same decision rule as pipeline.run) ----------------
        _, p1, meta = pipe.layer1.score(text)
        p1s[i] = p1
        n_ch, n_al = meta["total_chunks"], meta["alert_chunks_count"]
        # Production dual-key hard-block rule, shared with pipeline.run() via
        # pipeline.l1_hard_block (audit F-04: the evaluator previously had its
        # own divergent predicate, 'n_ch == 1 or n_al >= 2 or p1 >= 0.95').
        hard[i] = l1_hard_block(p1, n_ch, n_al, args.l1_block)

        # ---- Layer 2 on everything L1 did not already block -------------
        if not hard[i]:
            _, p2 = pipe.layer2.score(text)
            p2s[i] = p2

        if (i + 1) % 25 == 0:
            print(f"  audited {i + 1}/{len(texts)}")
            VRAMManager.clear_cache()

    # ------------------------------------------------------------------
    def tally(blocked: np.ndarray) -> dict:
        tp = int(((y == 1) & blocked).sum())
        fp = int(((y == 0) & blocked).sum())
        fn = int(((y == 1) & ~blocked).sum())
        tn = int(((y == 0) & ~blocked).sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "tpr": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn),
                "acc": (tp + tn) / max(1, len(y))}

    def policy(gate: float, thr: float, always: bool) -> np.ndarray:
        escalated = np.ones(len(y), dtype=bool) if always else (p1s >= gate)
        l2_block = escalated & ~hard & (p2s >= thr)   # NaN >= thr -> False
        return hard | l2_block

    print("=" * 70)
    print("GATE POLICY COMPARISON (held-out test split)")
    print("=" * 70)
    l1_only = int((hard & (y == 1)).sum())
    for label, blocked in [
        (f"Policy A  legacy gate p1>={args.l1_escalate:.2f}"
         f", L2 thr={active_thr:.4f}", policy(args.l1_escalate, active_thr, False)),
        (f"Policy B  always-on L2, L2 thr={active_thr:.4f}", policy(0.0, active_thr, True)),
    ]:
        st = tally(blocked)
        # Policy-actual L2 contribution: rows L2 blocked that L1 did not,
        # restricted to rows that this policy actually escalated to L2.
        l2_added = int((blocked & ~hard & (y == 1)).sum())
        l2_fp = int((blocked & ~hard & (y == 0)).sum())
        print(f"\n{label}")
        print(f"  TPR={st['tpr']:.4f} ({st['tp']}/{st['tp'] + st['fn']})  "
              f"FPR={st['fpr']:.4f} ({st['fp']}/{st['fp'] + st['tn']})  "
              f"acc={st['acc']:.4f}")
        print(f"  L1 caught {l1_only} | L2 added {l2_added} beyond L1 "
              f"| L2 false positives: {l2_fp}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP, POLICY B (always-on L2)")
    print("=" * 70)
    cand = sorted({0.50, 0.75, 0.90, 0.95, 0.98, artifact_thr, 0.99, 0.995})
    for thr in cand:
        if not (0.0 < thr <= 1.0):
            continue
        st = tally(policy(0.0, thr, True))
        star = "  <- artifact" if abs(thr - artifact_thr) < 1e-9 else ""
        print(f"  thr={thr:<6.4f}  TPR={st['tpr']:.4f}  FPR={st['fpr']:.4f}"
              f"  acc={st['acc']:.4f}{star}")

    # ------------------------------------------------------------------
    seen = ~hard & ~np.isnan(p2s)
    print("\n" + "=" * 70)
    print("L2 COVERAGE / STARVATION CHECK")
    print("=" * 70)
    print(f"  rows L1 hard-blocked (L2 never consulted): {int(hard.sum())}")
    print(f"  rows scored by L2                        : {int(seen.sum())}")
    inj_seen = int(((y == 1) & seen).sum())
    inj_total = int((y == 1).sum())
    print(f"  injections scored by L2                  : {inj_seen}/{inj_total}")
    if inj_seen:
        inj_scores = p2s[(y == 1) & seen]
        print(f"  L2 injection scores: min={inj_scores.min():.4f} "
              f"median={np.median(inj_scores):.4f} max={inj_scores.max():.4f}")
    ben_seen = (y == 0) & seen
    if ben_seen.sum():
        ben_scores = p2s[ben_seen]
        print(f"  L2 benign scores   : max={ben_scores.max():.4f} "
              f"mean={ben_scores.mean():.4f}  "
              f"(benign scoring >=0.95: {int((ben_scores >= 0.95).sum())})")

    if args.dump_scores:
        with open(args.dump_scores, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "label", "l1_score", "l2_score",
                        "l1_hard_blocked", "text"])
            for i, text in enumerate(texts):
                w.writerow([i, int(y[i]), f"{p1s[i]:.6f}",
                            "" if np.isnan(p2s[i]) else f"{p2s[i]:.6f}",
                            int(hard[i]), text.replace("\n", " ")[:300]])
        print(f"\nPer-row scores written -> {args.dump_scores}")

    VRAMManager.clear_cache()


if __name__ == "__main__":
    main()
