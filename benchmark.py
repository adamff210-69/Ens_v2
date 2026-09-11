"""
Benchmark harness for the input-gate stack (L1 -> L2).
=====================================================
Measures every configuration on every dataset in ONE model load and reports
bootstrap confidence intervals, because point estimates on a few hundred rows
are noise.

Configurations compared (all sharing the same L1/L2 scores):
  * L1-only            - surface guard(s) alone
  * L2-only            - hidden-state probe alone
  * stack (legacy)     - L1 hard-block OR (L1>=gate AND L2>=thr)
  * stack (always-on)  - L1 hard-block OR L2>=thr

No dual-key row is emitted: since this harness scores L2 on every
non-hard-blocked row, dual-key's blocked-or-not bit is algebraically
identical to always-on (audit finding F-05). Its serving-side behavior is
covered by TestDualKeyPolicy in tests/smoke_offline.py instead.

Datasets (test/held-out splits only — never used for training):
  * deepset/prompt-injections (test)
  * jackhhao/jailbreak-classification (test, if present)
  * leolee99/NotInject (benign over-defense set -> FPR estimate)
  * rubend18/ChatGPT-Jailbreak-Prompts (sample -> TPR estimate)

Usage (Kaggle):
    !python benchmark.py --model "Qwen/Qwen2.5-7B-Instruct" \
        --probe probe_qwen_ml.joblib --layers 12,16,20,24 \
        --l1_models "ProtectAI/deberta-v3-base-prompt-injection-v2,leolee99/PIGuard" \
        --per_dataset 250 --dump_scores bench_scores.csv
"""

import argparse
import csv

import numpy as np
from datasets import load_dataset

from pipeline import (DATASET_REVISIONS, InjectionDetectionPipeline,
                      VRAMManager, l1_hard_block, pinned_revision)


# --------------------------------------------------------------------------
# Dataset registry: (name, loader)
# --------------------------------------------------------------------------
def _load_deepset_test(cap):
    ds = load_dataset(
        "deepset/prompt-injections", split="test",
        revision=pinned_revision(DATASET_REVISIONS, "deepset/prompt-injections"))
    ds = ds.select(range(min(cap, len(ds))))
    return ([r["text"] for r in ds], np.array([int(r["label"]) for r in ds]))


def _load_jailbreak_classification_test(cap):
    ds = load_dataset(
        "jackhhao/jailbreak-classification", split="test",
        revision=pinned_revision(DATASET_REVISIONS,
                                 "jackhhao/jailbreak-classification"))
    ds = ds.select(range(min(cap, len(ds))))
    texts = [r["prompt"] for r in ds]
    # This corpus labels via "type" ("jailbreak"/"benign"); there is no "label"
    # column, so the old key raised KeyError and the surface silently dropped.
    labels = np.array([1 if str(r["type"]).lower().startswith("jail") else 0
                       for r in ds])
    return texts, labels


def _load_notinject(cap):
    """Benign-only over-defense set: every row is a NEGATIVE by construction.

    Real structure: config 'default', splits NotInject_one/two/three (113 rows
    each), columns prompt/word_list/category. All three splits are concatenated.
    """
    from datasets import get_dataset_split_names
    _nj_rev = pinned_revision(DATASET_REVISIONS, "leolee99/NotInject")
    try:
        splits = [s for s in get_dataset_split_names(
            "leolee99/NotInject", "default", revision=_nj_rev)
            if s.lower().startswith("notinject")]
    except Exception:
        splits = ["NotInject_one", "NotInject_two", "NotInject_three"]

    texts, per_split = [], {}
    for split in splits:
        try:
            ds = load_dataset("leolee99/NotInject", "default", split=split,
                              revision=_nj_rev)
            texts += [str(r.get("prompt") or r.get("text") or "") for r in ds]
            per_split[split] = len(ds)
        except Exception as e:  # noqa: BLE001
            print(f"  [notinject] split {split} failed: {e}")
    texts = [t for t in texts if t.strip()]
    if cap:
        texts = texts[:cap]
    if texts:
        print(f"  [notinject] loaded {len(texts)} benign trigger-word rows "
              f"from {per_split}")
    return texts, np.zeros(len(texts), dtype=int)


def _load_ttp(cap):
    """rubend18 ChatGPT-Jailbreak-Prompts: every row is a POSITIVE by construction."""
    ds = load_dataset(
        "rubend18/ChatGPT-Jailbreak-Prompts", split="train",
        revision=pinned_revision(DATASET_REVISIONS,
                                 "rubend18/ChatGPT-Jailbreak-Prompts"))
    ds = ds.select(range(min(cap, len(ds))))
    texts = []
    for r in ds:
        t = r.get("Prompt") or r.get("prompt") or r.get("text") or ""
        if str(t).strip():
            texts.append(str(t))
    return texts, np.ones(len(texts), dtype=int)


REGISTRY = {
    "deepset_test": _load_deepset_test,
    "jailbreak_test": _load_jailbreak_classification_test,
    "notinject_benign": _load_notinject,
    "jailbreak_prompts": _load_ttp,
}


def bootstrap_ci(hits: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Percentile bootstrap CI for a rate (mean of a 0/1 array)."""
    if len(hits) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    means = hits[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Input-gate stack benchmark")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    ap.add_argument("--probe", default="probe_qwen_layer20.joblib", type=str)
    ap.add_argument("--layers", default=None, type=str,
                    help="Probe layers used by the artifact, e.g. '12,16,20,24'")
    ap.add_argument("--layer", default=20, type=int, help="Compat single layer")
    ap.add_argument("--l1_models", default=None, type=str,
                    help="Comma-separated guard models for the L1 ensemble")
    ap.add_argument("--probe_threshold", default=None, type=float)
    ap.add_argument("--l1_block", default=0.85, type=float)
    ap.add_argument("--l1_gate", default=0.30, type=float,
                    help="Legacy gate used by the 'stack (legacy)' config")
    ap.add_argument("--per_dataset", default=250, type=int)
    ap.add_argument("--datasets", default="deepset_test,notinject_benign,jailbreak_prompts",
                    type=str)
    ap.add_argument("--dump_scores", default=None, type=str)
    ap.add_argument("--hf_token", default=None, type=str)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    l1_models = ([m.strip() for m in args.l1_models.split(",") if m.strip()]
                 if args.l1_models else None)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None

    print("Building pipeline (L1 ensemble + LLM + probe)...")
    pipe = InjectionDetectionPipeline(
        llm_model_name=args.model,
        hf_token=args.hf_token,
        probe_layer=args.layer,
        probe_layers=layers,
        l1_models=l1_models,
        probe_path=args.probe,
        load_in_4bit=args.load_in_4bit,
        l1_block_threshold=args.l1_block,
        l1_escalate_threshold=0.0,
        probe_threshold=args.probe_threshold,
        fail_closed=True,
    )
    thr = (args.probe_threshold if args.probe_threshold is not None
           else pipe.layer2.threshold)
    print(f"Probe thr={thr:.4f} | layers={pipe.layer2.layers} | "
          f"L1 guards={[c['name'] for c in pipe.layer1.classifiers]}")

    dump_rows = []
    summary = []

    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        loader = REGISTRY.get(name)
        if loader is None:
            print(f"[skip] unknown dataset key {name}")
            continue
        print(f"\n{'='*72}\nDATASET: {name}\n{'='*72}")
        try:
            texts, y = loader(args.per_dataset)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] load failed: {e}")
            continue
        if len(texts) == 0:
            continue
        print(f"{len(texts)} rows ({int(y.sum())} positives)")

        p1 = np.zeros(len(texts))
        p2 = np.full(len(texts), np.nan)
        hard = np.zeros(len(texts), dtype=bool)

        for i, text in enumerate(texts):
            _, s1, meta = pipe.layer1.score(text)
            p1[i] = s1
            # Production hard-block predicate, shared with pipeline.run()
            # and evaluate_end_to_end.py (audit F-04 long-term fix).
            hard[i] = l1_hard_block(s1, meta["total_chunks"],
                                    meta["alert_chunks_count"], args.l1_block)
            if not hard[i]:
                _, p2[i] = pipe.layer2.score(text)
            if (i + 1) % 25 == 0:
                print(f"  scored {i+1}/{len(texts)}")
                VRAMManager.clear_cache()

        # ---- configs -------------------------------------------------
        l1_only = p1 >= args.l1_block
        l2_only = p2 >= thr                      # NaN -> False
        legacy = hard | ((p1 >= args.l1_gate) & (p2 >= thr))
        always_on = hard | (p2 >= thr)

        # NOTE: no dual-key row here, by design (audit finding F-05).
        # This harness scores L2 on EVERY non-hard-blocked row, so dual-key's
        # blocked-or-not bit is algebraically identical to always-on:
        #   hard | ((p1 >= block) & ~hard & (p2 >= thr)) | (p2 >= thr)
        #     == hard | (p2 >= thr)
        # Dual-key differs from always-on only in *attribution* (which layer
        # blocked) and in latency, not in TPR/FPR — reporting it as a separate
        # policy row was a phantom ablation. The serving-side behavior is
        # verified instead by TestDualKeyPolicy in tests/smoke_offline.py.
        defs = {
            "L1-only": l1_only,
            "L2-only": l2_only,
            "stack(legacy gate)": legacy,
            "stack(always-on)": always_on,
        }

        for cfg, blocked in defs.items():
            tp = int(((y == 1) & blocked).sum())
            fp = int(((y == 0) & blocked).sum())
            fn = int(((y == 1) & ~blocked).sum())
            tn = int(((y == 0) & ~blocked).sum())
            tpr = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            tpr_lo, tpr_hi = bootstrap_ci(((y == 1) & blocked).astype(float)) if (tp + fn) else (np.nan, np.nan)
            fpr_lo, fpr_hi = bootstrap_ci(((y == 0) & blocked).astype(float)) if (fp + tn) else (np.nan, np.nan)
            print(f"  {cfg:<22s} TPR={tpr:.3f} [{tpr_lo:.3f},{tpr_hi:.3f}] ({tp}/{tp+fn})  "
                  f"FPR={fpr:.3f} [{fpr_lo:.3f},{fpr_hi:.3f}] ({fp}/{fp+tn})")
            summary.append((name, cfg, tp, tp + fn, fp, fp + tn, tpr, fpr))

        if args.dump_scores:
            for i, text in enumerate(texts):
                dump_rows.append({
                    "dataset": name, "index": i, "label": int(y[i]),
                    "l1_score": round(float(p1[i]), 6),
                    "l2_score": "" if np.isnan(p2[i]) else round(float(p2[i]), 6),
                    "l1_hard": int(hard[i]), "text": text.replace("\n", " ")[:280],
                })

    # ---- aggregate table (macro-average over datasets with both classes) ----
    print(f"\n{'='*72}\nSUMMARY (per dataset x config)\n{'='*72}")
    print(f"{'dataset':<20s} {'config':<22s} {'TPR':>6s} {'FPR':>6s}  n_pos n_neg")
    for row in summary:
        ds, cfg, tp, npos, fp, nneg, tpr, fpr = row
        print(f"{ds:<20s} {cfg:<22s} {tpr:6.3f} {fpr:6.3f}  {npos:5d} {nneg:5d}")

    if args.dump_scores:
        with open(args.dump_scores, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dump_rows[0].keys()))
            w.writeheader()
            w.writerows(dump_rows)
        print(f"\nPer-row scores -> {args.dump_scores}")

    VRAMManager.clear_cache()


if __name__ == "__main__":
    main()
