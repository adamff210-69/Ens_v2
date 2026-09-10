"""
Security Probe Calibration & Stratified Training Framework (v2)
==============================================================
Trains the Layer-2 hidden-state probe with:

  * **Multi-dataset** training (deepset + jailbreak corpora) via --datasets
  * **Hard negatives** from NotInject (benign text stuffed with attack trigger
    words) via --notinject - directly attacks over-defense / false positives
  * **Multi-layer** probing via --layers 12,16,20,24 (concatenated pooled
    hidden states; different layers encode different attack signals)
  * Evasion augmentation: delimiters, base64, zero-width, homoglyph, leetspeak,
    ROT13, HTML comments, fake role tags, whitespace splits
  * FPR-budget threshold calibration + stratified CV

Dataset spec format (comma-separated):
    hf_path:split:text_col:label_col:positive_labels
e.g.  deepset/prompt-injections:train:text:label:1
      jackhhao/jailbreak-classification:train:prompt:type:jailbreak
      rubend18/ChatGPT-Jailbreak-Prompts:train:Prompt:label:1

Kaggle example (v2 - recommended):
    !python train_probe.py --model "Qwen/Qwen2.5-7B-Instruct" \\
        --layers 12,16,20,24 --fpr_limit 0.01 --output probe_qwen_ml.joblib \\
        --datasets "deepset/prompt-injections:train:text:label:1,jackhhao/jailbreak-classification:train:prompt:label:jailbreak" \\
        --notinject --pos_variants 3
"""

import argparse
import random
from typing import Dict, List, Optional, Tuple

from pipeline import HiddenStateProbeLayer, VRAMManager, augment_injection_data

# Default extra corpora (verified available on the HF Hub, ungated)
DEFAULT_DATASETS = [
    "deepset/prompt-injections:train:text:label:1",
    # NOTE: this corpus labels via a "type" column ("jailbreak"/"benign").
    # A spec naming a non-existent column used to silently label every row 0,
    # which cost 1044 attacks. load_spec() now validates columns and shouts.
    "jackhhao/jailbreak-classification:train:prompt:type:jailbreak",
]
NOTINJECT_SOURCE = "leolee99/NotInject"
NOTINJECT_TAG = "notinject"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _is_positive(value, positive_labels: List[str]) -> bool:
    v = str(value).strip().lower()
    return v in {p.strip().lower() for p in positive_labels}


def load_spec(spec: str, max_rows: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """Load one 'path:split:text_col:label_col:pos_labels' dataset spec."""
    from datasets import load_dataset

    parts = spec.split(":")
    if len(parts) < 5:
        raise ValueError(
            f"Bad dataset spec {spec!r}. Expected "
            f"'hf_path:split:text_col:label_col:positive_labels'")
    path, split, text_col, label_col, pos_labels = parts[0], parts[1], parts[2], parts[3], parts[4]

    ds = load_dataset(path, split=split)

    # Fail loud on a typo'd column. Silently treating a missing label column as
    # "not positive" can turn an entire attack corpus into negatives, which is
    # exactly what happened with jackhhao's "type" column vs a "label" spec.
    cols = list(ds.column_names)
    missing = [c for c in (text_col, label_col) if c not in cols]
    if missing:
        raise ValueError(
            f"{path} ({split}) has no column(s) {missing}. "
            f"Available columns: {cols}. "
            f"Fix the spec to '{path}:{split}:<text_col>:<label_col>:<pos_labels>'."
        )

    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))

    texts, labels = [], []
    for row in ds:
        txt = row.get(text_col)
        if txt is None or not str(txt).strip():
            continue
        labels.append(1 if _is_positive(row.get(label_col), pos_labels.split("|")) else 0)
        texts.append(str(txt))

    from collections import Counter
    raw_values = Counter(str(r.get(label_col)).strip().lower() for r in ds)
    n_pos = sum(labels)
    print(f"  [dataset] {path} ({split}) -> {len(texts)} rows, {n_pos} positives "
          f"| raw {label_col} values: {dict(raw_values.most_common(5))}")
    if n_pos == 0 and texts:
        print(f"  [dataset] !! WARNING: {path} ({split}) contributed 0 positives. "
              f"pos_labels={pos_labels!r} matched nothing in {label_col}. "
              f"Actual values: {list(raw_values)[:6]}")
    return texts, labels


def load_notinject(max_rows: Optional[int] = None,
                   positives: bool = False) -> Tuple[List[str], List[int]]:
    """NotInject: benign text stuffed with injection trigger words.

    These are HARD NEGATIVES by construction (benign intent, attack-like
    vocabulary) - the direct data-side fix for over-defense. Real structure:
    config 'default' with splits NotInject_one/two/three.
    """
    from datasets import get_dataset_split_names, load_dataset

    try:
        splits = [s for s in get_dataset_split_names("leolee99/NotInject", "default")
                  if s.lower().startswith("notinject")]
    except Exception:
        splits = ["NotInject_one", "NotInject_two", "NotInject_three"]

    texts: List[str] = []
    labels: List[int] = []
    for split in splits:
        try:
            ds = load_dataset("leolee99/NotInject", "default", split=split)
            for row in ds:
                txt = row.get("prompt") or row.get("text") or ""
                if not str(txt).strip():
                    continue
                texts.append(str(txt))
                labels.append(1 if (positives and int(bool(row.get("label", 0)))) else 0)
        except Exception as e:  # noqa: BLE001
            print(f"  [notinject] split {split} failed: {e}")

    if max_rows:
        texts, labels = texts[:max_rows], labels[:max_rows]
    if texts:
        print(f"  [notinject] {len(texts)} hard negatives "
              f"(benign + trigger words)")
    return texts, labels


def _stratified_sample(indices: List[int], sources: List[str], n: int,
                       rng) -> List[int]:
    """Take ``n`` indices allocated across sources proportionally.

    Taking a plain random sample of positives would let the larger corpus crowd
    out the smaller one, so the probe could be trained on almost no examples of
    an entire attack family.
    """
    if len(indices) <= n:
        return list(indices)
    by_source: Dict[str, List[int]] = {}
    for i in indices:
        by_source.setdefault(sources[i], []).append(i)

    total = len(indices)
    quotas: Dict[str, int] = {}
    remainders = []
    for src, idxs in by_source.items():
        exact = n * len(idxs) / total
        quotas[src] = int(exact)
        remainders.append((exact - int(exact), src))

    shortfall = n - sum(quotas.values())
    remainders.sort(reverse=True)
    i = 0
    while shortfall > 0 and remainders:
        quotas[remainders[i % len(remainders)][1]] += 1
        shortfall -= 1
        i += 1

    keep: List[int] = []
    for src, idxs in by_source.items():
        q = min(quotas[src], len(idxs))
        keep += idxs if q >= len(idxs) else rng.sample(idxs, q)
    return keep


def build_training_set(
    specs: List[str],
    use_notinject: bool = True,
    cap_per_dataset: Optional[int] = None,
    balance: bool = True,
    seed: int = 42,
    protect_hard_negatives: bool = True,
) -> Tuple[List[str], List[int], List[str]]:
    """Load + merge all sources, then balance the classes.

    Returns ``(texts, labels, sources)`` where ``sources[i]`` names the corpus a
    row came from, which drives both the composition report and the protection
    of hard negatives during balancing.
    """
    all_texts: List[str] = []
    all_labels: List[int] = []
    all_sources: List[str] = []

    for spec in specs:
        path = spec.split(":")[0]
        try:
            t, l = load_spec(spec, max_rows=cap_per_dataset)
        except Exception as e:  # noqa: BLE001
            print(f"  [dataset] {spec} FAILED: {e}")
            continue
        tag = path.split("/")[-1]
        all_texts += t
        all_labels += l
        all_sources += [tag] * len(t)

    if use_notinject:
        t, l = load_notinject(max_rows=cap_per_dataset)
        all_texts += t
        all_labels += l
        all_sources += [NOTINJECT_TAG] * len(t)

    if not all_texts:
        raise RuntimeError("No training data loaded - check --datasets specs.")

    if balance:
        from collections import Counter

        pos = [i for i, l in enumerate(all_labels) if l == 1]
        neg = [i for i, l in enumerate(all_labels) if l == 0]
        n = min(len(pos), len(neg))
        rng = random.Random(seed)

        # Hard negatives (NotInject) are the entire over-defense fix. Keep every
        # one of them, then fill the remaining negative quota with the rest.
        if protect_hard_negatives:
            keep_hard = [i for i in neg if all_sources[i] == NOTINJECT_TAG]
        else:
            keep_hard = []
        if len(keep_hard) > n:
            keep_hard = rng.sample(keep_hard, n)
        hard_set = set(keep_hard)
        soft = [i for i in neg if i not in hard_set]
        quota_soft = max(0, n - len(keep_hard))
        keep_soft = soft if len(soft) <= quota_soft else rng.sample(soft, quota_soft)
        keep_neg = sorted(keep_hard + keep_soft)

        keep_pos = sorted(_stratified_sample(pos, all_sources, n, rng))

        # Report composition BEFORE re-indexing: keep_pos/keep_neg refer to the
        # pre-rebuild arrays.
        pos_comp = dict(Counter(all_sources[i] for i in keep_pos))
        neg_comp = dict(Counter(all_sources[i] for i in keep_neg))

        keep = sorted(keep_pos + keep_neg)
        all_texts = [all_texts[i] for i in keep]
        all_labels = [all_labels[i] for i in keep]
        all_sources = [all_sources[i] for i in keep]

        print(f"  [balance] {len(keep_pos)} positives {pos_comp} + "
              f"{len(keep_neg)} negatives {neg_comp}")

    return all_texts, all_labels, all_sources


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate causal hidden-state probes (v2: multi-dataset, multi-layer)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--layer", default=20, type=int,
                        help="Single probe layer (compat). Ignored if --layers is set")
    parser.add_argument("--layers", default=None, type=str,
                        help="Comma-separated probe layers for concatenated features, "
                             "e.g. '12,16,20,24' (recommended v2 default)")
    parser.add_argument("--pooling", default="last", type=str, choices=["last", "mean"])
    parser.add_argument("--fpr_limit", default=0.01, type=float,
                        help="FPR budget for threshold calibration")
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--output", default="probe_qwen_layer20.joblib", type=str)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), type=str,
                        help="Comma-separated dataset specs "
                             "'path:split:text_col:label_col:pos_labels'")
    parser.add_argument("--notinject", action="store_true",
                        help="Add NotInject benign-with-triggers rows as hard negatives")
    parser.add_argument("--pos_variants", default=3, type=int,
                        help="Evasion variants per positive (0 disables)")
    parser.add_argument("--cap_per_dataset", default=1500, type=int,
                        help="Max rows pulled from each source before balancing")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_protect_hard_negatives", action="store_true",
                        help="Allow NotInject hard negatives to be randomly "
                             "dropped when balancing (not recommended)")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--hf_token", default=None, type=str)
    args = parser.parse_args()

    probe_layers = ([int(x) for x in args.layers.split(",") if x.strip()]
                    if args.layers else None)
    tag = "TrainingInit"

    # ------------------------------------------------------------------
    print(f"[{tag}] Building training set from {len(args.datasets.split(','))} source(s)...")
    texts, labels, sources = build_training_set(
        [s.strip() for s in args.datasets.split(",") if s.strip()],
        use_notinject=args.notinject,
        cap_per_dataset=args.cap_per_dataset,
        seed=args.seed,
        protect_hard_negatives=not args.no_protect_hard_negatives,
    )
    print(f"[{tag}] Merged+balanced base set: {len(texts)} rows "
          f"({sum(labels)} positives / {len(labels) - sum(labels)} negatives)")

    groups = None
    if not args.no_augmentation:
        before = len(texts)
        texts, labels, groups = augment_injection_data(
            texts, labels, seed=args.seed,
            variants_per_positive=args.pos_variants,
            return_groups=True,
        )
        print(f"[{tag}] Augmented {before} -> {len(texts)} rows "
              f"({sum(labels)} positives, {len(set(groups))} source groups "
              f"for leak-free CV)")

    # ------------------------------------------------------------------
    layers_desc = probe_layers or [args.layer]
    print(f"[{tag}] Initializing probe: {args.model} layer(s) {layers_desc}...")
    probe = HiddenStateProbeLayer(
        model_name=args.model,
        layer=args.layer,
        layers=probe_layers,
        pooling=args.pooling,
        hf_token=args.hf_token,
        load_in_4bit=args.load_in_4bit,
    )

    probe.train(texts, labels, target_fpr=args.fpr_limit,
                n_splits=args.n_splits, groups=groups)
    probe.save(args.output)

    VRAMManager.clear_cache()
    print(f"[{tag}] Probe calibration complete -> {args.output}")


if __name__ == "__main__":
    main()
