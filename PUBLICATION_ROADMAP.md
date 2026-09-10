# Publication Roadmap — Prompt Injection Detection Pipeline

**Assessed:** 2026-09-10 · **Current artifact:** validated 3-layer pipeline, TPR 0.90 / FPR 1.8% on held-out deepset test split

---

## 1. Verdict

**As it stands: not publishable.** Not a criticism of the engineering — it's that the
*contribution* doesn't clear the bar, and the literature has moved. Three hard reasons:

1. **The method is now standard, not novel.** Linear probes on LLM hidden states for
   injection/jailbreak detection are an established line of work with multiple 2025–2026
   papers ([When Benchmarks Lie](https://arxiv.org/html/2602.14161),
   [Agentic LLMs encode IPI exposure](https://arxiv.org/html/2608.02657v2),
   [InstructDetector](https://arxiv.org/html/2505.06311v2),
   [JBShield @ USENIX Sec '25](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-341-zhang-shenyi.pdf)).
   Building L1+L2+L3 is engineering integration, not a research contribution.
2. **The evaluation is too thin for 2026.** One 546/116-row dataset; threshold selected
   on the test split; no confidence intervals; no comparisons against production baselines
   (Prompt Guard 2, Llama Guard 3, PIGuard). The field has moved to *distribution-shift*
   evaluation (Leave-One-Dataset-Out), where single-dataset AUC — including our 0.997 —
   is explicitly treated as insufficient ([AUC 0.998 Is Not Enough](https://arxiv.org/html/2606.22864v1)).
3. **Our L1 choice is a known liability.** ProtectAI v2 is documented as suffering
   trigger-word over-defense, with accuracy near random on NotInject-style benign text
   ([PIGuard, ACL 2025](https://aclanthology.org/2025.acl-long.1468.pdf)). We also
   observed this ourselves (repetitive-text false spikes). A reviewer will flag it.

**What we *do* have that's interesting:** the gate-starvation finding — L2 measured
0.997 AUC standalone but contributed ~1 catch end-to-end because the escalation gate
inherited L1's blind spots; always-on routing took TPR 0.37 → 0.90. That is a real,
quantified empirical result **about how guardrail stacks fail**, not about a new detector.

Important caveat for honesty: the underlying principle is classic cascade design
(a cheap first stage in a cascade must have near-total recall, or everything downstream
is starved). So this is *not* a novel principle — it is a **demonstration that production
LLM guardrail stacks violate it, and a measurement of the cost.** That must be positioned
as empirical/systems work, not as a discovery.

---

## 2. The only two framings that could be publishable

### Framing A — "Guardrail cascades don't compose" (measurement paper)
> Component metrics of LLM guardrail layers do not predict ensemble behavior; the routing
> policy between layers is a first-class design variable that can silently nullify whole
> layers.

Needs to be **general**, i.e. reproduce the phenomenon across:
- ≥ 3 L1 detectors (ProtectAI v2, Prompt Guard 2, PIGuard) × ≥ 2 probe backbones (Qwen2.5-7B, Llama-3.1-8B)
- ≥ 4 datasets spanning families: direct injection, indirect/RAG (BIPIA), tool/agentic (AgentDojo, InjecAgent), over-defense benign (NotInject)
- Routing policies compared: never-on L2, threshold gates at several τ, always-on, oracle, and a **learned deferral router**
- Metrics: TPR/FPR with **bootstrap CIs**, latency + VRAM (2×T4 reproducibility is a selling point), cost-recall Pareto curves

### Framing B — Framing A + a method (full paper)
Add a **calibrated deferral router**: a small head that predicts when L2 will disagree with
L1, trained on cheap features, giving most of always-on's recall at a fraction of its cost.
That converts "we found a problem" into "we found a problem and a fix with a Pareto
guarantee". Realistic as a conference paper *only* with adaptive attacks + agentic/multi-turn
scope added.

### Framing C — technical report / blog + artifact release (recommended if time-boxed)
No novelty bar, immediate visibility, and genuinely useful to practitioners: "we shipped a
2×T4-reproducible 3-layer injection guard and here's where composition bit us." Days of
work, not months.

---

## 3. Experiment checklist (what a reviewer will demand)

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | Threshold calibrated on a **train-validation** slice, never test | Removes the mild test-tuning objection | ~2 h |
| 2 | Bootstrap CIs on TPR/FPR | 56 benign rows → FPR granularity 1.8%/row; point estimates are meaningless alone | ~3 h |
| 3 | Baselines: Prompt Guard 2 (`[3](https://www.augmentcode.com/guides/prompt-injection-detection)` reports 97.5% recall @1% FPR English, 71.4% CyberSecEval indirect), Llama Guard 3, PIGuard | Reviewers require comparable systems, not just ablations of ours | 1–2 days |
| 4 | OOD / LODO evaluation across ≥4 datasets | Single-dataset results are desk-reject material now | 3–5 days |
| 5 | Reproduce the routing pathology on ≥2 detector pairs | Turns single-system anecdote into a phenomenon | 2–3 days |
| 6 | Latency + VRAM accounting per policy | The cost side of the Pareto claim | ~1 day |
| 7 | Learned deferral router (Framing B) | The actual novel artifact | 1–2 weeks |
| 8 | Adaptive attacks (obfuscation, paraphrasing, unicode, delimiter variants; ideally agentic/tool-output injection) | Security venues will not accept non-adaptive evaluation | 1–2 weeks |
| 9 | Harden or scope out L3 self-judge | Its prompt consumes untrusted text — reviewers will call it injectable | 3–5 days |
| 10 | Open-source release: code, artifacts, seed scripts, exact versions | Strongest part of our current work; keep it | ~1 day |

**Compute:** all of the above fits on 2×T4 (probe training ≈30 min per backbone;
the expensive part is many detectors × datasets × seeds = tens of GPU-hours, not hundreds).

---

## 4. Venue map (as of Sept 2026 — verify current CFPs)

| Tier | Venue | Fit | Realistic with |
|---|---|---|---|
| Report | arXiv preprint + blog + repo | Framing C | today |
| Workshop | NeurIPS/ICLR safety & security workshops; AISec; SaTML workshop tracks | Framing A | items 1–6, 10 (~3–5 weeks part-time) |
| Journal | **TMLR** (rolling; judges correctness/empirical quality over novelty) | Framing A/B with items 1–8 | ~2 months |
| Conf. (NLP) | ACL/NAACL/EMNLP main or Findings | Framing B | items 1–9 (~3–4 months, with a collaborator) |
| Conf. (Security) | USENIX Sec / CCS / SaTML main track | Framing B + strong adaptive eval | 4–6+ months; competitive |

Note: deadlines rotate. NeurIPS 2026 workshop CFPs largely closed in Aug–Sep 2026;
ICLR 2027 workshops (~Feb), SaTML 2027, AISec 2027, and the ACL 2027 cycle are the next
practical targets. TMLR accepts year-round.

---

## 5. Reviewer objections to pre-empt (write these into the paper yourself)

- "This is a system integration, not a contribution." → Frame as measurement of composition,
  generalized across detectors/backbones/datasets.
- "Your gate finding is just the classic cascade-recall principle." → Agree explicitly, then
  show that production guardrail stacks violate it and quantify the loss + fix.
- "Single dataset, small, dated." → LODO + over-defense set + realistic benign corpus.
- "No adaptive adversary." → item 8.
- "Your judge is itself prompt-injectable." → harden (delimit + ignore-instructions + dedicated
  small judge) or explicitly scope L3 out of the claims.
- "AUC 0.997 in-distribution proves nothing." → lead with OOD numbers; keep in-distribution
  as a sanity check only ([same lesson as the AUC-0.998 paper](https://arxiv.org/html/2606.22864v1)).

---

## 6. Recommendation

1. **Publish the technical report now** (Framing C): the gate-starvation result is genuinely
   useful and the artifact is reproducible on free-tier GPUs. Costs days.
2. **If a peer-reviewed paper is the goal**, commit to Framing A → B with a co-author:
   items 1–6 first (they alone make a credible workshop submission), then 7–9 for TMLR or
   an *ACL-tier venue.
3. **Do not** submit the current artifact to any venue — a rejection here is cheap to avoid
   and the fix is mostly measurement work we already have the harness for.

The strongest single sentence the work can support today:
> *"A hidden-state probe with 0.997 AUC contributed one detection in a production-style
> cascade because the routing gate inherited the surface classifier's blind spots; making
> the second layer always-on raised end-to-end recall from 0.37 to 0.90 at a 1.8% false-positive rate."*
