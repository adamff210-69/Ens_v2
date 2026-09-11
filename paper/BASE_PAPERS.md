# Base & Reference Papers — Ens_v2 Prompt-Injection Detection Pipeline

This file collects the **base and reference papers** that the `Ens_v2` project
(a three-layer prompt-injection detection pipeline) builds on and cites. Each
entry gives the full citation and a one-line note on how it maps to the
project. The same set is used in `paper.tex` (IEEE numbered references).

---

## 1. Threat model — prompt injection & jailbreaks

| # | Reference | Role in the project |
|---|-----------|---------------------|
| 1 | F. Perez and I. Ribeiro, "Ignore previous prompt: Attack techniques for language models," *NeurIPS ML Safety Workshop*, 2022. arXiv:2211.09527 | Canonical formulation of **goal hijacking / prompt leaking**; motivates L1 + L2 detection targets. |
| 2 | K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz, "Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection," *ACM AISec*, 2023. arXiv:2302.12173 | **Indirect injection** through retrieved/tool content; drives the `<untrusted_context>` handling and L3 output audit. |
| 3 | Y. Liu et al., "Prompt injection attack against LLM-integrated applications," arXiv:2306.05499, 2024. | Systematic taxonomy of injection surfaces for integrated LLM apps. |
| 4 | A. Zou et al., "Universal and transferable adversarial attacks on aligned language models," arXiv:2307.15043, 2023 (GCG). | **Adaptive/optimization-based jailbreaks**; the attack family we flag as future work (adaptive evaluation). |
| 5 | OWASP Foundation, "OWASP Top 10 for Large Language Model Applications," 2025. | Prompt injection ranked **LLM01**; frames the motivation. |

## 2. Surface prompt guards (Layer 1 lineage)

| # | Reference | Role in the project |
|---|-----------|---------------------|
| 6 | P. He, J. Gao, W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing," *ICLR*, 2023. arXiv:2111.09543 | **Backbone architecture** of the L1 classifier (and PIGuard / Prompt Guard 2). |
| 7 | ProtectAI, "deberta-v3-base-prompt-injection-v2," Hugging Face model, 2024. | The **L1 surface classifier** used in the pipeline. |
| 8 | Deepset, "deepset/prompt-injections," Hugging Face dataset, 2024. | The **primary training/evaluation dataset** (546 train / 116 test). |
| 9 | H. Li, X. Liu, N. Zhang, C. Xiao, "PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free," *ACL*, 2025, pp. 30420–30437. | Introduces **NotInject** (over-defense benchmark) and documents ProtectAI-v2's trigger-word over-defense; PIGuard is an optional L1 ensemble member. |
| 10 | Meta AI, "Llama Prompt Guard 2," model card, 2025. | **Production baseline** (97.5% recall @ 1% FPR English) the stack is compared against conceptually. |
| 11 | H. Inan et al., "Llama Guard: LLM-based input-output safeguard for human-AI conversations," arXiv:2312.06674, 2023. | Generative input/output guard baseline; also the lineage for the L3 self-judge idea. |

## 3. Hidden-state / representation-level detection (Layer 2 lineage)

| # | Reference | Role in the project |
|---|-----------|---------------------|
| 12 | G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," *ICLR Workshop*, 2017. arXiv:1610.01644 | **Linear probes on intermediate activations** — the method behind L2. |
| 13 | T. Wen et al., "Defending against indirect prompt injection by instruction detection" (InstructDetector), *Findings of EMNLP*, 2025. arXiv:2505.06311 | SOTA **hidden-state + gradient** instruction detector; 99.6% in-domain / 96.9% OOD. Positions our L2 as standard, not novel. |
| 14 | S. Zhang et al., "JBShield: Defending large language models from jailbreak attacks through activated concept analysis and manipulation," *USENIX Security*, 2025. arXiv:2502.07557 | **Concept-activation** defense on hidden states; sibling representation-level work. |
| 15 | M. Fomin, "When benchmarks lie: Evaluating malicious prompt classifiers under true distribution shift," arXiv:2602.14161, 2026. | **Leave-One-Dataset-Out (LODO)** evaluation; activation probes across 18 datasets. The benchmark our `benchmark.py` harness is designed to satisfy. |
| 16 | Y. Li, Z. Fan, Z. Zhuang, "When AUC 0.998 is not enough: A candidate evaluation protocol for hidden-state probes of indirect prompt injection," arXiv:2606.22864, 2026. | Directly reframes our headline ("0.997 AUC proves nothing") — motivates leading with end-to-end/OOD numbers. |

## 4. Cascades & deferral (the analytical lens for the central finding)

| # | Reference | Role in the project |
|---|-----------|---------------------|
| 17 | P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," *CVPR*, 2001. | The classic **cascade** pattern; source of the "first stage needs near-total recall" invariant that explains the gate-starvation finding. |
| 18 | D. Madras, T. Pitassi, R. Zemel, "Predict responsibly: Improving fairness and accuracy by learning to defer," *NeurIPS*, 2018. | Modern **learning-to-defer** framing; motivates the "learned deferral router" as future work. |

## 5. Base model

| # | Reference | Role in the project |
|---|-----------|---------------------|
| 19 | Qwen Team, Alibaba Group, "Qwen2.5 technical report," arXiv:2412.15115, 2024. | The **target 7B LLM** (Qwen2.5-7B-Instruct, 28 layers) around which all three layers wrap. |

---

## Key experimental facts from the project (for writing/reviewing)

- **Held-out set:** `deepset/prompt-injections` test split — 116 rows (60 injections, 56 benign).
- **L1 alone:** TPR 0.350 (21/60), FPR 0.000.
- **L2 probe standalone:** ROC-AUC 0.997, TPR 0.783 at threshold 0.9868, FPR 0.000.
- **Legacy escalation gate (τ=0.30):** only 1/60 injections reach L2 → end-to-end TPR 0.367.
- **Always-on L2:** TPR 0.833 @ τ=0.9868 (FPR 0.000) → **0.900 @ τ=0.90 (FPR 1.8%)**; L2 adds 29 catches beyond L1.
- **Hardware:** 2× Tesla T4 (16 GiB), fp16, Qwen2.5-7B-Instruct, probe on layer 20.
- **Training:** deepset train (546) + jailbreak-classification (527 pos.) + NotInject (339 hard negatives); balanced 730/730; StratifiedGroupKFold; out-of-fold FPR-budget (≤1%) threshold calibration; 7 evasion families + delimiter + base64 augmentation.
