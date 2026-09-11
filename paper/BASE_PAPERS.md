# Base & Reference Papers — Ens_v2 Prompt-Injection Detection Pipeline

Research papers only. This list is the *canonical* reference set used in
`paper.tex` / `paper.pdf` (IEEE numbered references). Non-paper sources
(model cards, dataset pages, Hugging Face repositories, vendor web pages) were
removed from the reference list; where a specific model or dataset is named in
the paper, it is named as a proper noun and cited through the research paper
that introduced or characterizes it.

**DOI status:** every DOI below was resolved and checked against the
publisher's official record (Crossref / DataCite / ACL Anthology / arXiv)
on 2026-09-11. All resolve to the cited paper.

| # | Reference | DOI (verified) |
|---|-----------|----------------|
| 1 | F. Perez and I. Ribeiro, "Ignore Previous Prompt: Attack Techniques for Language Models," *NeurIPS ML Safety Workshop*, 2022. | 10.48550/arXiv.2211.09527 |
| 2 | K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection," *Proc. 16th ACM Workshop on Artificial Intelligence and Security (AISec)*, 2023, pp. 79–90. | 10.1145/3605764.3623985 |
| 3 | Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, and Y. Liu, "Prompt Injection Attack Against LLM-Integrated Applications," arXiv preprint, 2024. | 10.48550/arXiv.2306.05499 |
| 4 | A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, "Universal and Transferable Adversarial Attacks on Aligned Language Models," arXiv preprint, 2023. | 10.48550/arXiv.2307.15043 |
| 5 | H. Inan et al., "Llama Guard: LLM-Based Input-Output Safeguard for Human-AI Conversations," arXiv preprint, 2023. | 10.48550/arXiv.2312.06674 |
| 6 | H. Li, X. Liu, N. Zhang, and C. Xiao, "PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free," *Proc. 63rd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2025, pp. 30420–30437. | 10.18653/v1/2025.acl-long.1468 |
| 7 | H. Li, X. Liu, and C. Xiao, "InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models," arXiv preprint, 2024. | 10.48550/arXiv.2410.22770 |
| 8 | T. Wen, C. Wang, X. Yang, H. Tang, Y. Xie, L. Lyu, Z. Dou, and F. Wu, "Defending Against Indirect Prompt Injection by Instruction Detection," *Findings of the Association for Computational Linguistics: EMNLP 2025*, 2025, pp. 19472–19487. | 10.18653/v1/2025.findings-emnlp.1060 |
| 9 | S. Zhang, Y. Zhai, K. Guo, H. Hu, S. Guo, Z. Fang, L. Zhao, C. Shen, C. Wang, and Q. Wang, "JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation," *Proc. USENIX Security Symposium*, 2025. | 10.48550/arXiv.2502.07557 |
| 10 | M. Fomin, "When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift," arXiv preprint, 2026. | 10.48550/arXiv.2602.14161 |
| 11 | Y. Li, Z. Fan, and Z. Zhuang, "When AUC 0.998 Is Not Enough: A Candidate Evaluation Protocol for Hidden-State Probes of Indirect Prompt Injection in Multimodal Computer-Use Agents," arXiv preprint, 2026. | 10.48550/arXiv.2606.22864 |
| 12 | G. Alain and Y. Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes," *Proc. ICLR Workshop Track*, 2017. | 10.48550/arXiv.1610.01644 |
| 13 | P. Viola and M. Jones, "Rapid Object Detection Using a Boosted Cascade of Simple Features," *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2001, pp. I-511–I-518. | 10.1109/CVPR.2001.990517 |
| 14 | D. Madras, T. Pitassi, and R. Zemel, "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer," *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2018. | 10.48550/arXiv.1711.06664 |
| 15 | P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa Using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," *Proc. International Conference on Learning Representations (ICLR)*, 2023. | 10.48550/arXiv.2111.09543 |
| 16 | Qwen Team, Alibaba Group, "Qwen2.5 Technical Report," arXiv preprint, 2024. | 10.48550/arXiv.2412.15115 |

---

## How each reference maps to the project

- **Threat model (1–4):** Perez & Ribeiro (goal hijacking / prompt leaking),
  Greshake et al. (indirect injection), Liu et al. (injection taxonomy for
  LLM-integrated apps), Zou et al. (GCG-style jailbreaks).
- **Surface guards & over-defense (5–7, 15):** Llama Guard (generative input/
  output guard), PIGuard (introduces the NotInject over-defense benchmark and
  documents ProtectAI-v2 trigger-word over-defense), InjecGuard (over-defense
  mitigation; the research-paper source for prompt-guard over-defense),
  DeBERTaV3 (the encoder backbone of the L1 classifier and of PIGuard/Prompt
  Guard-style models).
- **Hidden-state detection (8–12):** InstructDetector and JBShield
  (representation-level injection/jailbreak detection), Fomin and Li et al.
  (LODO / candidate-evaluation protocols for hidden-state probes), Alain &
  Bengio (linear probes — the method behind Layer 2).
- **Cascades & deferral (13–14):** Viola & Jones (cascade design — the source
  of the "first stage needs near-total recall" invariant behind the
  gate-starvation finding), Madras et al. (learning to defer).
- **Base model (16):** Qwen2.5 technical report — the target LLM
  (Qwen2.5-7B-Instruct) around which all three layers wrap.

## Note on naming vs. citing

The paper names several artifacts that have **no research-paper citation**
available and are therefore *named, not referenced*: the ProtectAI
`deberta-v3-base-prompt-injection-v2` model, the Deepset `prompt-injections`
dataset, and Meta's Llama Prompt Guard 2. Their properties (e.g., over-defense,
held-out test split of 116 rows) are attributed to the research papers above
that document them.
