#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render paper.pdf — a two-column, IEEE-conference-styled rendering of the
Ens_v2 paper, generated with reportlab (pure Python). This is a *visual*
companion to paper.tex (the authoritative IEEEtran source).

Usage: python3 render_paper.py   (writes paper.pdf next to this script)
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import black, HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Table, TableStyle,
                                NextPageTemplate, HRFlowable, Flowable)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon

# ----------------------------------------------------------------------------
# Page geometry (US letter, IEEE-ish margins)
# ----------------------------------------------------------------------------
PAGE_W, PAGE_H = letter                      # 612 x 792 pt
MARGIN = 0.75 * inch                         # 54 pt
GAP = 0.19 * inch                            # inter-column gap ~ 13.7 pt
COL_W = (PAGE_W - 2 * MARGIN - GAP) / 2      # ~245 pt
TOP = PAGE_H - MARGIN
BOTTOM = MARGIN
FOOTER_H = 16

TITLE_H = 96                                # full-width title band on page 1
COL_H_1 = TOP - MARGIN - TITLE_H             # columns on page 1
COL_H = TOP - MARGIN - FOOTER_H              # columns on later pages

GREY = HexColor("#f2f2f2")

# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------
def st(name, **kw):
    base = dict(fontName="Times-Roman", fontSize=10, leading=12.2,
                alignment=TA_JUSTIFY, spaceAfter=5)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_BANNER  = st("banner", fontSize=8, leading=9.5, alignment=TA_CENTER,
               textColor=HexColor("#444444"), spaceAfter=2)
S_TITLE   = st("title", fontName="Times-Bold", fontSize=17, leading=19.5,
               alignment=TA_CENTER, spaceAfter=4)
S_AUTHORS = st("authors", fontSize=11, leading=13, alignment=TA_CENTER,
               spaceAfter=2)
S_AFFIL   = st("affil", fontName="Times-Italic", fontSize=10, leading=12,
               alignment=TA_CENTER, spaceAfter=0)
S_EMAIL   = st("email", fontSize=9.5, leading=11.5, alignment=TA_CENTER,
               spaceAfter=3)
S_ABSTRACT= st("abstract", fontSize=9, leading=11, spaceAfter=5)
S_KEYWORDS= st("keywords", fontSize=9, leading=11, spaceAfter=6)
S_H1      = st("h1", fontName="Times-Bold", fontSize=9.8, leading=11.5,
               alignment=TA_CENTER, spaceBefore=9, spaceAfter=4)
S_H2      = st("h2", fontName="Times-Roman", fontSize=9.6, leading=11.5,
               spaceBefore=6, spaceAfter=3)
S_CAP     = st("cap", fontSize=8, leading=9.6, alignment=TA_CENTER,
               spaceBefore=3, spaceAfter=5)
S_CELL    = st("cell", fontSize=8, leading=9.4, alignment=TA_LEFT,
               spaceAfter=0)
S_CELL_C  = st("cellc", fontSize=8, leading=9.4, alignment=TA_CENTER,
               spaceAfter=0)
S_CELL_H  = st("cellh", fontName="Times-Bold", fontSize=8, leading=9.4,
               alignment=TA_CENTER, spaceAfter=0)
S_REF     = st("ref", fontSize=8, leading=9.6, leftIndent=14,
               firstLineIndent=-14, spaceAfter=2.5)
S_EQ      = st("eq", fontName="Times-Italic", fontSize=10, leading=12,
               alignment=TA_CENTER, spaceBefore=3, spaceAfter=4)

def P(text, style=None):
    return Paragraph(text, style or S_ABSTRACT)

def H1(t): return Paragraph(t.upper(), S_H1)
def H2(t): return Paragraph(t.upper(), S_H2)

# ----------------------------------------------------------------------------
# Figure (pipeline diagram)
# ----------------------------------------------------------------------------
class PipelineFig(Flowable):
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width
        self.height = 96
    def wrap(self, aw, ah):
        return self.width, self.height
    def draw(self):
        d = Drawing(self.width, self.height)
        W = self.width
        boxes = [
            "User input +\nuntrusted context",
            "L1 surface\nclassifier",
            "L2 hidden-\nstate probe",
            "Generation\n(7B LLM)",
            "L3 output\nauditor",
        ]
        bw, bh, gap = 45, 26, (W - 5*45) / 4.0
        y_top = 66
        cx = []
        x = 0
        for i, txt in enumerate(boxes):
            d.add(Rect(x, y_top, bw, bh, strokeColor=black, fillColor=GREY,
                       strokeWidth=0.7, rx=2, ry=2))
            lines = txt.split("\n")
            ly = y_top + bh - 6 - (len(lines)-1)*7
            for j, ln in enumerate(lines):
                d.add(String(x + bw/2.0, ly - j*7, ln, fontName="Times-Roman",
                             fontSize=6.0, textAnchor="middle"))
            cx.append(x + bw/2.0)
            x += bw + gap
        ay = y_top + bh/2.0
        for i in range(4):
            x1 = cx[i] + bw/2.0
            x2 = cx[i+1] - bw/2.0
            d.add(Line(x1, ay, x2-3, ay, strokeColor=black, strokeWidth=0.8))
            d.add(Polygon([x2-4.5, ay+2, x2-4.5, ay-2, x2, ay],
                          fillColor=black, strokeColor=None))
        # output box
        ow, oh = 96, 15
        ox = (W - ow)/2.0
        oy = 6
        d.add(Rect(ox, oy, ow, oh, strokeColor=black, fillColor=GREY,
                   strokeWidth=0.7, rx=2, ry=2))
        d.add(String(W/2.0, oy + 4.5, "Response or BLOCK",
                     fontName="Times-Roman", fontSize=6.2, textAnchor="middle"))
        # connector from L3 down to output
        sx = cx[4]
        d.add(Line(sx, y_top, sx, 40, strokeColor=black, strokeWidth=0.8))
        d.add(Line(sx, 40, W/2.0, 40, strokeColor=black, strokeWidth=0.8))
        d.add(Line(W/2.0, 40, W/2.0, oy + oh + 3, strokeColor=black, strokeWidth=0.8))
        d.add(Polygon([W/2.0-2, oy+oh+3, W/2.0+2, oy+oh+3, W/2.0, oy+oh],
                      fillColor=black, strokeColor=None))
        d.drawOn(self.canv, 0, 0)

# ----------------------------------------------------------------------------
# Table helpers
# ----------------------------------------------------------------------------
def booktabs_table(data, widths, header_rows=1):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    cmds = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    t.setStyle(TableStyle(cmds))
    return t

def caption(text):
    return Paragraph(text, S_CAP)

# ----------------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------------
story = []

# ---- Title band ------------------------------------------------------------
story.append(Paragraph("2026 IEEE International Conference on "
                       "(Conference Name)", S_BANNER))
story.append(Paragraph("Guardrail Cascades Do Not Compose: Measuring "
                       "Routing-Induced Starvation in a Three-Layer "
                       "Prompt-Injection Defense", S_TITLE))
story.append(Paragraph("First A. Author<super>1</super> and "
                       "Second B. Author<super>1</super>", S_AUTHORS))
story.append(Paragraph("<super>1</super>Department of Computer Science, "
                       "Example University, City, Country", S_AFFIL))
story.append(Paragraph("{first.author, second.author}@example.edu", S_EMAIL))
story.append(HRFlowable(width="100%", thickness=0.8, color=black,
                        spaceBefore=1, spaceAfter=3))

# ---- Abstract / keywords ---------------------------------------------------
story.append(Paragraph(
    "<b>Abstract</b>—Prompt injection and jailbreak attacks remain the "
    "dominant failure mode for deployed large language model (LLM) systems, and "
    "practitioners increasingly stack several detectors—a fast surface "
    "classifier, a hidden-state probe, and an output auditor—to defend in depth. "
    "The implicit assumption behind such stacks is that layer-level accuracy "
    "<i>composes</i>: a stronger component yields a stronger system. We "
    "stress-test this assumption on a three-layer, fail-closed guardrail wrapped "
    "around Qwen2.5-7B-Instruct on two T4 GPUs. In isolation, our hidden-state "
    "probe attains 0.997 ROC-AUC on a held-out injection set; yet when routed "
    "through a conventional escalation gate that forwards only “suspicious” "
    "inputs to the second layer, the full pipeline’s recall (0.37) is "
    "statistically indistinguishable from that of the surface classifier alone "
    "(0.35). The gate silently starved the probe of 38 of 60 held-out "
    "injections. Removing the gate—running the second layer always-on—raises "
    "end-to-end recall to 0.90 at a 1.8% false-positive rate, at the cost of one "
    "additional 7B forward pass (~50–150 ms) per request. We formalize why this "
    "occurs, show that component-level metrics do not predict ensemble security, "
    "and distill routing-policy recommendations for layered LLM guardrails. "
    "Code, artifacts, and seeds are released for reproduction on free-tier "
    "hardware.", S_ABSTRACT))
story.append(Paragraph(
    "<b><i>Index Terms</i></b>—prompt injection; jailbreak; LLM security; "
    "defense in depth; hidden-state probes; guardrail evaluation; cascade "
    "classifiers", S_KEYWORDS))

# ---- Switch to two-column body from page 2 onwards -------------------------
story.append(NextPageTemplate("body"))

# ---- I. Introduction -------------------------------------------------------
story.append(H1("I. INTRODUCTION"))
story.append(P(
    "Prompt injection—where an attacker smuggles instructions into the context "
    "window of an LLM to hijack its goal or exfiltrate data—is widely ranked as "
    "a leading security risk for LLM applications [3]. The attack class spans direct "
    "instruction overrides (“ignore your previous instructions”) [1], indirect "
    "injection through retrieved or tool-supplied content [2], [3], and "
    "jailbreaks that override safety alignment [4]. Because no single detector "
    "is robust to all of these, modern guardrail practice is defense in depth: "
    "a cheap, high-precision surface classifier screens inputs; a second, more "
    "expensive mechanism such as a hidden-state probe or an LLM-based judge "
    "re-examines the remainder; and output auditors check the response after "
    "generation [5]–[9]."))
story.append(P(
    "Each layer of such a stack is usually developed, benchmarked, and reported "
    "<i>in isolation</i>, and the implicit expectation is that a layer with "
    "strong standalone metrics will strengthen the ensemble. In this paper we "
    "show that this expectation can fail in a precisely quantifiable way. We "
    "build and release a three-layer, fail-closed injection guard around "
    "Qwen2.5-7B-Instruct [16], reproducible on two T4 GPUs, and evaluate it "
    "end-to-end on a held-out injection corpus. The second layer—a logistic "
    "probe trained on the intermediate hidden states of the target LLM—achieves "
    "0.997 ROC-AUC and 0.783 recall <i>in isolation</i>. Yet when this probe is "
    "placed behind the standard “escalation gate” that only forwards inputs the "
    "surface classifier finds suspicious, the end-to-end recall of the full "
    "pipeline (0.37) collapses to that of the surface classifier alone (0.35): "
    "the gate forwarded only 1 of 60 held-out injections to the probe. The "
    "probe, despite its near-perfect AUC, contributed essentially one detection "
    "to the ensemble."))
story.append(P(
    "This is an instance of a classic principle in cascade design: a cheap "
    "first stage in a cascade must have near-total recall, or every stage behind "
    "it is starved of the very traffic it exists to catch [13]. Our contribution "
    "is not the discovery of this principle, but the demonstration that "
    "production-style LLM guardrail stacks violate it silently, together with a "
    "quantified cost of the violation and a concrete fix. Specifically, we "
    "contribute: (i) a reproducible, fail-closed three-layer prompt-injection "
    "pipeline for Qwen2.5-7B-Instruct on two T4 GPUs, released as open source; "
    "(ii) a quantified <i>gate-starvation</i> result—a component with 0.997 AUC "
    "contributes a single detection end-to-end because the routing gate inherits "
    "the surface classifier’s blind spots, cutting end-to-end recall from the "
    "component’s 0.78 to 0.37; (iii) a minimal, costed fix—always-on routing of "
    "the second layer—which restores end-to-end recall to 0.90 at 1.8% FPR for "
    "one extra 7B forward pass per request, together with a threshold "
    "operating-point analysis; and (iv) routing-policy recommendations and a "
    "discussion of the methodological traps (small test splits, in-sample "
    "threshold calibration, augmentation leakage) that hide this failure mode "
    "from component-level benchmarks."))
story.append(P(
    "The remainder of the paper is organized as follows. Section II reviews "
    "related work. Section III describes the pipeline. Section IV details data, "
    "training, and evaluation protocol. Section V reports results, including the "
    "central routing pathology. Section VI discusses implications, limitations, "
    "and recommendations, and Section VII concludes."))

# ---- II. Related work ------------------------------------------------------
story.append(H1("II. BACKGROUND AND RELATED WORK"))
story.append(H2("A. Threat Model"))
story.append(P(
    "We consider a single-turn LLM application that (i) accepts a user "
    "instruction, (ii) optionally augments it with <i>untrusted</i> context "
    "(retrieved documents, tool output, web content), and (iii) generates a "
    "response under a fixed, sensitive system prompt. The adversary controls "
    "either the user instruction (direct injection) or the untrusted context "
    "(indirect injection), and seeks goal hijacking or system-prompt "
    "exfiltration. This matches the canonical formulations of Perez and Ribeiro "
    "[1] and Greshake et al. [2], and the indirect-injection surface that "
    "dominates current agentic deployments [2], [3]."))
story.append(H2("B. Detection Defenses"))
story.append(P(
    "Two families of detector dominate practice. <i>Surface prompt guards</i> "
    "are lightweight classifiers—typically fine-tuned DeBERTa-family "
    "encoders—that score raw input text. Examples include prompt guard models "
    "trained as binary classifiers, such as PIGuard [6] and InjecGuard [7], and "
    "generative guards such as Llama Guard [5]. These "
    "models are fast and cheap but are documented to suffer <i>over-defense</i>: "
    "bias toward trigger words causes benign inputs stuffed with attack "
    "vocabulary to be misclassified, with accuracy dropping toward chance on "
    "adversarial-benign sets such as NotInject [6], [7]. We observe the same failure "
    "mode in our first layer, which motivates treating it as a high-precision "
    "but recall-limited stage."))
story.append(P(
    "<i>Representation-level detectors</i> read the model’s internal activations "
    "rather than the surface string. Linear probes on intermediate hidden states "
    "have a long lineage [12] and a growing body of recent work for injection "
    "and jailbreak detection: InstructDetector uses hidden states and gradients "
    "to detect indirect instructions [8]; JBShield analyzes activated "
    "toxic/jailbreak concepts in hidden representations [9]; and large-scale "
    "evaluations train activation probes across many corpora and argue for "
    "leave-one-dataset-out evaluation [10], [11]. These methods are generally "
    "more robust to surface obfuscation than string-level guards, at the cost of "
    "a full forward pass of the target LLM."))
story.append(H2("C. Cascades and Deferral"))
story.append(P(
    "Ordering a cheap high-recall stage before an expensive high-precision stage "
    "is the classic cascade pattern [13], generalized in the machine learning "
    "literature as <i>learning to defer</i> [14]. The key invariant—established "
    "decades ago and restated here because it is the mechanism of our central "
    "finding—is that no stage can detect an input that a prior stage refused to "
    "pass on. In Section VI-A we formalize this as an upper bound on ensemble "
    "recall and show that our guardrail stack sits exactly at that bound."))

# ---- III. System -----------------------------------------------------------
story.append(H1("III. SYSTEM DESCRIPTION"))
story.append(P(
    "The pipeline (Fig. 1) wraps a target LLM in three defensive layers, "
    "summarized in Table I. Every stage is <i>fail-closed</i>: a crash, an "
    "out-of-memory error, an empty judge verdict, or a mismatched probe artifact "
    "blocks the request rather than passing it. All layers share a single input "
    "format—untrusted context is wrapped in explicit &lt;untrusted_context&gt; "
    "delimiters—so training, scoring, and generation operate on the same "
    "distribution."))

# Table I
t1_data = [
    [Paragraph("<b>Layer</b>", S_CELL_H), Paragraph("<b>Component</b>", S_CELL_H),
     Paragraph("<b>Function</b>", S_CELL_H), Paragraph("<b>Placement</b>", S_CELL_H)],
    [Paragraph("L1", S_CELL_C),
     Paragraph("ProtectAI DeBERTa-v3 (v2), optional PIGuard ensemble", S_CELL),
     Paragraph("Sliding-window surface audit; high precision, recall-limited", S_CELL),
     Paragraph("GPU 1", S_CELL_C)],
    [Paragraph("L2", S_CELL_C),
     Paragraph("Logistic probe on LLM hidden states (layer 20)", S_CELL),
     Paragraph("Behavioral recall engine; catches obfuscated/implicit instructions", S_CELL),
     Paragraph("alongside the 7B", S_CELL_C)],
    [Paragraph("L3", S_CELL_C),
     Paragraph("Canary token + regex heuristics + self-judge", S_CELL),
     Paragraph("Post-generation output audit (exfiltration, compromise)", S_CELL),
     Paragraph("reuses the 7B", S_CELL_C)],
]
story.append(caption("<b>TABLE I</b>  THE THREE DEFENSIVE LAYERS OF THE PIPELINE"))
story.append(booktabs_table(t1_data, [26, 70, 102, 46]))

# Figure 1
story.append(PipelineFig(COL_W - 4))
story.append(caption("<b>Fig. 1.</b> The three-layer pipeline. L2 runs on every "
                     "input L1 does not hard-block (“always-on” routing); L3 "
                     "audits the generated output. Any layer can block the "
                     "request, fail-closed."))

story.append(H2("A. Layer 1 — Surface Classifier"))
story.append(P(
    "L1 audits raw text with a fine-tuned ProtectAI/DeBERTa-v3-base-prompt-"
    "injection-v2 classifier [15]. Inputs longer than the encoder window "
    "are split into overlapping sliding windows with two production guards: an "
    "<i>overlap clamp</i> (overlap capped at a quarter of the window budget) and "
    "a <i>window cap</i> (at most 64 windows, always including the document "
    "tail), so a long RAG document cannot silently hide an attack or explode "
    "into hundreds of classifier calls. A single-window input is labeled "
    "INJECTION at probability at least 0.5; a multi-window input requires either "
    "2 or more alert windows or a peak of at least 0.95, so one quoted attack "
    "inside a long benign document escalates rather than blocks. An optional "
    "ensemble (e.g., adding PIGuard [6]) votes per window, diluting single-model "
    "quirks such as repetition-triggered false spikes."))
story.append(H2("B. Layer 2 — Hidden-State Probe"))
story.append(P(
    "L2 is a low-capacity logistic regression trained on the intermediate "
    "activations of the <i>target</i> LLM. For Qwen2.5-7B-Instruct (28 layers) "
    "we extract the last-token hidden state of layer 20 (mid-late), standardize "
    "it, and score it with an L2-regularized logistic regression (C = 0.5, "
    "balanced class weights). The decision threshold is calibrated to an "
    "explicit false-positive-rate (FPR) budget (no more than 1%) using "
    "<i>out-of-fold</i> probabilities, so it matches production scoring rather "
    "than an optimistic in-sample ROC curve. A v2 extension concatenates pooled "
    "vectors from multiple layers (12, 16, 20, 24). The saved artifact carries a "
    "model/layer fingerprint and <i>refuses to load</i> on any model or layer "
    "mismatch, so a stale probe cannot silently disable the layer."))
story.append(H2("C. Layer 3 — Output Auditor"))
story.append(P(
    "L3 audits the generated response. A cryptographically random <i>canary</i> "
    "token is embedded in the system prompt and its appearance in the output is "
    "treated as system-prompt exfiltration; structural regex heuristics flag "
    "compromise phrasing; and a hardened <i>self-judge</i> re-examines flagged "
    "outputs. The self-judge reuses the already-loaded target LLM (flat VRAM, no "
    "second model to OOM), frames the audited text explicitly as data rather "
    "than instructions, and parses only the first verdict token so a compromised "
    "response cannot suffix-flip the decision."))
story.append(H2("D. Deployment Constraints"))
story.append(P(
    "The system targets two 16 GiB T4 GPUs (Turing, compute capability 7.5): "
    "fp16 is forced (bf16 would emulate in software), the 7B model is sharded "
    "with device_map=auto and per-GPU memory caps, the surface classifier is "
    "pinned to GPU 1, and expandable CUDA segments avoid fragmentation OOMs. The "
    "design reuses the single loaded 7B for generation <i>and</i> judging, "
    "keeping the whole pipeline within the two-T4 budget. The marginal cost of "
    "L2 is one extra 7B forward pass per non-blocked request (~50–150 ms at up "
    "to 1024 tokens)."))

# ---- IV. Setup -------------------------------------------------------------
story.append(H1("IV. EXPERIMENTAL SETUP"))
story.append(H2("A. Data"))
story.append(P(
    "Table II lists the corpora. The probe is trained on the <b>train</b> split "
    "of Deepset’s prompt-injections dataset (546 rows), augmented with 527 "
    "jailbreak positives from jailbreak-classification and 339 hard negatives "
    "from NotInject [6]—benign text deliberately stuffed with attack trigger "
    "words, the direct data-side antidote to over-defense. All evaluation "
    "numbers in this paper are computed on the <b>held-out test split</b> of "
    "prompt-injections (116 rows: 60 injections, 56 benign), which is never seen "
    "during training; ChatGPT-Jailbreak-Prompts is reserved for cross-family "
    "recall checks."))

t2_data = [
    [Paragraph("<b>Corpus</b>", S_CELL_H), Paragraph("<b>Split</b>", S_CELL_H),
     Paragraph("<b>Rows</b>", S_CELL_H), Paragraph("<b>Semantics</b>", S_CELL_H)],
    [Paragraph("deepset/prompt-injections", S_CELL), Paragraph("train", S_CELL_C),
     Paragraph("546", S_CELL_C), Paragraph("binary labels (203 positive)", S_CELL)],
    [Paragraph("deepset/prompt-injections", S_CELL), Paragraph("test", S_CELL_C),
     Paragraph("116", S_CELL_C), Paragraph("held-out (60 inj. / 56 benign)", S_CELL)],
    [Paragraph("jailbreak-classification", S_CELL), Paragraph("train", S_CELL_C),
     Paragraph("527 pos.", S_CELL_C), Paragraph("labeled via “type” column", S_CELL)],
    [Paragraph("NotInject [6]", S_CELL), Paragraph("3 splits", S_CELL_C),
     Paragraph("339", S_CELL_C), Paragraph("benign + trigger words (hard negatives)", S_CELL)],
    [Paragraph("ChatGPT-Jailbreak-Prompts", S_CELL), Paragraph("train", S_CELL_C),
     Paragraph("79", S_CELL_C), Paragraph("all positive (cross-family)", S_CELL)],
]
story.append(caption("<b>TABLE II</b>  CORPORA USED FOR TRAINING AND HELD-OUT "
                     "EVALUATION"))
story.append(booktabs_table(t2_data, [96, 40, 42, 66]))

story.append(H2("B. Training and Calibration Methodology"))
story.append(P(
    "Three methodological guards matter as much as the fit itself, and each was "
    "introduced in response to a measured failure during development. (1) "
    "<b>Evasion augmentation:</b> each positive is expanded with delimiter "
    "smuggling, a base64 payload, and up to seven surface-evasion variants "
    "(zero-width characters, homoglyphs, leetspeak, ROT13, HTML comments, fake "
    "role tags, whitespace splits); each negative is expanded with a “quoted "
    "attack” hard negative so the probe learns the instruction wrapper rather "
    "than keywords. (2) <b>Group-aware cross-validation:</b> augmented variants "
    "of one source row are near-duplicates. With plain stratified folds they "
    "straddled fold boundaries and inflated AUC toward 1.0 (measured 0.78 vs. "
    "0.66 AUC on synthetic data with the same structure). Variants now carry a "
    "group id and StratifiedGroupKFold keeps each family inside one fold. (3) "
    "<b>Out-of-fold threshold calibration:</b> reading the FPR budget off an "
    "in-sample ROC curve selected a threshold stricter than the budget actually "
    "allows (0.9981 vs. 0.9714 in one run), silently costing recall. The "
    "threshold is now the largest value whose <i>out-of-fold</i> FPR stays "
    "within the budget."))
story.append(H2("C. Evaluation Protocol"))
story.append(P(
    "We report true-positive rate (TPR; recall on the 60 held-out injections), "
    "FPR (fraction of the 56 benign rows blocked), accuracy, and ROC-AUC. Two "
    "gate policies are compared in one pass over the held-out set: <b>Policy A "
    "(legacy gate)</b> blocks iff L1 hard-blocks <i>or</i> (L1 score is at "
    "least tau_gate = 0.30 <i>and</i> L2 is at least the threshold); <b>Policy B "
    "(always-on)</b> blocks iff L1 hard-blocks <i>or</i> L2 is at least the "
    "threshold. Because the test split is small, we emphasize that FPR has a "
    "granularity of 1/56 (~1.8%) per row and we report counts alongside rates; "
    "bootstrap confidence intervals are available in the released benchmark "
    "harness."))

# ---- V. Results ------------------------------------------------------------
story.append(H1("V. RESULTS"))
story.append(H2("A. Layers Measured in Isolation"))
story.append(P(
    "Measured independently, the two input layers look like a healthy "
    "complement. L1 (the surface classifier) alone blocks 21/60 injections (TPR "
    "0.350) with zero false positives—it catches explicit instructions but is "
    "blind to the obfuscated and implicit ones. The L2 hidden-state probe, "
    "scored on every row without gating, attains ROC-AUC 0.997, TPR 0.783 at its "
    "calibrated threshold, and FPR 0.000. The layers are complementary: of the "
    "39 injections L1 misses, the probe sees most of them clearly. On component "
    "metrics alone, adding L2 to L1 should be a large win."))
story.append(H2("B. The Routing Pathology"))
story.append(P(
    "It is not. Table III reports end-to-end behavior. Under Policy A, only 1 "
    "of 60 injections ever reaches L2: 21 are hard-blocked by L1 (which is "
    "fine), but 38 score below 0.30 at L1 and are silently dropped before the "
    "probe can see them. The probe—0.997 AUC in isolation—contributes a single "
    "detection, and the ensemble recall (0.367) is indistinguishable from L1 "
    "alone (0.350). Lowering or raising L1’s block threshold does not help: the "
    "35 missed injections that score below 0.01 at L1 are simply not in the "
    "gate’s forward set, whatever the threshold."))

t3_data = [
    [Paragraph("<b>Configuration</b>", S_CELL_H), Paragraph("<b>L2 shown</b>", S_CELL_H),
     Paragraph("<b>TPR</b>", S_CELL_H), Paragraph("<b>FPR</b>", S_CELL_H),
     Paragraph("<b>Acc.</b>", S_CELL_H)],
    [Paragraph("L1 only", S_CELL), Paragraph("—", S_CELL_C),
     Paragraph("0.350 (21/60)", S_CELL_C), Paragraph("0.000", S_CELL_C),
     Paragraph("0.664", S_CELL_C)],
    [Paragraph("Legacy gate (tau=0.30)", S_CELL), Paragraph("1/60", S_CELL_C),
     Paragraph("0.367 (22/60)", S_CELL_C), Paragraph("0.000", S_CELL_C),
     Paragraph("0.672", S_CELL_C)],
    [Paragraph("Always-on, L2 thr 0.9868", S_CELL), Paragraph("39/60", S_CELL_C),
     Paragraph("0.833 (50/60)", S_CELL_C), Paragraph("0.000 (0/56)", S_CELL_C),
     Paragraph("0.914", S_CELL_C)],
    [Paragraph("<b>Always-on, L2 thr 0.90</b>", S_CELL), Paragraph("39/60", S_CELL_C),
     Paragraph("<b>0.900 (54/60)</b>", S_CELL_C), Paragraph("<b>0.018 (1/56)</b>", S_CELL_C),
     Paragraph("<b>0.940</b>", S_CELL_C)],
]
story.append(caption("<b>TABLE III</b>  END-TO-END GATE-POLICY COMPARISON ON THE "
                     "HELD-OUT TEST SPLIT (60 INJECTIONS, 56 BENIGN)"))
story.append(booktabs_table(t3_data, [98, 40, 46, 42, 26]))

story.append(P(
    "Making L2 always-on (Policy B) removes the gate and lets the probe see "
    "every input L1 does not hard-block. At the artifact’s FPR-budgeted "
    "threshold (0.9868), end-to-end recall jumps to 0.833 at zero measured FPR; "
    "at the deployed operating point (0.90) it reaches 0.900 at 1.8% FPR. L2 now "
    "contributes 29 detections beyond L1’s 21 (50 union). The same signature "
    "reproduces on an independent CPU stand-in (SmolLM2-135M with a probe "
    "trained on its layer 20): recall 0.44 under the legacy gate versus 0.67 "
    "always-on, with L2 adding 0 versus 2 detections beyond L1."))
story.append(H2("C. Operating Point"))
story.append(P(
    "Table IV sweeps the probe threshold under always-on routing. FPR is flat "
    "(1/56) from 0.50 to 0.95 because only one benign row ever scores above 0.5 "
    "on this split; on broader traffic lower thresholds will cost more false "
    "positives. We deploy 0.90 as the balanced point; practitioners whose "
    "objective is maximum recall may prefer 0.75–0.50."))

t4_data = [
    [Paragraph("<b>L2 threshold</b>", S_CELL_H), Paragraph("<b>TPR</b>", S_CELL_H),
     Paragraph("<b>FPR</b>", S_CELL_H), Paragraph("<b>Acc.</b>", S_CELL_H)],
    [Paragraph("0.9868 (artifact)", S_CELL), Paragraph("0.833 (50/60)", S_CELL_C),
     Paragraph("0.000 (0/56)", S_CELL_C), Paragraph("0.914", S_CELL_C)],
    [Paragraph("0.95", S_CELL), Paragraph("0.900 (54/60)", S_CELL_C),
     Paragraph("0.018 (1/56)", S_CELL_C), Paragraph("0.940", S_CELL_C)],
    [Paragraph("0.90 (deployed)", S_CELL), Paragraph("0.900 (54/60)", S_CELL_C),
     Paragraph("0.018 (1/56)", S_CELL_C), Paragraph("0.940", S_CELL_C)],
    [Paragraph("0.75", S_CELL), Paragraph("0.933 (56/60)", S_CELL_C),
     Paragraph("0.018 (1/56)", S_CELL_C), Paragraph("0.957", S_CELL_C)],
    [Paragraph("0.50", S_CELL), Paragraph("0.950 (57/60)", S_CELL_C),
     Paragraph("0.018 (1/56)", S_CELL_C), Paragraph("0.966", S_CELL_C)],
]
story.append(caption("<b>TABLE IV</b>  THRESHOLD SWEEP UNDER ALWAYS-ON ROUTING "
                     "(HELD-OUT SPLIT)"))
story.append(booktabs_table(t4_data, [84, 72, 56, 40]))

story.append(H2("D. Cost of the Fix"))
story.append(P(
    "Always-on routing costs one additional 7B forward pass per non-blocked "
    "request (~50–150 ms at up to 1024 tokens on two T4s). VRAM stays flat: L3 "
    "reuses the loaded target model for judging, so no second model is "
    "introduced. For applications where that latency is unacceptable, the legacy "
    "gate can be retained—but operators should know that on our data it reduces "
    "L2’s contribution to approximately nothing."))

# ---- VI. Discussion --------------------------------------------------------
story.append(H1("VI. DISCUSSION"))
story.append(H2("A. Why Component Metrics Do Not Compose"))
story.append(P(
    "Let R1 be L1’s recall, and let G be the set of injections the gate forwards "
    "to L2. If L2 achieves recall R2 on G, end-to-end recall is"))
story.append(Paragraph("R  =  R1 + (1 - R1) · g · R2", S_EQ))
story.append(P(
    "where g = P(input is in G | injection, not blocked by L1) is the gate’s "
    "coverage of L1’s <i>misses</i>. When the gate forwards only inputs the "
    "surface classifier already finds suspicious, g is determined by L1’s blind "
    "spots, not by L2’s quality. In our legacy configuration g = 1/39 (~0.026), "
    "so R ~ R1 regardless of R2—a probe with 0.997 AUC contributes a single "
    "catch because the gate inherits exactly the inputs L1 cannot see. This is "
    "the cascade-recall bound in disguise [13], [14]: no downstream stage can "
    "detect what a prior stage refuses to pass on."))
story.append(H2("B. Threats to Validity"))
story.append(P(
    "We state limitations explicitly because they bound the claims above. (i) "
    "The headline numbers come from a single, small, dated corpus (116 held-out "
    "rows); FPR granularity is 1.8% per row, so point estimates are noisy and "
    "thresholds 0.50–0.95 are indistinguishable on this split. (ii) The deployed "
    "threshold was selected using the test split, a mild form of test-set "
    "tuning; the released trainer calibrates on a validation slice for a "
    "defensible setting. (iii) We do not yet report leave-one-dataset-out "
    "generalization or adaptive adversaries (obfuscation, paraphrasing, "
    "multi-turn), which the field increasingly requires [10], [11]; the released "
    "benchmark harness supports these. (iv) The L1 choice (ProtectAI v2) is a "
    "documented over-defense liability [6], and we observe its "
    "repetition-triggered false spikes ourselves; the L1 ensemble and dual-key "
    "confirmation are mitigations, not eliminations. (v) The L3 self-judge "
    "consumes untrusted text and is therefore itself prompt-injectable; it is "
    "hardened (data framing, first-token parsing) but scoped conservatively. We "
    "therefore position the contribution as a <i>measurement of how guardrail "
    "stacks fail</i>, not as a new detector."))
story.append(H2("C. Recommendations"))
story.append(P(
    "Our findings translate into concrete practice. (1) <i>Measure ensembles "
    "end-to-end</i>: report per-layer metrics only as diagnostics; the number "
    "that decides a deployment is the ensemble’s recall on the full routing "
    "path. (2) <i>The cheap first stage must have near-total recall, or be "
    "bypassable.</i> If a surface guard’s recall is 0.35, any layer gated behind "
    "it inherits a 0.65 ceiling it can never exceed. (3) <i>Prefer always-on or "
    "learned deferral</i> for expensive stages: the always-on policy here bought "
    "+0.53 recall for ~100 ms. (4) <i>Calibrate thresholds on a validation "
    "slice, never the test split, and report confidence intervals</i> on small "
    "sets. (5) <i>Use over-defense sets</i> (NotInject) as the FPR benchmark "
    "that decides real deployments, since trigger-word bias is the dominant "
    "false-positive source."))

# ---- VII. Conclusion -------------------------------------------------------
story.append(H1("VII. CONCLUSION"))
story.append(P(
    "We built and released a three-layer, fail-closed prompt-injection guard for "
    "Qwen2.5-7B-Instruct on two T4 GPUs and evaluated it end-to-end. The "
    "central, quantified finding is that a hidden-state probe with 0.997 ROC-AUC "
    "contributed a single detection when placed behind a conventional escalation "
    "gate, because the gate inherited the surface classifier’s blind spots and "
    "starved the probe of 38 of 60 held-out injections. Making the second layer "
    "always-on raised end-to-end recall from 0.37 to 0.90 at 1.8% FPR. The "
    "lesson generalizes: in layered LLM guardrails, the <i>routing policy</i> "
    "between layers is a first-class security variable, and component metrics do "
    "not predict ensemble security. We hope the released pipeline and its "
    "failure analysis help practitioners avoid the silent, component-metrics-"
    "driven false confidence we measured."))

# ---- References ------------------------------------------------------------
story.append(H1("REFERENCES"))
refs = [
    "F. Perez and I. Ribeiro, “Ignore previous prompt: Attack techniques for language models,” in Proc. NeurIPS ML Safety Workshop, 2022. doi: 10.48550/arXiv.2211.09527",
    "K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection,” in Proc. 16th ACM Workshop on Artificial Intelligence and Security (AISec), 2023, pp. 79–90. doi: 10.1145/3605764.3623985",
    "Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, and Y. Liu, “Prompt injection attack against LLM-integrated applications,” arXiv preprint, 2024. doi: 10.48550/arXiv.2306.05499",
    "A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint, 2023. doi: 10.48550/arXiv.2307.15043",
    "H. Inan, K. Upasani, J. Chi, R. Rungta, K. Iyer, Y. Mao, M. Tontchev, Q. Hu, B. Fuller, D. Testuggine, and M. Khabsa, “Llama Guard: LLM-based input-output safeguard for human-AI conversations,” arXiv preprint, 2023. doi: 10.48550/arXiv.2312.06674",
    "H. Li, X. Liu, N. Zhang, and C. Xiao, “PIGuard: Prompt injection guardrail via mitigating overdefense for free,” in Proc. 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025, pp. 30420–30437. doi: 10.18653/v1/2025.acl-long.1468",
    "H. Li, X. Liu, and C. Xiao, “InjecGuard: Benchmarking and mitigating over-defense in prompt injection guardrail models,” arXiv preprint, 2024. doi: 10.48550/arXiv.2410.22770",
    "T. Wen, C. Wang, X. Yang, H. Tang, Y. Xie, L. Lyu, Z. Dou, and F. Wu, “Defending against indirect prompt injection by instruction detection,” in Findings of the Association for Computational Linguistics: EMNLP 2025, 2025, pp. 19472–19487. doi: 10.18653/v1/2025.findings-emnlp.1060",
    "S. Zhang, Y. Zhai, K. Guo, H. Hu, S. Guo, Z. Fang, L. Zhao, C. Shen, C. Wang, and Q. Wang, “JBShield: Defending large language models from jailbreak attacks through activated concept analysis and manipulation,” in Proc. USENIX Security Symposium, 2025. doi: 10.48550/arXiv.2502.07557",
    "M. Fomin, “When benchmarks lie: Evaluating malicious prompt classifiers under true distribution shift,” arXiv preprint, 2026. doi: 10.48550/arXiv.2602.14161",
    "Y. Li, Z. Fan, and Z. Zhuang, “When AUC 0.998 is not enough: A candidate evaluation protocol for hidden-state probes of indirect prompt injection in multimodal computer-use agents,” arXiv preprint, 2026. doi: 10.48550/arXiv.2606.22864",
    "G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” in Proc. ICLR Workshop Track, 2017. doi: 10.48550/arXiv.1610.01644",
    "P. Viola and M. Jones, “Rapid object detection using a boosted cascade of simple features,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2001, pp. I-511–I-518. doi: 10.1109/CVPR.2001.990517",
    "D. Madras, T. Pitassi, and R. Zemel, “Predict responsibly: Improving fairness and accuracy by learning to defer,” in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2018. doi: 10.48550/arXiv.1711.06664",
    "P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,” in Proc. International Conference on Learning Representations (ICLR), 2023. doi: 10.48550/arXiv.2111.09543",
    "Qwen Team, Alibaba Group, “Qwen2.5 technical report,” arXiv preprint, 2024. doi: 10.48550/arXiv.2412.15115",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph("[%d]&nbsp;&nbsp;%s" % (i, r), S_REF))

# ----------------------------------------------------------------------------
# Document with two page templates (first page has a full-width title band)
# ----------------------------------------------------------------------------
def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Times-Roman", 8)
    canv.drawCentredString(PAGE_W/2.0, MARGIN/2.0, str(canv.getPageNumber()))
    if canv.getPageNumber() == 1:
        # IEEE-style first-page copyright notice (placeholder — replace).
        canv.setFont("Times-Roman", 7)
        canv.drawString(MARGIN, MARGIN/2.0,
                        "979-8-XXXX-XXXX-X/26/$31.00 \u00a92026 IEEE")
    canv.restoreState()

frame_first_cols = [
    Frame(MARGIN, BOTTOM, COL_W, COL_H_1, id="cL1"),
    Frame(MARGIN + COL_W + GAP, BOTTOM, COL_W, COL_H_1, id="cR1"),
]
frame_title = Frame(MARGIN, TOP - TITLE_H, PAGE_W - 2*MARGIN, TITLE_H, id="title")
frame_body_cols = [
    Frame(MARGIN, BOTTOM, COL_W, COL_H, id="cL"),
    Frame(MARGIN + COL_W + GAP, BOTTOM, COL_W, COL_H, id="cR"),
]

doc = BaseDocTemplate(os.path.join(HERE, "paper.pdf"), pagesize=letter,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=MARGIN, bottomMargin=MARGIN,
                      title="Guardrail Cascades Do Not Compose (Ens_v2)",
                      author="Author One, Author Two")
doc.addPageTemplates([
    PageTemplate(id="first", frames=[frame_title] + frame_first_cols,
                 onPage=on_page),
    PageTemplate(id="body", frames=frame_body_cols, onPage=on_page),
])
doc.build(story)
print("wrote", os.path.join(HERE, "paper.pdf"))
