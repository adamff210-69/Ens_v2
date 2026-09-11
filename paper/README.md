# IEEE Conference Paper — Ens_v2 (Prompt-Injection Detection Pipeline)

This folder contains the IEEE-conference paper written from the `Ens_v2`
repository (a three-layer prompt-injection detection pipeline). **The project
repository itself was not modified** — everything here lives in this separate
`paper/` folder.

## Files

| File | Purpose |
|------|---------|
| `paper.tex` | **Authoritative IEEEtran LaTeX source** (IEEE conference two-column class). Compile on Overleaf or with `pdflatex paper.tex` (run twice for cross-references). |
| `paper.pdf` | A rendered, two-column IEEE-style PDF (visual preview of `paper.tex`). |
| `render_paper.py` | Script that regenerates `paper.pdf` (pure-Python reportlab; used because no TeX distribution is available in this sandbox). |
| `BASE_PAPERS.md` | The **base & reference papers** (research papers only, with verified DOIs) the project builds on, and a note on how each maps to the project. |
| `README.md` | This file. |

## Reference list policy

The reference list contains **research papers only** — no model cards, dataset
pages, or vendor web pages. Non-paper artifacts named in the text (ProtectAI's
DeBERTa-v3 injection model, the Deepset `prompt-injections` dataset, Meta's
Llama Prompt Guard 2) are named as proper nouns and cited through the research
papers that document them. Every reference has a DOI that was verified to
resolve against the publisher's record (Crossref / DataCite / ACL Anthology /
arXiv) on 2026-09-11; each DOI appears in both `paper.tex` and the PDF.

## Before submitting

1. **Replace the placeholder author block** in `paper.tex` (title/author/email
   section near the top) with real names, affiliations, and e-mail addresses.
2. **Replace the two formatting placeholders** in `render_paper.py`: the
   conference banner line `2026 IEEE International Conference on (Conference
   Name)` and the first-page copyright notice `979-8-XXXX-XXXX-X/26/$31.00
   ©2026 IEEE` (the publisher assigns the real copyright string at acceptance).
3. **Choose the target IEEE conference** and adjust length if needed (the draft
   is ~5 pages, two-column, which fits most IEEE conference tracks).
4. Re-verify the numbers against a fresh run of the repository's
   `evaluate_end_to_end.py` / `benchmark.py` before camera-ready, since the
   figures in the paper are the ones recorded in `PROJECT_STATUS.md` /
   `README.md` (held-out `deepset/prompt-injections` test split).

## IEEE format notes

- `paper.tex` uses the official `IEEEtran` conference class, which natively
  produces the standard IEEE conference layout (full-width title/author block,
  `Abstract—` and `Index Terms—`, Roman-numeral small-caps section headings,
  two-column body, numbered references).
- `paper.pdf` is a visual rendering of the same layout generated with
  reportlab (no TeX in this sandbox): centered bold title, author block with
  affiliation superscripts, `Abstract—`/`Index Terms—` labels, centered
  uppercase section headings, uppercase table captions above each table
  (`TABLE I`, `TABLE II`, …), `Fig. 1.` caption below the figure, and a
  first-page IEEE copyright notice.

## How to compile the LaTeX

Overleaf: create a new project, upload `paper.tex`, set the compiler to
`pdflatex`. All packages used (`cite`, `amsmath`, `graphicx`, `booktabs`,
`tikz`, `xcolor`, `hyperref`) are standard in TeX Live / Overleaf.

Locally: `pdflatex paper.tex && pdflatex paper.tex`.

## Regenerating the PDF preview

```bash
python3 -m venv .venv-paper
.venv-paper/bin/pip install reportlab
.venv-paper/bin/python render_paper.py
```

## Positioning note (honesty)

The repository's own `PUBLICATION_ROADMAP.md` is candid that the *detector*
itself is not novel (hidden-state probes are an established line of work).
Accordingly, this paper is framed as a **measurement/systems contribution** —
the central claim is the *gate-starvation* finding:

> A hidden-state probe with **0.997 ROC-AUC contributed a single detection**
> end-to-end because the escalation gate inherited the surface classifier's
> blind spots; making the second layer always-on raised end-to-end recall from
> **0.37 → 0.90** at a **1.8% FPR**.

Limitations (single small dataset, threshold selected on the test split, no
adaptive adversary, known L1 over-defense) are stated explicitly in
Section VI-B, per the roadmap's guidance.
