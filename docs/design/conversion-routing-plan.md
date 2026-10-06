# Conversion routing v2: a layered profiler, four runways, a gate that knows when to escalate

Status: **proposal, nothing built.** Written after 0.9.22. It changes stage 3 (Convert) of the indexing pipeline;
stages 1-2 and 4-8, the page cache, the trace format's spirit and the repair (3.4) and reconcile (3.5) steps stay.
Two decisions are deliberately left to the end of this work (section 9): the docling fallback policy and the models
for machines without Apple Silicon.

## 1. Why

Measured on the owner's documents (see `document-conversion-plan.md`, "Status after 0.9.1-0.9.22"):

- Every page without a usable text layer goes to the vision model (VLM), even a clean scan of plain prose that a
  conventional OCR engine reads in a second and cannot loop on.
- The 4B reader loops on dense pages: 56 of 578 image-read pages were runaways, and 49 of them passed the gate
  before 0.9.18. The gate, not the router, is the weakest link.
- The profiler never looks at the page image. It cannot tell a clean scan from a photograph or a ruled table,
  cannot tell a scanner's hidden OCR layer (often good) from a broken one, and cannot see content that a text
  layer leaves out (text drawn as outlines, text inside a picture).
- Repair with the 8B model is the most expensive step; it should run only on pages that really need it.

### Baseline: the owner's last full run (0.9.22, dashboard figures)

2,145 documents, 19,723 pages in successfully converted files, 17,772 of them reused from the page cache.

| Pages by kind | | Outcome | |
|---|---|---|---|
| digital (text layer) | 16,448 (83 %) | pass | 17,846 |
| scanned (VLM) | 1,496 | low | 1,505 (7.6 %, in 487 documents) |
| image files | 231 | no text | 293 |
| embedded pictures | 394 | repaired | 92 |
| scanned by docling OCR | 45 | | |
| text, office | 720, 389 | | |

Gate findings: table shape 639, coverage 435, low resolution 262, docling grade 105, totals 84, script 63, running
balance 18, degenerate 16. Repair: 0 of 13 suspect cells fixed. Times: convert 27 h 56 min, of which 3.2 Read
13 h 39 min and 3.3 Gate 4 h 30 min; document reader 2.09 M tokens in 17 h 26 min (33 tokens/s); profile 1 min 11 s;
CPU 4 h 41 min.

What it says for this plan:
- **The VLM is the cost.** About 1,700 scan and image pages, plus 394 picture reads, take most of the reading time.
  The owner says most scans are English, so 3.2b is aimed at the expensive part. The page cache matters too: a run
  over unchanged documents reuses pages, so the saving shows on new documents and whenever a setting re-converts them.
- **3.1A costs nothing** (71 s for 19,736 pages). 3.1B only has to run on the roughly 3,000 pages that are not plainly
  digital.
- **The gate flags many digital pages that nothing acts on.** Repair handles scans only, and table shape (639) and
  coverage (435) are probably mostly digital pages. Either docling loses content on them (the ink-residue and
  layer-agreement checks would show it), or the checks are noisy. This has to be found out before the gate decides
  more routes.
- **The gate is slow:** 4 h 30 min, a third of the read time, so about 0.8 s a page on average. Find out why (it may
  re-check reused pages) before adding checks.
- **Cell repair did nothing here** (0 of 13); the 92 repaired pages came from the 8B page re-read.

## 2. Philosophy

**When sure, read with docling (with or without OCR). When in doubt, read with the VLM.** Refined:

1. *Sure* is a calibrated statement, not a feeling: a cheap runway is chosen only when the profile's evidence
   for it clears thresholds that the routing harness (section 6) has measured against verified pages. Every
   other page is "in doubt".
2. The two kinds of mistake cost differently. Sending an easy page to the VLM costs time. Sending a hard page to
   docling costs *silently wrong text*, which is only caught if the gate catches it. So a cheap runway is allowed
   only where the gate has a check that would catch that runway's typical failure (section 5.2), and the router
   errs towards the VLM.
3. Doubt is also capability: a page whose script the configured OCR engine cannot read (Marathi and Hindi with
   Apple Vision on the owner's macOS) is in doubt for runway b however clean it is.
4. Every decision is recorded with its reasons and features, so a wrong route can be found afterwards and the
   thresholds tuned without guessing.

## 3. The pipeline

Picture: [`conversion-routing-v2.svg`](conversion-routing-v2.svg).

```
3  CONVERT (per document)
   3.1  PROFILE  -- per page; stops at the first layer that makes the route clear
        3.1A  document facts      pypdfium2 objects, no rendering                       ~ms / page
        3.1B  page-image facts    one render at 100-150 dpi, numpy + Pillow              ~50-150 ms / page
        3.1C  probes (hooks)      optional light tools / models, only for pages A+B leave open, budgeted
        route = router.decide(features, capabilities) -> runway, confidence, reasons
   3.2  READ     -- one of four runways per page (blank pages are not read)
        3.2a  docling, no OCR           trustworthy, complete text layer
        3.2b  docling + OCR             no usable layer; clean print in a supported script; no hard structure
        3.2c  docling + VLM on regions  good layer plus large pictures / unexplained ink regions
        3.2d  VLM, whole page           everything in doubt: photos, complex tables, handwriting, poor scans,
                                        unsupported script, unknown
   3.3  GATE     -- graded, located verdict per page -> pass | escalate | repair | low
        escalation: 3.2a/b/c failure -> 3.2d (once); 3.2d failure -> 3.4
   3.4  REPAIR   (unchanged role: heavier VLM, cell and page re-reads)
   3.5  RECONCILE (unchanged)
```

Non-PDF inputs: image files get 3.1B/3.1C and go to 3.2b or 3.2d (today they always go to the VLM); Office/HTML
stay with docling as a whole (3.2a, no change); Markdown/text are copied.

Renumbering note: today's labels are 3.2a docling, 3.2b VLM scans, 3.2c pictures. In v2, 3.2b becomes docling+OCR
and the VLM page reader becomes 3.2d. Stored traces use branch codes (`digital`, `raster`, `embedded`, `fallback`),
not stage labels, so old records stay readable; the dashboard labels and `stages.py` change together.

## 4. 3.1 Profile

### 4.1 3.1A document facts (pypdfium2, no render)

Today: characters, `text_ok`, picture cover, picture dpi, rotation, script, `hidden_ocr_layer` (a guess: text over a
full-page picture), ink and hash at 36 dpi for near-empty pages. Added:

| Feature | How | What it decides |
|---|---|---|
| Invisible text (render mode 3) | text objects' render mode | an exact hidden-OCR-layer flag instead of the guess (pypdfium2 support to be verified) |
| Fonts without a Unicode map; private-use code points | font objects, extracted text | a layer that decodes to garbage, 3.2a not trustworthy |
| Character boxes | text page char boxes | input to 3.1B's ink-residue check; reading-order sanity |
| Vector rules | line / rectangle path objects | ruled table present: needs docling TableFormer (3.2a/c), not plain text |
| Column structure | x-gaps of character boxes | multi-column layout (affects reading order checks) |
| Text-layer plausibility | word-validity share per script (wordlists) | a layer that is text-shaped but nonsense |

A page is *decided at 3.1A* when it has a plausible visible text layer, no pictures over a threshold, no
unexplained-content risk signals: route 3.2a, skip rendering. On born-digital documents this should be most pages,
which keeps the profile cheap.

### 4.2 3.1B page-image facts (one render, numpy + Pillow)

| Feature | How | What it decides |
|---|---|---|
| Ink residue | ink mask minus character boxes minus picture boxes | content the text layer does not hold: 3.2c (region) or 3.2d (most of the page) |
| Skew | projection-profile variance over small angles | deskew before OCR; heavy skew lowers confidence for 3.2b |
| Blur | variance of the Laplacian | soft scans are in doubt |
| Contrast, noise | histogram spread, speckle density | poor scans are in doubt |
| Text-line regularity | horizontal projection peaks | printed text lines vs photo vs handwriting-like irregularity |
| Ruled lines | long horizontal/vertical runs (morphology on the binary image) | table on a scan: 3.2d (docling OCR is weak on scanned tables) |
| Photo-likeness | colour spread, large smooth non-text areas, edge density | photographed page or picture: 3.2d |
| Effective dpi | page pixels vs physical size | below ~150 dpi in doubt |

Rendering is done once and kept for the read step when the page will be read as an image (3.2b/d already render).

### 4.3 3.1C probes (hooks; nothing on by default in the first version)

A probe is a small pluggable step with one interface: `applies(features) -> bool`, `run(image, features) -> features`,
a time budget, a version string that goes into the record. Probes run only for pages that 3.1A+B leave undecided.
Candidates, in the order to try:

1. **Script and orientation** with Tesseract OSD (`--psm 0`): deterministic, CPU, Linux and Mac. Answers the
   question 3.1A cannot answer for a scan: which script, which rotation. Decides 3.2b eligibility (engine supports
   the script?).
2. **Text detection** (a DBNet/PP-OCR detector through ONNX Runtime, a few MB, CPU): text boxes, their count and
   area. Gives the expected amount of text on the page (also used by the gate, 5.2) and a better ink-residue map.
3. **OCR confidence probe**: run the 3.2b engine itself on the page and read its mean confidence; for 3.2b the read
   *is* the probe, so this hook mostly formalises "try 3.2b, gate, escalate".
4. **Layout detection** (docling's layout model, or a small DocLayout-YOLO in ONNX): table, picture, formula and
   text regions.
5. A page-type classifier (handwriting, form, photo): only if 1-4 plateau; it needs labelled data.

### 4.4 The router

`router.decide(features, capabilities)` returns runway, confidence and reasons. Rules are ordered and their
thresholds live in a versioned data table (not scattered constants), so the harness can sweep them. Capabilities
are what this machine can run (VLM usable, OCR engines and their scripts); they feed the doubt rule (section 2.3) and
the deferred fallback decision (section 9). Sketch:

1. Blank (no text, no ink) -> not read.
2. Plausible visible layer, little residue, no large pictures -> 3.2a.
3. Plausible layer + large pictures or residue confined to regions -> 3.2c.
4. Hidden OCR layer: plausible and agreeing with the image (residue low) -> 3.2a; otherwise treated as no layer.
5. No usable layer: clean (dpi, blur, contrast, skew within limits), printed text lines, no ruled table, no photo,
   script supported by an available OCR engine -> 3.2b.
6. Everything else -> 3.2d.

**Scans: try 3.2b first, verify, escalate.** The owner reports that most scanned pages are English. A scan has no
text layer, so its script is unknown until something reads it. Two ways to learn it:
- read the page with 3.2b and check the result's script, quality and size in the gate;
- or ask the OSD probe (3.1C) first.

When most scans are clean Latin print, the first way is the cheaper one. Clean scans go to 3.2b, and only those that
fail the gate are read again by 3.2d. A wrong guess costs one OCR pass, about a second, not a VLM read. Pages that 3.1B
already marks as hard (photos, ruled tables, poor quality) go straight to 3.2d. Rule 5's "supported script" condition
is then checked on the result rather than predicted, and the OSD probe only becomes worth adding if the harness shows
many wasted 3.2b passes (for example a collection that is mostly Devanagari).

## 5. 3.2 Read and 3.3 Gate

### 5.1 Runways

| Runway | Reader | OCR | Today's equivalent |
|---|---|---|---|
| 3.2a | docling (layout + TableFormer) | off, or only for embedded pictures | `digital` |
| 3.2b | docling, page as image | full page, engine chosen by script (e.g. Tesseract for Devanagari, Apple Vision for Latin) | `fallback` (today only after the VLM failed) |
| 3.2c | docling + VLM on picture / residue crops | off for the text, VLM for the regions | `embedded` (today picture objects only, not residue regions) |
| 3.2d | VLM, whole page | the model reads | `raster` / `image` |

The page cache key already includes the reader tag, so a page re-routed to the reader it had before is not read again.

### 5.2 Gate v2

Today's checks stay (coverage, script, docling grade, table shape, resolution, degenerate, running balance,
totals). The changes:

- **Expected size.** From 3.1B (ink area per character height) or the 3.1C text detector: how much text the page
  should have. Flag output far smaller (a dropped region) or far larger (invention, loops). Today scans are only
  checked for "almost nothing came out".
- **Plausibility.** Word-validity share per script and a character n-gram score, instead of "share of single-letter
  words". Catches OCR noise and wrong-script output.
- **Column types.** A numeric column with letters, dates and amounts in their formats, repeated headers.
- **Reader confidence.** OCR engines report per-word confidence (3.2b). For the VLM, token log-probabilities are a hook
  to try (whether `mlx-vlm` exposes them is not verified).
- **Agreement with the layer** (3.2a/c): the output must contain the layer's words, not only its character count.
- **Graded and located.** Each failed check carries a severity and, where it can, a region or a table cell. The
  verdict maps to an action:
  - pass;
  - escalate: a cheap runway failed, so read again with 3.2d, once;
  - repair: 3.2d failed with located problems, so 3.4, which already crops cells and re-reads pages;
  - low: nothing left to try, or repair failed.

Each runway has the checks that catch its own failures (section 2.2). A runway that has no such check for a page
type is not eligible for it: for example, a scanned table goes to 3.2d unless the arithmetic validators can verify
3.2b's table.

## 6. The routing harness

The harness makes routing measurable and is the place where 3.1C probes earn their way in. It reuses the
benchmark's gold sets (`<home>/conversion_gold/`), its scoring (`metrics.py`) and its engine registry.

- **Input:** a page set. Verified gold pages give real quality numbers (CER, numeric-cell exact, table similarity,
  balance pass). Unverified pages give proxies only (agreement between runways, plausibility, arithmetic).
- **Run:** for every page, compute the features (3.1A, 3.1B, any probes) and read it with *every* runway that applies
  (a, b with each eligible engine, c, d), then gate each result. Results are stored per page and runway: text, time,
  memory, gate verdict, quality. The page cache avoids paying twice.
- **Oracle:** for each page, the cheapest runway whose quality reaches the target.
- **Reports:**
  - router vs oracle confusion matrix;
  - the cost/quality frontier as thresholds are swept;
  - **gate false-pass rate**, meaning pages the gate passed whose quality is below target;
  - escalation and repair rates;
  - the cost of profiling.
- **Shadow mode:** in a real indexing run the new router can decide alongside the old one and record both decisions
  in the trace, without changing what is read. Real runs then show where the two differ.
- **Ground truth without personal data:** a generator of synthetic scans, made from known text (Latin and Devanagari,
  tables with arithmetic) with controlled damage: dpi, blur, skew, noise, JPEG, photo perspective. It gives exact truth at
  scale for tests and threshold sweeps. Real behaviour still needs a small verified set from the owner's documents,
  which never enters the repository.
- **Interfaces:** `rag-search bench route SET [--runways a,b,c,d] [--probes osd,textdet] [--shadow]`; a report in the
  Playground's benchmark section; `api.bench_route*`.

## 7. Phases

Each phase leaves the product working and measurable. No routing behaviour changes before R4.

| Phase | Content | Exit criterion |
|---|---|---|
| **R0a trace mining** (first, no reading; **built**: `scripts/mine_traces.py`) | a report from the stored traces of a run: pages by kind × gate check × outcome, read and gate time per kind, the low pages listed per check for sampling; the profile cost; why the gate takes 4 h 30 min | the owner samples about 20 low pages per main check (table shape, coverage) and says which are real |
| **R0 harness skeleton** | `bench route` running today's pipeline as runway engines (a = docling digital, b = docling OCR, d = VLM); per-page records; oracle and reports; synthetic damaged-scan generator in `tests/data` tooling | reports on the synthetic set and one owner document set |
| **R1 profiler 3.1A + 3.1B** (runs alongside R2) | the features of 4.1 and 4.2 in the profile record; probe interface (3.1C hooks) with a no-op probe; features in the harness | feature distributions per oracle runway; profile cost per page measured |
| **R2 runway b as first-class** (priority: most scans are English) | try-3.2b-first for clean scans (3.1B quality features), deskew before OCR, the result's script checked; brings forward the two gate checks 3.2b needs, expected size and plausibility, with escalation to 3.2d; behind a setting, off until the harness agrees | harness: b vs d quality and time on the English scans; share of 3.2b passes that escalate |
| **R3 gate v2** | the rest of the gate: column types, graded and located verdicts, escalation actions | false-pass rate measured and lower than today on the same pages |
| **R4 router v2 in shadow** | `router.decide` v2 with the threshold table; both decisions in the trace; dashboard shows disagreements | agreement report on real runs; thresholds chosen from the frontier |
| **R5 switch on** | v2 routing and the escalation ladder live; image files through the router; 3.2c on residue regions | quality not worse than v1 on the verified set; VLM pages and repair runs fewer |
| **R6 first probes** | Tesseract OSD; ONNX text detector; each behind its hook, on only where the harness shows a gain | measured gain per probe |
| **R7 decisions** | fallback policy (section 9.1) and non-Mac models (section 9.2) | recorded in this document and ARCHITECTURE.md |

Records: the page record gains `route` (`runway`, `confidence`, `reasons`, `router` version), compact `features`,
and `escalated_from`. The trace version and `convert_profile` change. The page cache key does not change, so
re-routing reuses earlier reads by the same reader.

## 8. Critical analysis

**What is strong**
- One principle for the router, sure means cheap and doubt means VLM, with "sure" measured rather than assumed.
- The ink-residue check (3.1B) closes a real hole cheaply: content that the text layer does not have.
- The harness turns tuning into measurement, and gives 3.1C a way to prove itself before it costs anything.
- Shadow mode lets real runs show the effect before anything changes.

**Risks and weak points**

1. **The gate decides whether this works.** A smarter router makes cheap runways more frequent, so every
   confidently wrong cheap read depends on the gate catching it. History is a warning: 49 of 56 runaways passed
   before 0.9.18. That is why R3 comes before R4/R5 and why the false-pass rate is the primary metric, ahead of
   speed.
2. **3.2b's value depends on the mix, and the mix favours it.**
   - The owner has many well-scanned English documents. Those without a text layer go to the VLM today (branch
     `raster`); those with a scanner's OCR layer are already read as digital (3.2a). The first group is 3.2b's main
     target: clean Latin print read by Apple Vision or Tesseract through docling in seconds, against tens of seconds
     to minutes per page for the VLM, and an OCR engine cannot loop or invent text.
   - It is weaker where the Devanagari passbooks and deeds are: Apple Vision on this macOS has no Marathi or Hindi,
     Tesseract reads them but writes no tables, and docling's OCR has been weak on scanned tables. Those pages stay
     on 3.2d.
   - So 3.2b is kept and moved forward (R2 straight after R0). The harness has to confirm it on the English scans:
     quality against 3.2d, and the gate's ability to catch the pages where it fails (multi-column, faint, tables).
3. **Ground truth is scarce.**
   - No verified gold set exists yet ("no baseline numbers exist until a gold set has been checked").
   - Agreement between two readers is a weak proxy: they can agree on the same mistake.
   - Synthetic damaged scans give exact truth but a different distribution.
   - The plan needs the owner to verify a small set, around 50 to 100 pages across the page types. That labour is
     on the critical path of R4.
4. **Escalation can cost more than it saves.** A page that fails 3.2b and is then read by 3.2d pays for both. If
   3.2b's failure rate on its eligible pages is high, routing straight to 3.2d is cheaper. The frontier in the harness
   decides the threshold. Thresholds may differ between document collections; the first version keeps one global
   table.
5. **Profiling cost.**
   - Rendering every page at 150 dpi on a 50 000-page library is roughly an hour of CPU. That is affordable next to
     conversion, but not free.
   - 3.1B therefore runs only on pages 3.1A does not decide. Probes run only on pages 3.1A+B do not decide, and
     within a budget.
   - On the Mac, probes compete with the docling pool and the VLM for CPU and memory.
6. **Complexity grows.** There are more paths, records, cache tags and labels. Keep the runway set at four, put the
   rules in one data table, and keep the vocabulary shared between engine, trace, CLI and UI, as the current plan
   already requires.
7. **Hidden OCR layers are often good** (ABBYY-made scans). Distrusting all of them would send good pages to the VLM.
   The plan trusts a layer when it is plausible and the image agrees with it (low residue), which needs both 3.1A
   and 3.1B.
8. **"Doubt means VLM" assumes a VLM.** On Linux, in the cloud, or with memory short, 3.2d is not available. That is
   the deferred decision in 9.1; the router's `capabilities` input is there so the decision is made at routing time,
   not as a hidden per-page fallback.
9. **Determinism.** Probes and thresholds become part of what a conversion depends on. Their versions go into the
   page record. A router change re-routes pages but re-reads only pages whose reader changed.
10. **Out of scope here:** pictures inside Office files are not read by the VLM today, and docling's own VLM
    pipeline (setting `pipeline = vlm`) bypasses all of this. Both are noted, not addressed.

## 9. Decisions deferred to the end (R7)

### 9.1 Fallback policy

Today a page the VLM cannot read falls back to docling full-page OCR, then to Apple Vision and Tesseract. To decide
after R5, with harness numbers in hand:
- what happens to a page in doubt when 3.2d is unavailable (route to 3.2b and flag it, or leave it unread and
  flagged);
- whether a 3.2d failure goes to repair only, or also to a plain-OCR last resort;
- which of the current last resorts stay.

### 9.2 Models without a hard Mac dependency

The reader and repair models run through `mlx-vlm`, which is Apple Silicon only, and Apple Vision is macOS only. To
choose:
- VLM backends for Linux and other GPUs (for example transformers or vLLM with the same Qwen3-VL family, or other
  open document VLMs), behind the existing `module:attr` backend interface;
- OCR engines that run everywhere (Tesseract, RapidOCR/ONNX);
- probes that are ONNX/CPU by design;
- tier B tests for each backend that can run in the cloud.

## 10. Documents to update when this lands

`ARCHITECTURE.md` (3, 5.1.1, 5.1.3), `stages.py` descriptions and the dashboard labels (3.2a-d), `README.md`
(Indexing tab, settings), `docs/design/document-conversion-plan.md` (status pointer to this plan), and this
document's status per phase.
