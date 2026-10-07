# Review of the profiler and the gate (leads to 0.9.29)

Asked for as: review the profiler and gate logic thoroughly, for correctness and for anything they miss. Read line by
line: `profiler.py`, `router.py`, `gate.py`, `degenerate.py`, `scanfacts.py`, `residue.py`, the text checks of
`docling_convert.py` (`_page_text_ok`, `dominant_script`, `has_real_text`), and the parts of `routed.py` that plan
pages and call the gate. `ARCHITECTURE.md` 5.1.1 describes the result.

## Method

Reading gave suspects; each was then tried before anything was changed: on pages generated for the purpose (PDFs
built with pypdfium2: rotated, with a shifted page box, with invisible text; images in every mode Pillow writes),
and on the 2,067 traces and source files of the full run of 0.9.27 (20,197 pages), without copying any content.
A suspect that could not be shown wrong was left alone.

## Defects fixed

| # | Where | What was wrong | On the full run | Fix |
|---|---|---|---|---|
| 1 | profiler, residue | Picture boxes and text rectangles in the page's own coordinates, which ignore `/Rotate` and the origin of the page box | 467 rotated pages, 49 of them with large pictures (2 in lane c, where the box is used) | `profiler.page_box` (PDFium's page-to-device mapping); clipped to the page |
| 2 | profiler | `hidden_ocr_layer` = any text over a full-page picture | 556 pages so marked: 506 have visible text (HiQPdf, PDFsharp, Chromium, Firefox, FrameMaker ...), 50 invisible text (a scanner) | invisible text only (render mode 3 / 7); the 506 get the comparison with their text layer back |
| 3 | gate | A scanner's OCR layer was read and trusted with no check of its own | the 50 pages | `plausibility` and `column_types` for such a page |
| 4 | `_page_text_ok` | Sound text layers called garbled, so the page was drawn and read by the document reader: long tokens that are values; bullets and icons in private-use or control codes; zero-width spaces | 94 pages called garbled, 81 of them sound (50 long tokens, about 25 symbols, 6 zero-width spaces); 28 were then flagged `script` because the reading held the same long tokens | only long runs of letters are run-together words; a lone symbol is not damage; format characters are not counted |
| 5 | profiler, vlm, applevision | A picture with a transparent ground was converted as it is: black ground, ink 1.0, dark writing invisible to every reader | none in this corpus (generated cases) | `profiler.opaque` |
| 6 | profiler | Every frame of any image file was a page | 2 phone JPEGs with a second frame (both blank, so nothing was indexed twice) | only TIFF frames are pages |
| 7 | scanfacts | Skew searched to 6 degrees, pages straightened up to 8: beyond 6 the measure was wrong and looked small | not measurable from the traces | search to 10 |
| 8 | gate, routed | The script of a garbled layer compared with the reading; a Tesseract last resort accepted whatever was not a runaway | 1 page doubted for its script (CJK); Tesseract replaced nothing in this run | the garbled layer's script is not used; the last resort must be words |
| 9 | routed | "No text" of a cached page said no reader had read it | 1 image, tried again on every update run | the cached reader is named, and the result is remembered |

## Checked and found right

* **128 text pages with a layer of 40 to 99 characters came out empty and nothing failed.** Looked at one by one
  through their cleaned layers: 129 of the 179 empty text pages hold nothing but a running header, footer or page
  number, the rest a footer with a roman page number. Nothing is lost; "no text" is the right outcome.
* **The 550 pages then marked as hidden OCR**, run through the OCR checks: 3 fail `plausibility`, 6 `column_types`;
  the median share of implausible words is 4.5 %. The size check (text against ink) would flag 63 of them (11 %);
  without verified pages that is as likely to be shading and photographs as lost text, so it was not added.
* **Images inside form objects** (523 of 8,099 sampled pictures): their bounds look like page coordinates (none is the
  unit square a form's own space would give; 36 lie off the page, as 179 top-level ones do).
* **16-bit, CMYK and bilevel images**: converted correctly for the ink figure and the readers.
* **The validators' names**: only `running_balance` and `totals` exist, so the gate's fallback label cannot mislabel.

## Left open

* **A page with a text layer and little of it** (a hidden OCR layer that holds only part of the page) is caught only
  by the size check above, which needs ground truth before it can be trusted on these pages.
* **A page genuinely written in a script other than Latin or Devanagari** and read by the document reader is still
  marked `degenerate` ("a script that is not on the page"): the rule was made for a reader that drifts from Devanagari
  into Bengali or Gujarati and cannot tell the two cases apart. The page is kept and flagged, no longer replaced.
  A setting for the scripts to expect would settle it.
* **Text lines and ruled lines are counted on the page as it lies**, so a page skewed by more than about 2 degrees has
  "no text lines" and goes to the document reader whatever else is true: the straighten-and-Tesseract path of lane b
  is reached only by pages with short lines. Counting them on the straightened page would open that lane to skewed
  scans; that shifts pages to the weaker reader and should be measured first.
* **`plausibility` counts acronyms without vowels and tokens that mix letters and digits** (`RTGS`, `80C`, part
  numbers) as implausible. It only sends a lane-b page to the document reader, the safe direction.
* **docling opens an image file itself** when no document reader is available: a transparent picture is not put on
  paper on that path.
* **The measured skew of a page beyond 10 degrees** is arbitrary within the range (12 measured 9); such a page is rare
  and the OCR checks catch a poor reading.
