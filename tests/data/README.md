# The test corpus

One small, synthetic file for every kind of input rag-search handles: 45 files, about 800 KB, no real
person's data. Tests copy what they need into a temporary data folder; the corpus itself is never written to.

```
corpus/pdf/          text, statement tables, scans, mixed, blank page, picture on a text page, hidden OCR layer,
                     garbled text layer, rotated, 23 pages, password-protected, damaged
corpus/office/       .docx .xlsx .pptx .html .htm .csv .adoc
corpus/text/         .md .txt (Devanagari, nested folder, whitespace only, a hidden file)
corpus/images/       .png .jpg (EXIF-rotated, no text) .tif (3 frames) .bmp .webp .heic
corpus/unsupported/  .doc .rtf .zip, no extension, an Office lock file: reported or never listed
corpus/collision/    manual.pdf + manual.docx: two files, one document name
corpus.json          what each file is for and what it must produce (the single description of the corpus)
make_corpus.py       rebuilds corpus/ (generator-only dependencies are listed in its docstring)
```

`corpus.json` has, per file: `shows` (why it exists), and either `skip` (hidden / lock / unsupported) or

* `route`: the page count and the branch of each page that the profiler and router must choose. Checked by
  `tests/portable/test_corpus.py` on every machine, with no model.
* `real`: how the file must end when it goes through the real pipeline (status, the strip of page branches,
  outcomes, failed gate checks, joined tables) and a `needle`, a phrase that is in that file only. Checked by
  `tests/real/test_corpus_real.py`, which indexes the whole corpus once with docling and small real models.
* `known_issue` (with `today_status` when the file ends differently from `real.status`): behaviour that is
  wrong today. The test expects the file *not* to end as described and fails when it does, so a fix is noticed
  and the entry is then deleted.

## Adding a kind of input

1. Add a function to `make_corpus.py` that writes the file (with a needle in it), call it in `build()`, run the script.
2. Add the file to `corpus.json`: `route` first; run `tests/portable/test_corpus.py`; then `real`, run
   `tests/real/test_corpus_real.py`, and check the result by looking at it (`index/` and `markup/` of the run)
   before writing it down as expected.
3. Commit the generated file together with `corpus.json`. The files are committed so that the tests need none
   of the generator's libraries.

The needles are distinct from each other on purpose (a search for a needle must rank its own file first); a test
checks that they are.

## What the corpus does not cover

Anything that needs a vision model or Apple Vision (`tests/machine/` on the Mac): the document reader on scans
and photographs, Apple Vision as the last reader, and cell repair with a second reader. Without them scans and
images go through docling's OCR (tesseract on Linux), which is why `scan.pdf` is a `fallback` page here and a
`raster` page on the Mac.
