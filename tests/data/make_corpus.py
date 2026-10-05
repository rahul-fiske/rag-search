#!/usr/bin/env python3
"""Build the test corpus: one small, synthetic file for every kind of input rag-search handles.

    python tests/data/make_corpus.py            # writes tests/data/corpus/ (committed) and checks corpus.json

The files are committed, so the tests never need this script or its libraries.  Rebuild only to change the
corpus, then commit the files together with ``corpus.json`` (which says what each file must produce).
Generator-only dependencies: reportlab, Pillow, pypdfium2, python-docx, openpyxl, python-pptx (the last three
come with docling), optional pillow-heif (the .heic file) and LibreOffice ``soffice`` (the legacy .doc file).

Every text is invented: no names, account numbers or addresses of real people.  Each file carries a *needle*,
a phrase that occurs in no other file, so a test can search for it and expect exactly that file.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_FALLBACKS = (FONT, "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf")
DPI = 200                                      # "scanned" pages: sharp enough for OCR, small enough to commit

LOREM = ("The committee reviewed the quarterly plan and agreed to move the archive to the new building. "
         "Each department will label its boxes before the end of the month, and the facilities team will "
         "schedule the move over two weekends so that no office is closed during working hours.")

# a bank statement whose arithmetic holds: balance = previous balance - debit + credit
STATEMENT = [("01-03-2024", "Opening balance", "", "", "10,000.00"),
             ("02-03-2024", "Salary credit", "", "2,819.00", "12,819.00"),
             ("03-03-2024", "Rent payment", "5,000.00", "", "7,819.00"),
             ("04-03-2024", "Grocery store", "450.00", "", "7,369.00"),
             ("05-03-2024", "Interest", "", "31.00", "7,400.00"),
             ("06-03-2024", "Electricity bill", "1,200.00", "", "6,200.00")]
STATEMENT_HEADER = ("Date", "Description", "Debit", "Credit", "Balance")


def _font(size: int):
    from PIL import ImageFont

    for f in FONT_FALLBACKS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _rl_canvas(path: Path, **kw):
    from reportlab import rl_config
    from reportlab.pdfgen import canvas

    rl_config.invariant = 1                    # no creation date or random id: the same bytes every time
    return canvas.Canvas(str(path), **kw)


def _paragraphs(c, lines: list[str], x=60, y=780, size=11, leading=15, width=95) -> float:
    import textwrap

    c.setFont("Helvetica", size)
    for line in lines:
        if line.startswith("# "):
            c.setFont("Helvetica-Bold", size + 5)
            c.drawString(x, y, line[2:])
            c.setFont("Helvetica", size)
            y -= leading * 1.8
            continue
        for part in textwrap.wrap(line, width) or [""]:
            c.drawString(x, y, part)
            y -= leading
        y -= leading * 0.5
    return y


def scan_image(lines: list[str], size=(1165, 1654), dpi=DPI, font_px=34, table=None):
    """A clean "scan" (white page, black print) of *lines*, optionally followed by a ruled *table*."""
    from PIL import Image, ImageDraw

    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    f = _font(font_px)
    y = int(size[1] * 0.06)
    for line in lines:
        d.text((int(size[0] * 0.08), y), line, fill=0, font=f)
        y += int(font_px * 1.6)
    if table:
        y += font_px
        cols = [0.06, 0.24, 0.50, 0.65, 0.80, 0.96]
        xs = [int(c * size[0]) for c in cols]
        tf = _font(int(font_px * 0.62))
        row_h = int(font_px * 1.25)
        for r, row in enumerate(table):
            top = y + r * row_h
            d.line((xs[0], top, xs[-1], top), fill=0, width=2)
            for k, cell in enumerate(row):
                d.text((xs[k] + 6, top + 6), cell, fill=0, font=tf)
        bottom = y + len(table) * row_h
        d.line((xs[0], bottom, xs[-1], bottom), fill=0, width=2)
        for x in xs:
            d.line((x, y, x, bottom), fill=0, width=2)
    im.info["dpi"] = (dpi, dpi)
    return im


def save_png(im, path: Path, dpi=DPI) -> None:
    im.save(path, dpi=(dpi, dpi), optimize=True)


def image_pdf(images, path: Path, dpi=DPI) -> None:
    """A PDF whose pages are pictures only (a scanner's output): no text layer."""
    first, *rest = [i.convert("L") for i in images]
    first.save(path, "PDF", resolution=float(dpi), save_all=True, append_images=rest)


# ── PDFs ─────────────────────────────────────────────────────────────────────────────────────

def pdf_text(path: Path) -> None:
    c = _rl_canvas(path)
    _paragraphs(c, ["# Archive relocation plan", LOREM,
                    "Needle: the quartz harbor ledger is kept in the north wing.", LOREM])
    c.showPage()
    _paragraphs(c, ["# Schedule", "Week one moves the reading room; week two moves the stacks. " * 3, LOREM])
    c.showPage()
    _paragraphs(c, ["# Contacts", "Questions go to the facilities desk, extension four hundred. " * 3])
    c.showPage()
    c.save()


def pdf_table_reportlab(path: Path) -> None:
    """The same statement drawn by reportlab: docling leaves `&#124;` in some of its cells (a known issue)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab import rl_config

    rl_config.invariant = 1
    st = getSampleStyleSheet()
    t = Table([STATEMENT_HEADER, *STATEMENT])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                           ("ALIGN", (2, 1), (-1, -1), "RIGHT")]))
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build([Paragraph("Statement of account, March 2024", st["Title"]),
               Paragraph("Needle: rust prairie statement.", st["Normal"]), t,
               Paragraph("Closing balance 6,200.00. " + LOREM, st["Normal"])])


def _office_pdf(docx_path: Path, path: Path) -> None:
    """*docx_path* printed to PDF by LibreOffice: the kind of PDF a word processor writes."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SystemExit("LibreOffice (soffice) is needed for the word-processor PDFs")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", td, str(docx_path)],
                       check=True, capture_output=True, timeout=180)
        shutil.move(str(Path(td) / (docx_path.stem + ".pdf")), path)


def _statement_docx(path: Path, needle: str, rows) -> None:
    import docx

    d = docx.Document()
    _fixed_core(d.core_properties)
    d.add_paragraph(needle)
    t = d.add_table(rows=0, cols=5)
    t.style = "Table Grid"
    for row in (STATEMENT_HEADER, *rows):
        for cell, text in zip(t.add_row().cells, row):
            cell.text = text
    d.save(path)


def pdf_statement(path: Path) -> None:
    """A born-digital statement from a word processor; its running balance holds."""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / path.with_suffix(".docx").name
        _statement_docx(src, "Statement of account, March 2024. Needle: cobalt meadow savings statement.",
                        STATEMENT)
        _office_pdf(src, path)


def pdf_table_across_pages(path: Path) -> None:
    """A long statement table that runs from page 1 onto page 2 (the header is not repeated)."""
    rows, bal = [], 10000.0
    for i in range(1, 61):
        debit, credit = (125.0 * (i % 4), 0.0) if i % 3 else (0.0, 310.0 + i)
        bal = bal - debit + credit
        rows.append((f"{i:02d}-04-2024", f"Entry {i}", f"{debit:,.2f}" if debit else "",
                     f"{credit:,.2f}" if credit else "", f"{bal:,.2f}"))
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / path.with_suffix(".docx").name
        _statement_docx(src, "Needle: amber canyon passbook, April.", rows)
        _office_pdf(src, path)


def pdf_scan(path: Path) -> None:
    image_pdf([scan_image(["Scanned memo", "Needle: violet glacier memo.",
                           "The boiler inspection is due in May.", "Please keep the corridor clear."]),
               scan_image(["Page two of the scanned memo", "The inspector will arrive at nine.",
                           "Thank you for your patience."])], path)


def pdf_scan_statement(path: Path) -> None:
    image_pdf([scan_image(["Statement of account (scanned)", "Needle: saffron delta statement."],
                          table=[STATEMENT_HEADER, *STATEMENT])], path)


def pdf_lowres_scan(path: Path) -> None:
    im = scan_image(["Faxed notice", "Needle: pewter lagoon bulletin.", "Low resolution copy."],
                    size=(560, 790), dpi=96, font_px=18)
    image_pdf([im], path, dpi=96)


def pdf_mixed(path: Path) -> None:
    import pypdfium2 as pdfium

    tmp_t, tmp_s = path.with_name("_t.pdf"), path.with_name("_s.pdf")
    c = _rl_canvas(tmp_t)
    _paragraphs(c, ["# Mixed document", "Needle: indigo orchard report.", LOREM])
    c.showPage()
    _paragraphs(c, ["# Second text page", LOREM])
    c.showPage()
    c.save()
    image_pdf([scan_image(["Signed approval page", "Approved by the board in March."])], tmp_s)
    a, b = pdfium.PdfDocument(str(tmp_t)), pdfium.PdfDocument(str(tmp_s))
    a.import_pages(b)
    a.save(str(path))
    a.close(), b.close()
    tmp_t.unlink(), tmp_s.unlink()


def pdf_blank_page(path: Path) -> None:
    import pypdfium2 as pdfium

    tmp_t, tmp_b = path.with_name("_t.pdf"), path.with_name("_b.pdf")
    c = _rl_canvas(tmp_t)
    _paragraphs(c, ["# Cover letter", "Needle: maroon atlas letter.", LOREM])
    c.showPage()
    c.save()
    from PIL import Image
    image_pdf([Image.new("L", (1165, 1654), 255)], tmp_b)                     # a scanned blank sheet
    a, b = pdfium.PdfDocument(str(tmp_t)), pdfium.PdfDocument(str(tmp_b))
    a.import_pages(b)
    a.save(str(path))
    a.close(), b.close()
    tmp_t.unlink(), tmp_b.unlink()


def pdf_text_scan_blank(path: Path) -> None:
    """Two text pages, a scanned page with print on it, and a blank scanned sheet; the scans are 800 x 1000
    pixel pictures placed at their pixel size (a low-resolution scan: the gate flags it)."""
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw

    parts = [path.with_name("_t.pdf"), path.with_name("_s.pdf"), path.with_name("_b.pdf")]
    c = _rl_canvas(parts[0], pagesize=(612, 792))
    for word in ("alpha", "beta"):
        c.setFont("Helvetica", 12)
        c.drawString(72, 700, f"Page text that is comfortably longer than the forty character limit for a usable layer. {word}")
        c.showPage()
    c.save()
    scan = Image.new("RGB", (800, 1000), (255, 255, 255))
    d = ImageDraw.Draw(scan)
    for y in range(100, 700, 40):
        d.rectangle((100, y, 700, y + 18), fill=(0, 0, 0))
    scan.save(parts[1])
    Image.new("RGB", (800, 1000), (255, 255, 255)).save(parts[2])
    doc = pdfium.PdfDocument(str(parts[0]))
    for extra in parts[1:]:
        other = pdfium.PdfDocument(str(extra))
        doc.import_pages(other)
        other.close()
    doc.save(str(path))
    doc.close()
    for f in parts:
        f.unlink()


def pdf_picture_on_text(path: Path) -> None:
    """A text page with a large picture (40 % of the page) that holds text of its own."""
    from reportlab.lib.utils import ImageReader

    pic = scan_image(["WAREHOUSE RECEIPT", "Needle: jade compass receipt.", "Total paid 4,512.00"],
                     size=(900, 500), font_px=40)
    buf = io.BytesIO()
    pic.save(buf, "PNG")
    buf.seek(0)
    c = _rl_canvas(path)
    y = _paragraphs(c, ["# Expense report", "The receipt below was attached by the traveller. " + LOREM])
    c.drawImage(ImageReader(buf), 60, y - 300, width=475, height=264)
    c.showPage()
    c.save()


def pdf_hidden_ocr(path: Path) -> None:
    """A scanner's searchable PDF: a full-page picture with an invisible text layer on top."""
    from reportlab.lib.utils import ImageReader

    lines = ["Searchable scan", "Needle: copper willow invoice.", "Invoice total 980.00 due in April."]
    im = scan_image(lines, size=(1240, 1754))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    c = _rl_canvas(path)
    c.drawImage(ImageReader(buf), 0, 0, width=595, height=842)
    t = c.beginText(48, 780)
    t.setTextRenderMode(3)                                                  # invisible
    t.setFont("Helvetica", 16)
    for line in lines + [LOREM[:90]]:
        t.textLine(line)
    c.drawText(t)
    c.showPage()
    c.save()


def pdf_garbled(path: Path) -> None:
    """A text layer whose spaces were lost (a broken font map): words run together."""
    c = _rl_canvas(path)
    c.setFont("Helvetica", 9)
    run = LOREM.replace(" ", "")
    y = 780
    for i in range(0, len(run), 85):
        c.drawString(40, y, run[i:i + 85])
        y -= 13
    c.showPage()
    c.save()


def pdf_rotated(path: Path) -> None:
    c = _rl_canvas(path)
    c.setPageRotation(90)
    _paragraphs(c, ["# Rotated page", "Needle: scarlet tundra diagram.", LOREM])
    c.showPage()
    c.save()


def pdf_many_pages(path: Path) -> None:
    """23 text pages: more than one run of pages (a run is at most 20)."""
    c = _rl_canvas(path)
    for n in range(1, 24):
        _paragraphs(c, [f"# Section {n}", f"Section {n} of the handbook. " + LOREM[:120]
                        + (" Needle: silver fjord handbook." if n == 22 else "")])
        c.showPage()
    c.save()


def pdf_encrypted(path: Path) -> None:
    from reportlab.lib import pdfencrypt

    enc = pdfencrypt.StandardEncryption("open-sesame", ownerPassword="owner-secret", canPrint=0)
    c = _rl_canvas(path, encrypt=enc)
    _paragraphs(c, ["# Confidential", "Needle: onyx prairie contract.", LOREM])
    c.showPage()
    c.save()


def pdf_broken(path: Path, good: Path) -> None:
    data = good.read_bytes()
    path.write_bytes(data[: len(data) // 3])                                # cut off: no xref, no trailer


# ── Office, HTML, CSV, AsciiDoc ──────────────────────────────────────────────────────────────

def _fixed_core(props) -> None:
    import datetime as dt

    when = dt.datetime(2024, 1, 1)
    props.author, props.last_modified_by, props.created, props.modified = "rag-search tests", "", when, when


def docx_report(path: Path) -> None:
    import docx
    from docx.shared import Inches

    d = docx.Document()
    _fixed_core(d.core_properties)
    d.add_heading("Library renovation report", 0)
    d.add_paragraph("Needle: emerald beacon renovation. " + LOREM)
    d.add_heading("Costs", 1)
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for cell, text in zip(t.rows[0].cells, ("Item", "Quantity", "Cost")):
        cell.text = text
    for row in (("Shelves", "40", "12,000.00"), ("Lamps", "25", "3,750.00"), ("Chairs", "60", "9,000.00")):
        for cell, text in zip(t.add_row().cells, row):
            cell.text = text
    d.add_heading("Floor plan", 1)
    buf = io.BytesIO()
    scan_image(["FLOOR PLAN", "Reading room east"], size=(600, 300), font_px=30).save(buf, "PNG")
    buf.seek(0)
    d.add_picture(buf, width=Inches(3))
    d.add_paragraph("The work is planned for the summer break.")
    d.save(path)


def docx_lock(path: Path) -> None:
    path.write_bytes(b"\x00" * 162)                                        # what Word leaves while a file is open


def xlsx_budget(path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    _fixed_core(wb.properties)
    ws = wb.active
    ws.title = "Budget"
    ws.append(["Category", "Planned", "Actual"])
    for row in (("Travel", 5000, 4200), ("Training", 3000, 3100), ("Equipment", 8000, 7600)):
        ws.append(row)
    ws.append(["Needle: tangerine summit budget", None, None])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Note"])
    ws2.append(["Approved at the spring meeting."])
    wb.save(path)


def pptx_slides(path: Path) -> None:
    from pptx import Presentation

    p = Presentation()
    _fixed_core(p.core_properties)
    for title, body in (("Onboarding", "Needle: crimson lantern onboarding."),
                        ("First week", "Meet the team, set up the laptop, read the handbook."),
                        ("Questions", "Ask your buddy or the help desk.")):
        s = p.slides.add_slide(p.slide_layouts[1])
        s.shapes.title.text = title
        s.placeholders[1].text = body
    p.save(path)


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Opening hours</title></head><body>
<h1>Opening hours</h1>
<p>Needle: {needle}.</p>
<p>The museum is open every day except Monday.</p>
<table><tr><th>Day</th><th>Hours</th></tr><tr><td>Tuesday</td><td>10-18</td></tr>
<tr><td>Sunday</td><td>10-16</td></tr></table>
<ul><li>Guided tours at eleven</li><li>Free entry on the first Sunday</li></ul>
</body></html>
"""

ADOC = """= Deployment guide

Needle: azure quarry deployment.

== Steps

. Build the package.
. Copy it to the server.
. Restart the service.

[cols="1,2"]
|===
|Step |Command

|Build
|make package

|Restart
|systemctl restart app
|===
"""

MD = """# Team notes

Needle: olive horizon notes.

## Decisions

* Move the stand-up to ten o'clock.
* Review the backlog every Friday.

| Owner | Task |
|---|---|
| Ops | Rotate the keys |
| Dev | Fix the login page |

```bash
make test
```
"""

HINDI = """# पुस्तकालय सूचना

सुई: केसरिया नदी सूचना। पुस्तकालय सोमवार को बंद रहेगा और मंगलवार से सामान्य समय पर खुलेगा।

The library notice is also available in English at the front desk.
"""


# ── images ───────────────────────────────────────────────────────────────────────────────────

def images(dirp: Path) -> None:
    from PIL import Image, ImageDraw

    save_png(scan_image(["Scanned form", "Needle: lilac pinnacle form.", "Signature on page one."]),
             dirp / "scan.png")
    # a photo taken sideways: stored rotated, EXIF says "rotate 90 clockwise to display"
    up = scan_image(["Receipt photo", "Needle: mustard valley receipt.", "Paid 12.50"], size=(1165, 820))
    exif = Image.Exif()
    exif[0x0112] = 6
    up.rotate(90, expand=True).convert("RGB").save(dirp / "receipt-sideways.jpg", quality=80, exif=exif)
    lo = scan_image(["Small picture", "Needle: khaki ridge note."], size=(500, 700), dpi=96, font_px=22)
    save_png(lo, dirp / "lowres.png", dpi=96)
    frames = [scan_image([f"Fax page {n}", "Needle: teal meadow fax." if n == 3 else "Sent from the office."])
              for n in (1, 2, 3)]
    frames[0].save(dirp / "fax.tif", save_all=True, append_images=frames[1:], compression="tiff_lzw",
                   dpi=(DPI, DPI))
    scan_image(["Bitmap scan", "Needle: ochre spire bitmap."], size=(800, 600), font_px=30).convert("1").save(
        dirp / "bitmap.bmp")                                                     # 1 bit per pixel: 60 KB, not 480
    scan_image(["Web picture", "Needle: plum basin picture."], size=(800, 600), font_px=30).save(
        dirp / "web-picture.webp", quality=80)
    photo = Image.new("RGB", (900, 600), (90, 140, 200))                       # sky and grass, no text
    d = ImageDraw.Draw(photo)
    d.rectangle((0, 380, 900, 600), fill=(60, 150, 70))
    d.ellipse((650, 60, 780, 190), fill=(250, 220, 90))
    photo.save(dirp / "photo-no-text.jpg", quality=70)
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        scan_image(["Phone photo", "Needle: ruby harbor photo."], size=(800, 600), font_px=30).convert("RGB").save(
            dirp / "phone.heic", quality=60)
    except ImportError:
        print("pillow-heif not installed: phone.heic not written", file=sys.stderr)


# ── files indexing does not read ─────────────────────────────────────────────────────────────

def legacy_doc(path: Path, source_docx: Path) -> bool:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("LibreOffice not found: legacy .doc not written", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([soffice, "--headless", "--convert-to", "doc", "--outdir", td, str(source_docx)],
                       check=True, capture_output=True, timeout=180)
        shutil.move(str(Path(td) / (source_docx.stem + ".doc")), path)
    return True


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    d = {name: OUT / name for name in ("pdf", "office", "text", "images", "unsupported", "collision")}
    for p in d.values():
        p.mkdir(parents=True)

    pdf_text(d["pdf"] / "text.pdf")
    pdf_statement(d["pdf"] / "statement.pdf")
    pdf_table_reportlab(d["pdf"] / "statement-pipe-artifacts.pdf")
    pdf_table_across_pages(d["pdf"] / "table-across-pages.pdf")
    pdf_scan(d["pdf"] / "scan.pdf")
    pdf_scan_statement(d["pdf"] / "scan-statement.pdf")
    pdf_lowres_scan(d["pdf"] / "lowres-scan.pdf")
    pdf_mixed(d["pdf"] / "mixed.pdf")
    pdf_blank_page(d["pdf"] / "blank-page.pdf")
    pdf_text_scan_blank(d["pdf"] / "text-scan-blank.pdf")
    pdf_picture_on_text(d["pdf"] / "picture-on-text.pdf")
    pdf_hidden_ocr(d["pdf"] / "hidden-ocr-layer.pdf")
    pdf_garbled(d["pdf"] / "garbled-text-layer.pdf")
    pdf_rotated(d["pdf"] / "rotated.pdf")
    pdf_many_pages(d["pdf"] / "many-pages.pdf")
    pdf_encrypted(d["pdf"] / "password.pdf")
    pdf_broken(d["pdf"] / "damaged.pdf", d["pdf"] / "text.pdf")

    docx_report(d["office"] / "report.docx")
    xlsx_budget(d["office"] / "budget.xlsx")
    pptx_slides(d["office"] / "slides.pptx")
    (d["office"] / "hours.html").write_text(HTML.format(needle="navy orchard hours"), encoding="utf-8")
    (d["office"] / "legacy-page.htm").write_text(HTML.format(needle="bronze cove page"), encoding="utf-8")
    (d["office"] / "inventory.csv").write_text(
        "item,count,location\nprojector,4,room 12\nwhiteboard,9,room 3\nneedle: sepia canyon inventory,1,store\n",
        encoding="utf-8")
    (d["office"] / "deploy.adoc").write_text(ADOC, encoding="utf-8")

    (d["text"] / "notes.md").write_text(MD, encoding="utf-8")
    (d["text"] / "readme.txt").write_text("README\n\nNeedle: granite meadow readme.\n\n" + LOREM + "\n",
                                          encoding="utf-8")
    (d["text"] / "hindi.md").write_text(HINDI, encoding="utf-8")
    (d["text"] / "empty.txt").write_text("   \n\n", encoding="utf-8")
    nested = d["text"] / "archive" / "2019"
    nested.mkdir(parents=True)
    (nested / "old-notes.md").write_text("# Old notes\n\nNeedle: cedar valley archive.\n\n" + LOREM + "\n",
                                         encoding="utf-8")
    (d["text"] / ".hidden-draft.md").write_text("# Draft\n\nNeedle: hidden draft, never indexed.\n",
                                                encoding="utf-8")

    images(d["images"])

    if not legacy_doc(d["unsupported"] / "legacy.doc", d["office"] / "report.docx"):
        pass
    (d["unsupported"] / "notes.rtf").write_text(r"{\rtf1\ansi Needle: rtf is not read.\par}", encoding="ascii")
    with zipfile.ZipFile(d["unsupported"] / "bundle.zip", "w") as z:
        z.writestr(zipfile.ZipInfo("inside.txt", (2024, 1, 1, 0, 0, 0)), "zip contents are not read")
    (d["unsupported"] / "no-extension").write_text("a file without an extension\n", encoding="utf-8")
    docx_lock(d["unsupported"] / "~$report.docx")

    # two files that map to one document name: only one of them is indexed
    shutil.copy(d["pdf"] / "text.pdf", d["collision"] / "manual.pdf")
    docx_report(d["collision"] / "manual.docx")


def check() -> int:
    """Every file on disk is in corpus.json and every entry exists."""
    manifest = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
    listed = {e["path"] for e in manifest["files"]}
    on_disk = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()}
    missing, extra = sorted(listed - on_disk), sorted(on_disk - listed)
    for p in missing:
        print(f"listed in corpus.json but not built: {p}", file=sys.stderr)
    for p in extra:
        print(f"built but not listed in corpus.json: {p}", file=sys.stderr)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(on_disk)} files, {total / 1024:.0f} KiB")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    build()
    sys.exit(check())
