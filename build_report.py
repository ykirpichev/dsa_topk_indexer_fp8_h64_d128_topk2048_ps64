#!/usr/bin/env python3
"""Build report.pdf from report.md using Chrome headless.

Steps:
  1. Crop excess whitespace from diagrams (adds 20px uniform padding)
  2. Convert report.md to a styled HTML file
  3. Call Chrome --headless --print-to-pdf
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parent
DIAG_DIR = REPO / "images" / "diagrams"
OUT_PDF = REPO / "report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PYTHON = "/opt/miniconda3/envs/forge/bin/python"

# ── 1. Crop whitespace from PNGs ────────────────────────────────────────────

CROP_SCRIPT = """
import sys, os
from pathlib import Path
import numpy as np
from PIL import Image

DIAG_DIR = Path(sys.argv[1])
PAD = 20          # uniform padding around content

for name in sorted(os.listdir(DIAG_DIR)):
    if not name.endswith('.png'):
        continue
    path = DIAG_DIR / name
    img = Image.open(path).convert('RGBA')
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    non_white = ~((r >= 245) & (g >= 245) & (b >= 245))
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if not rows.any():
        print(f"  {name}: all white, skipping")
        continue
    top    = max(0,               np.argmax(rows)           - PAD)
    bottom = min(img.size[1] - 1, len(rows) - np.argmax(rows[::-1]) - 1 + PAD)
    left   = max(0,               np.argmax(cols)           - PAD)
    right  = min(img.size[0] - 1, len(cols) - np.argmax(cols[::-1]) - 1 + PAD)
    cropped = img.crop((left, top, right + 1, bottom + 1))
    # Convert to RGB (white background) before saving
    bg = Image.new('RGB', cropped.size, (255, 255, 255))
    bg.paste(cropped, mask=cropped.split()[3])
    bg.save(path)
    print(f"  {name}: {img.size} -> {bg.size}")
"""


def crop_diagrams():
    print("Cropping whitespace from diagrams...")
    result = subprocess.run(
        [PYTHON, "-c", CROP_SCRIPT, str(DIAG_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print("CROP ERROR:", result.stderr, file=sys.stderr)
        sys.exit(1)


# ── 2. Convert Markdown → HTML ───────────────────────────────────────────────

def md_to_html(md_text: str) -> str:
    """Minimal Markdown → HTML converter covering the features used in report.md."""
    lines = md_text.split("\n")
    html_lines: list[str] = []
    i = 0

    def flush_p(buf):
        if buf:
            html_lines.append(f"<p>{''.join(buf)}</p>")
            buf.clear()

    para_buf: list[str] = []
    in_table = False
    in_ul = False

    def inline(s: str) -> str:
        # Bold
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        # Italic (single *)
        s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
        # Inline code
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        # Links [text](url)
        s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
        # Keep HTML entities / raw HTML as-is
        return s

    while i < len(lines):
        line = lines[i]

        # Blank line
        if line.strip() == "":
            flush_p(para_buf)
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
            i += 1
            continue

        # HR
        if re.match(r'^---+\s*$', line):
            flush_p(para_buf)
            html_lines.append("<hr>")
            i += 1
            continue

        # Headings
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            flush_p(para_buf)
            level = len(m.group(1))
            html_lines.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue

        # Figure: ![caption](path)
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)', line.strip())
        if m:
            flush_p(para_buf)
            caption = inline(m.group(1))
            path = m.group(2)
            html_lines.append(
                f'<figure>'
                f'<img src="{path}" alt="{caption}">'
                f'<figcaption>{caption}</figcaption>'
                f'</figure>'
            )
            i += 1
            continue

        # Table: starts with |
        if line.startswith('|'):
            flush_p(para_buf)
            if not in_table:
                # Check if next line is separator
                if i + 1 < len(lines) and re.match(r'^\|[-:| ]+\|', lines[i + 1]):
                    html_lines.append('<table><thead><tr>')
                    cells = [c.strip() for c in line.strip('|').split('|')]
                    for c in cells:
                        html_lines.append(f'<th>{inline(c)}</th>')
                    html_lines.append('</tr></thead><tbody>')
                    in_table = True
                    i += 2  # skip header + separator
                    continue
                else:
                    html_lines.append('<table><tbody>')
                    in_table = True
            # Body row
            cells = [c.strip() for c in line.strip('|').split('|')]
            html_lines.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in cells) + '</tr>')
            i += 1
            continue

        if in_table and not line.startswith('|'):
            html_lines.append("</tbody></table>")
            in_table = False

        # Unordered list
        m = re.match(r'^[-*+]\s+(.*)', line)
        if m:
            flush_p(para_buf)
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{inline(m.group(1))}</li>")
            i += 1
            continue

        # Raw HTML lines (references section)
        if line.strip().startswith('<p>') or line.strip().startswith('<'):
            flush_p(para_buf)
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            html_lines.append(line)
            i += 1
            continue

        # Continuation of paragraph / new paragraph line
        para_buf.append(inline(line) + " ")
        i += 1

    flush_p(para_buf)
    if in_ul:
        html_lines.append("</ul>")
    if in_table:
        html_lines.append("</tbody></table>")

    return "\n".join(html_lines)


CSS = """
/* ── Page setup ─────────────────────────────────────────────────────── */
@page {
    size: letter;
    margin: 14mm 14mm 14mm 14mm;
}

/* ── Base typography ─────────────────────────────────────────────────── */
body {
    font-family: "Linux Libertine O", "Palatino Linotype", Georgia, serif;
    font-size: 9pt;
    line-height: 1.38;
    color: #111;
    margin: 0; padding: 0;
}

/* ── Title block (single column, full width) ─────────────────────────── */
.title-block {
    text-align: center;
    margin-bottom: 6pt;
}
.title-block h1 {
    font-size: 14pt;
    margin: 0 0 4pt 0;
    line-height: 1.2;
}
.title-block .authors {
    font-size: 9.5pt;
    margin: 2pt 0;
}
.title-block .venue {
    font-size: 8.5pt;
    font-style: italic;
    color: #444;
    margin: 2pt 0 6pt 0;
}

/* ── Abstract (2-column to save space) ─────────────────────────────── */
.abstract {
    margin: 0 0 6pt 0;
    font-size: 8.5pt;
}
.abstract h2 {
    font-size: 9pt;
    font-weight: bold;
    text-align: center;
    margin: 0 0 3pt 0;
    border: none;
    column-span: all;
}
.abstract-inner {
    column-count: 2;
    column-gap: 12pt;
    text-align: justify;
}
.abstract p {
    margin: 0 0 2pt 0;
}

.keywords {
    font-size: 8pt;
    margin: 4pt 0 8pt 0;
}
.keywords strong { font-weight: bold; }

hr { border: none; border-top: 0.5pt solid #888; margin: 6pt 0; }

/* ── Two-column body ─────────────────────────────────────────────────── */
.body-columns {
    column-count: 2;
    column-gap: 12pt;
    column-rule: 0.5pt solid #ccc;
    text-align: justify;
    column-fill: auto;
}

/* ── Headings ────────────────────────────────────────────────────────── */
h1 { font-size: 13pt; }
h2 {
    font-size: 10pt;
    font-weight: bold;
    margin: 6pt 0 2pt 0;
    break-after: avoid;
}
h3 {
    font-size: 9.5pt;
    font-weight: bold;
    margin: 4pt 0 2pt 0;
    break-after: avoid;
}

/* ── Paragraphs ──────────────────────────────────────────────────────── */
p { margin: 0 0 3pt 0; }

/* ── Figures — inline in column flow (no column-span) ───────────────── */
figure {
    margin: 5pt 0 4pt 0;
    text-align: center;
    break-inside: avoid;
    /* No column-span: figures flow within the column, no forced breaks */
}
figure img {
    /* Fit within one column width (≈260pt). max-height limits tall diagrams.
       Both constraints together prevent any single figure from dominating. */
    max-width: 100%;
    max-height: 200pt;
    width: auto;
    height: auto;
    display: block;
    margin: 0 auto 3pt auto;
}
figcaption {
    font-size: 7pt;
    color: #333;
    text-align: left;
    line-height: 1.3;
}

/* ── Tables ──────────────────────────────────────────────────────────── */
table {
    border-collapse: collapse;
    font-size: 7.5pt;
    width: 100%;
    margin: 3pt 0;
    break-inside: avoid;
}
th, td {
    border: 0.5pt solid #aaa;
    padding: 2pt 4pt;
}
th {
    background: #eee;
    font-weight: bold;
    text-align: center;
}
td { text-align: left; }

/* ── Wide tables span both columns ──────────────────────────────────── */
.wide-table {
    column-span: all;
}

/* ── Lists ───────────────────────────────────────────────────────────── */
ul {
    margin: 2pt 0 4pt 0;
    padding-left: 14pt;
}
li { margin-bottom: 2pt; }

/* ── Code ────────────────────────────────────────────────────────────── */
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 8pt;
    background: #f5f5f5;
    padding: 0 2pt;
}

/* ── References ─────────────────────────────────────────────────────── */
.references {
    column-count: 2;
    column-gap: 14pt;
    font-size: 7.5pt;
}
.references p {
    margin-bottom: 3pt;
    break-inside: avoid;
}

/* ── Disclosure block ────────────────────────────────────────────────── */
.disclosure {
    font-size: 8pt;
    margin: 6pt 0;
    padding: 4pt 6pt;
    border-left: 2pt solid #aaa;
    break-inside: avoid;
}
"""


def build_html(md_path: Path) -> str:
    md_text = md_path.read_text()

    # ── Split out title/authors, abstract, body, references ────────────
    # Title is H1, authors are **…**, venue is *…*
    # Abstract section ends at "---" after keywords
    # References section starts at "## References"

    lines = md_text.split("\n")

    # Find title (first H1)
    title_line = next((l for l in lines if l.startswith("# ")), "")
    title = title_line.lstrip("# ").strip()

    # Authors line: first **...** line after the title
    title_idx = next((i for i, l in enumerate(lines) if l.startswith("# ")), 0)
    authors_line = next((l for l in lines[title_idx:] if l.startswith("**") and l.endswith("**")), "")
    authors = authors_line.strip("*").strip()

    # Venue line (italic)
    venue_line = next((l for l in lines if l.startswith("*") and "MLSys" in l), "")
    venue = venue_line.strip("*").strip()

    # Abstract block: between "## Abstract" and first "---" after it
    try:
        abs_start = next(i for i, l in enumerate(lines) if l.strip() == "## Abstract")
        abs_end   = next(i for i, l in enumerate(lines) if i > abs_start and l.strip().startswith("---"))
    except StopIteration:
        abs_start, abs_end = 0, 0

    abstract_md = "\n".join(lines[abs_start + 1 : abs_end]).strip()
    # Extract keywords line
    kw_match = re.search(r'\*\*Keywords:\*\*.*', abstract_md)
    keywords_html = ""
    if kw_match:
        keywords_html = f'<p class="keywords">{inline_convert(kw_match.group(0))}</p>'
        abstract_md = abstract_md[: kw_match.start()].strip()

    abstract_html = md_to_html(abstract_md)

    # Disclosure block: "## AI / Agent-Assisted Development Disclosure"
    try:
        disc_start = next(i for i, l in enumerate(lines) if "Agent-Assisted Development Disclosure" in l)
        disc_end   = next(
            (i for i, l in enumerate(lines) if i > disc_start + 1 and l.strip().startswith("---")),
            len(lines),
        )
    except StopIteration:
        disc_start, disc_end = -1, -1

    # References block
    try:
        ref_start = next(i for i, l in enumerate(lines) if l.strip() == "## References")
    except StopIteration:
        ref_start = len(lines)

    # Body: from after the second "---" to before the disclosure (or references)
    body_start = abs_end + 1
    body_end   = disc_start if disc_start > 0 else ref_start

    body_md = "\n".join(lines[body_start:body_end]).strip()

    # Fix image paths to be relative to repo root
    body_md = body_md.replace("images/diagrams/", str(DIAG_DIR) + "/")

    body_html = md_to_html(body_md)

    # Disclosure
    if disc_start > 0:
        disc_md = "\n".join(lines[disc_start : disc_end]).strip()
        disc_html = md_to_html(disc_md)
    else:
        disc_html = ""

    # References
    ref_md = "\n".join(lines[ref_start:]).strip()
    ref_html = md_to_html(ref_md)

    # ── Post-process: make wide tables span columns ─────────────────────
    # Tables in the per-workload section (UUID tables) should span both cols
    body_html = re.sub(
        r'(<table>)(.*?)(</table>)',
        lambda m: m.group(0),  # leave as-is; Chrome handles break nicely
        body_html,
        flags=re.DOTALL,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{CSS}
</style>
</head>
<body>

<div class="title-block">
  <h1>{title}</h1>
  <p class="authors">{authors}</p>
  <p class="venue">{venue}</p>
</div>

<div class="abstract">
  <h2>Abstract</h2>
  <div class="abstract-inner">
  {abstract_html}
  </div>
</div>
{keywords_html}

<hr>

<div class="body-columns">
{body_html}
</div>

{"<div class='disclosure'>" + disc_html + "</div>" if disc_html else ""}

<div class="references">
{ref_html}
</div>

</body>
</html>
"""


def inline_convert(s: str) -> str:
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return s


# ── 3. Generate PDF via Chrome headless ──────────────────────────────────────

def build_pdf(html_path: Path, pdf_path: Path):
    print(f"Generating PDF: {pdf_path}")
    result = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            str(html_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("CHROME ERROR:", result.stderr[:500], file=sys.stderr)
        sys.exit(1)
    print(f"  Written: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    crop = "--no-crop" not in sys.argv
    if crop:
        crop_diagrams()

    print("Building HTML...")
    html_content = build_html(REPO / "report.md")

    html_path = REPO / "report.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  Written: {html_path}")

    build_pdf(html_path, OUT_PDF)
    print("Done.")


if __name__ == "__main__":
    main()
