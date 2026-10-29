"""Step 6: the PDF report, from the study folder's Markdown (needs reportlab; matplotlib for the DejaVu fonts).

    python -m kems.report STUDY_DIR [--title "Board X 1.2.0: SI / EMI review"] [--out STUDY_DIR/report.pdf]

Order: STUDY_DIR/SUMMARY.md (written last, by you or Claude: the ranked findings and what to change), TRIAGE.md (step
1), then every FINDINGS_*.md under STUDY_DIR/runs/ (step 5, one section per run, sorted by path), then METHOD.md: the
study's own, else kicad-ems's. Images resolve against the folder of the Markdown that references them. The Markdown reader takes
headings, paragraphs, bullets, numbered items, pipe tables, images and code blocks; nothing else."""
import argparse
import datetime
import glob
import html
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)


def fonts():
    """DejaVu from matplotlib when it is there (Unicode: µ, Ω, ±); Helvetica / Courier otherwise."""
    try:
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        for name, f in (('DV', 'DejaVuSans.ttf'), ('DVB', 'DejaVuSans-Bold.ttf'), ('DVI', 'DejaVuSans-Oblique.ttf'),
                        ('DVM', 'DejaVuSansMono.ttf')):
            pdfmetrics.registerFont(TTFont(name, os.path.join(d, f)))
        addMapping('DV', 0, 0, 'DV')
        addMapping('DV', 1, 0, 'DVB')
        addMapping('DV', 0, 1, 'DVI')
        addMapping('DV', 1, 1, 'DVB')
        return 'DV', 'DVB', 'DVI', 'DVM'
    except Exception:
        return 'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Courier'


FN, FB, FI, FM = fonts()


INK, INK2, RULE, BAND = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#d9d8d3'), \
    colors.HexColor('#f2f1ed')
ST = {
    'title': ParagraphStyle('title', fontName=FB, fontSize=17, leading=21, textColor=INK, spaceAfter=4),
    'sub': ParagraphStyle('sub', fontName=FN, fontSize=9, leading=12, textColor=INK2, spaceAfter=10),
    'h1': ParagraphStyle('h1', fontName=FB, fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5),
    'h2': ParagraphStyle('h2', fontName=FB, fontSize=10.5, leading=13, textColor=INK, spaceBefore=8, spaceAfter=3),
    'h3': ParagraphStyle('h3', fontName=FB, fontSize=9.2, leading=12, textColor=INK, spaceBefore=6, spaceAfter=2),
    'p': ParagraphStyle('p', fontName=FN, fontSize=8.6, leading=11.8, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    'li': ParagraphStyle('li', fontName=FN, fontSize=8.6, leading=11.8, textColor=INK, leftIndent=12, bulletIndent=2,
                         spaceAfter=3, bulletFontName=FN),
    'cap': ParagraphStyle('cap', fontName=FI, fontSize=7.6, leading=10, textColor=INK2, spaceAfter=8),
    'cell': ParagraphStyle('cell', fontName=FN, fontSize=7.1, leading=9.0, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName=FB, fontSize=7.1, leading=9.0, textColor=INK),
    'code': ParagraphStyle('code', fontName=FM, fontSize=6.8, leading=8.6, textColor=INK, leftIndent=8, spaceAfter=4),
}
W = 7.0 * inch


def P(t, s='p'):
    return Paragraph(t, ST[s])


def fig(path, width=W, caption=None):
    if not os.path.exists(path):
        return [P('<i>[figure %s not available]</i>' % os.path.basename(path), 'cap')]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    h = width * ih / iw
    if h > 8.6 * inch:                                  # keep a tall figure on one page
        width, h = width * 8.6 * inch / h, 8.6 * inch
    out = [Image(path, width=width, height=h)]
    if caption:
        out.append(P(caption, 'cap'))
    return [KeepTogether(out)]


def table(rows, widths, header=1, zebra=True):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), ST['cellb' if r < header else 'cell'])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header)
    sty = [('LINEBELOW', (0, header - 1), (-1, header - 1), 0.6, INK2), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
           ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
           ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3),
           ('LINEBELOW', (0, -1), (-1, -1), 0.4, RULE)]
    if zebra:
        for r in range(header, len(rows)):
            if (r - header) % 2 == 1:
                sty.append(('BACKGROUND', (0, r), (-1, r), BAND))
    t.setStyle(TableStyle(sty))
    return t


def md_inline(t):
    t = html.unescape(t).replace('\\|', '|')
    t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    code = []                                                            # code spans keep their * (results/*.json)
    t = re.sub(r'`([^`]+)`', lambda m: code.append(m.group(1)) or '\x00%d\x00' % (len(code) - 1), t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', t)                     # links: their text
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])', r'<i>\1</i>', t)
    return re.sub('\x00(\\d+)\x00', lambda m: '<font face="%s">%s</font>' % (FM, code[int(m.group(1))]), t)


def md_story(md, base, top=1):
    """A small Markdown reader: headings, paragraphs, bullets, numbered items, pipe tables, images, code blocks.
    Heading level L becomes h1 / h2 / h3 from `top` on; images resolve against `base`."""
    out, para, stack, i = [], [], [], 0
    lines = md.splitlines()

    def flush():
        if para:
            out.append(P(md_inline(' '.join(para))))
            para.clear()
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.strip().startswith('```'):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(Preformatted('\n'.join(code), ST['code']))
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        m = re.match(r'^(#{1,4})\s+(.*)', ln)
        if m:
            flush()
            lvl = min(3, len(m.group(1)) - 1 + top)
            out.append(CondPageBreak(1.3 * inch))
            out.append(P(md_inline(m.group(2)), 'h%d' % lvl))
            i += 1
            continue
        m = re.match(r'^!\[(.*?)\]\((.*?)\)', ln.strip())
        if m:
            flush()
            p = m.group(2)
            if not os.path.isabs(p):
                p = os.path.join(base, p)
            out += fig(p, caption=md_inline(m.group(1)) if m.group(1) else None)
            i += 1
            continue
        if ln.strip().startswith('|'):
            flush()
            raw = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                if not all(re.match(r'^:?-{2,}:?$', c) for c in cells if c):
                    raw.append(cells)
                i += 1
            if raw:
                ncol = max(len(r) for r in raw)
                raw = [r + [''] * (ncol - len(r)) for r in raw]
                lens = [min(60, max(8, max(len(r[c]) for r in raw))) for c in range(ncol)]
                widths = [W * L / sum(lens) for L in lens]
                rows = [[P(md_inline(c), 'cellb' if k == 0 else 'cell') for c in r] for k, r in enumerate(raw)]
                out.append(table(rows, widths))
                out.append(Spacer(1, 6))
            continue
        m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)', ln)
        if m:
            flush()
            item = [m.group(3)]
            i += 1
            while i < len(lines) and lines[i].startswith(' ' * (len(m.group(1)) + 2)) and lines[i].strip() and \
                    not re.match(r'^\s*([-*]|\d+\.)\s+', lines[i]):
                item.append(lines[i].strip())
                i += 1
            bt = '•' if m.group(2) in '-*' else m.group(2)
            ind = len(m.group(1))                     # nesting level from the indents seen in this list
            while stack and stack[-1] > ind:
                stack.pop()
            if not stack or stack[-1] < ind:
                stack.append(ind)
            lvl = len(stack) - 1
            st = ParagraphStyle('lix', parent=ST['li'], leftIndent=12 + 12 * lvl, bulletIndent=2 + 12 * lvl)
            out.append(Paragraph(md_inline(' '.join(item)), st, bulletText=bt))
            continue
        para.append(ln.strip())
        stack.clear()
        i += 1
    flush()
    return out


def read(path):
    return open(path).read() if os.path.exists(path) else None


def strip_title(md):
    return re.sub(r'^# .*\n', '', md, count=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('study')
    ap.add_argument('--title')
    ap.add_argument('--out')
    a = ap.parse_args()
    study = os.path.abspath(a.study)
    out = a.out or os.path.join(study, 'report.pdf')
    date = datetime.date.today().isoformat()
    title = a.title or '%s: signal integrity and EMI review' % os.path.basename(study)
    story = [P(md_inline(title), 'title'),
             P('openEMS FDTD on rented GPUs (kicad-ems, openems-gpu). %s.' % date, 'sub')]
    parts = [('SUMMARY.md', None), ('TRIAGE.md', 'Triage: what was looked at, and why')]
    for name, head in parts:
        md = read(os.path.join(study, name))
        if md:
            if head:
                story.append(PageBreak())
                story.append(P(head, 'h1'))
                md = strip_title(md)
            story += md_story(md, study, top=1 if not head else 2)
    runs = sorted(glob.glob(os.path.join(study, 'runs', '**', 'FINDINGS_*.md'), recursive=True))
    if runs:
        story.append(PageBreak())
        story.append(P('Simulation results', 'h1'))
        for f in runs:
            story.append(CondPageBreak(2 * inch))
            story += md_story(open(f).read(), os.path.dirname(f), top=2)
    md = read(os.path.join(study, 'METHOD.md')) or read(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'METHOD.md'))         # the study's own, else kicad-ems's
    if md:
        story.append(PageBreak())
        story.append(P('Method and limits', 'h1'))
        story += md_story(strip_title(md), study, top=2)
    doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch, title=title, author='kicad-ems')

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(FN, 7)
        canvas.setFillColor(INK2)
        canvas.drawString(0.75 * inch, 0.45 * inch, '%s, %s' % (re.sub('<[^>]+>', '', title), date))
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch, 'page %d' % doc_.page)
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('wrote', out)


if __name__ == '__main__':
    main()
