"""Generate polished bachelor thesis presentation in FIS VŠE template.

v3 — expanded to 15 slides with proper explanation of key constructs.
Each slide uses a different layout (cards, flowchart, native charts, concept
diagrams). All charts are native PowerPoint charts so they stay editable.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy

from pptx import Presentation
from pptx.util import Pt, Cm, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.oxml.ns import qn

REPO = Path(r"C:\Users\WX794ZX\Downloads\Pers\School\Bakalarka")
SRC = REPO / "prezentace_v2.pptx"
OUT = REPO / "prezentace_v3.pptx"

# FIS / VŠE palette
FIS_GREEN = RGBColor(0x00, 0x98, 0x81)
FIS_GREEN_LIGHT = RGBColor(0xCC, 0xEB, 0xE6)
FIS_DARK = RGBColor(0x4A, 0x4A, 0x49)
FIS_CYAN = RGBColor(0x5B, 0xC4, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY_BG = RGBColor(0xF4, 0xF4, 0xF2)
GREY_LINE = RGBColor(0xDD, 0xDD, 0xDD)
TEXT_DARK = RGBColor(0x2B, 0x2B, 0x2B)
RED = RGBColor(0xC0, 0x39, 0x2B)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
ORANGE_LIGHT = RGBColor(0xFD, 0xF1, 0xE7)
BLUE = RGBColor(0x2E, 0x6C, 0xA8)
FONT = "Arial"
FONT_MATH = "Cambria Math"

# --- open template (from existing v2 pptx) ---------------------------------
prs = Presentation(SRC)
SW, SH = prs.slide_width, prs.slide_height

# Remove any sample slides shipped inside the template
xml_slides = prs.slides._sldIdLst
slides_list = list(xml_slides)
for sld in slides_list:
    rId = sld.get(qn('r:id'))
    prs.part.drop_rel(rId)
    xml_slides.remove(sld)

M0 = prs.slide_masters[0]
M2 = prs.slide_masters[2]
COVER = M0.slide_layouts[0]
TITLE_CONTENT = M2.slide_layouts[0]
SECTION = M2.slide_layouts[2]


# ======================================================================
# helpers
# ======================================================================
def rm_ph(slide, idx):
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx == idx:
            sp = ph._element; sp.getparent().remove(sp)
            return


def set_title(slide, text, color=FIS_DARK, size=22):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == 0:
            tf = ph.text_frame; tf.clear()
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
            r = p.add_run(); r.text = text
            r.font.name = FONT; r.font.size = Pt(size); r.font.bold = True
            r.font.color.rgb = color
            return ph


def add_para(tf, text, *, size=14, bold=False, color=TEXT_DARK,
             align=PP_ALIGN.LEFT, first=False, space_before=0, italic=False,
             font=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    r = p.add_run(); r.text = text
    r.font.name = font or FONT; r.font.size = Pt(size)
    r.font.bold = bold; r.font.italic = italic
    r.font.color.rgb = color
    return p


def add_equation(tf, text, *, size=14, color=TEXT_DARK, align=PP_ALIGN.CENTER,
                 first=False, space_before=0, bold=False):
    return add_para(tf, text, size=size, color=color, align=align,
                    first=first, space_before=space_before, bold=bold,
                    italic=True, font=FONT_MATH)


def add_rect(slide, left, top, width, height, *, fill=None, line=None,
             radius=None, anchor=MSO_ANCHOR.TOP, shape=MSO_SHAPE.RECTANGLE):
    box = slide.shapes.add_shape(shape, left, top, width, height)
    box.shadow.inherit = False
    if fill is None:
        box.fill.background()
    else:
        box.fill.solid(); box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line; box.line.width = Pt(0.75)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Cm(0.25); tf.margin_right = Cm(0.25)
    tf.margin_top = Cm(0.15); tf.margin_bottom = Cm(0.15)
    tf.vertical_anchor = anchor
    return box


def add_arrow(slide, x1, y1, x2, y2, color=FIS_GREEN, weight=2.0,
              end_arrow=True):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    ln = line.line._get_or_add_ln()
    if end_arrow:
        from lxml import etree
        tail = ln.find(qn('a:tailEnd'))
        if tail is None:
            tail = etree.SubElement(ln, qn('a:tailEnd'))
        tail.set('type', 'triangle'); tail.set('w', 'med'); tail.set('len', 'med')
    return line


def add_text_only(slide, left, top, width, height, text, *,
                  size=12, bold=False, color=TEXT_DARK,
                  align=PP_ALIGN.LEFT, italic=False, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    add_para(tf, text, size=size, bold=bold, color=color, align=align,
             italic=italic, first=True)
    return tb


def add_accent_bar(slide, *, top, width=Cm(1.5), left=Cm(1.2),
                   color=FIS_GREEN, height=Pt(4)):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    bar.fill.solid(); bar.fill.fore_color.rgb = color
    bar.line.fill.background(); bar.shadow.inherit = False
    return bar


def add_footer(slide, page_no, total,
               text="\u0160. Slansk\u00fd \u00b7 Bakal\u00e1\u0159sk\u00fd semin\u00e1\u0159 \u00b7 V\u0160E FIS"):
    add_text_only(slide, Cm(1.2), SH - Cm(0.7), SW - Cm(4.0), Cm(0.55),
                  text, size=9, color=RGBColor(0x99, 0x99, 0x99))
    add_text_only(slide, SW - Cm(2.6), SH - Cm(0.7), Cm(1.4), Cm(0.55),
                  f"{page_no} / {total}", size=9,
                  color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


def add_bullets(slide, left, top, width, height, items, *, size=13,
                line_spacing=1.15, bullet_color=FIS_GREEN):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = 0
    glyphs = {0: "\u25aa", 1: "\u2013", 2: "\u00b7"}
    first = True
    for entry in items:
        if isinstance(entry, tuple):
            level, text = entry
        else:
            level, text = 0, entry
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        p.space_after = Pt(3)
        g = glyphs.get(level, "\u00b7")
        r1 = p.add_run()
        r1.text = (" " * (level * 4)) + g + "  "
        r1.font.name = FONT; r1.font.size = Pt(size); r1.font.bold = True
        r1.font.color.rgb = bullet_color
        r2 = p.add_run(); r2.text = text
        r2.font.name = FONT; r2.font.size = Pt(size - (1 if level > 0 else 0))
        r2.font.color.rgb = TEXT_DARK
    return tb


def add_callout(slide, left, top, width, height, text, *,
                fill=FIS_GREEN_LIGHT, accent=FIS_GREEN, size=12, bold=True):
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Cm(0.18), height)
    stripe.fill.solid(); stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background(); stripe.shadow.inherit = False
    box = add_rect(slide, left + Cm(0.18), top, width - Cm(0.18), height,
                   fill=fill, anchor=MSO_ANCHOR.MIDDLE)
    tf = box.text_frame; tf.margin_left = Cm(0.4)
    add_para(tf, text, size=size, color=FIS_DARK, bold=bold, first=True)
    return box


def add_kpi(slide, left, top, width, height, value, label, *,
            value_color=FIS_GREEN, value_size=22):
    box = add_rect(slide, left, top, width, height, fill=WHITE, line=GREY_LINE)
    tf = box.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_para(tf, value, size=value_size, bold=True, color=value_color,
             align=PP_ALIGN.CENTER, first=True)
    add_para(tf, label, size=10, color=FIS_DARK, align=PP_ALIGN.CENTER,
             space_before=2)
    return box


def header_card(slide, left, top, width, height, header_text, header_color,
                hdr_h=Cm(1.0)):
    box = add_rect(slide, left, top, width, height, fill=WHITE, line=GREY_LINE)
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, hdr_h)
    hdr.fill.solid(); hdr.fill.fore_color.rgb = header_color
    hdr.line.fill.background(); hdr.shadow.inherit = False
    htf = hdr.text_frame; htf.vertical_anchor = MSO_ANCHOR.MIDDLE
    htf.margin_left = Cm(0.4)
    add_para(htf, header_text, size=12, bold=True, color=WHITE, first=True)
    return box, top + hdr_h


def style_chart(chart, *, series_colors, value_format='0.00',
                show_legend=False, axis_font_size=10):
    chart.has_title = False
    chart.has_legend = show_legend
    if show_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(axis_font_size)
        chart.legend.font.name = FONT
    plot = chart.plots[0]
    for i, ser in enumerate(plot.series):
        color = series_colors[i % len(series_colors)]
        fill = ser.format.fill
        fill.solid(); fill.fore_color.rgb = color
        ser.format.line.fill.background()
    try:
        for axis in (chart.category_axis, chart.value_axis):
            tf = axis.tick_labels.font
            tf.size = Pt(axis_font_size); tf.name = FONT
            tf.color.rgb = FIS_DARK
        chart.value_axis.tick_labels.number_format = value_format
    except Exception:
        pass


TOTAL_SLIDES = 16


def new_slide(title_text, page_no):
    s = prs.slides.add_slide(TITLE_CONTENT)
    for idx in (1, 10, 11, 12):
        rm_ph(s, idx)
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 0:
            ph.left = Cm(1.2); ph.top = Cm(3.2)
            ph.width = Cm(23.0); ph.height = Cm(1.2)
            break
    set_title(s, title_text, color=FIS_DARK, size=22)
    add_accent_bar(s, top=Cm(4.45))
    add_footer(s, page_no, TOTAL_SLIDES)
    return s


# ======================================================================
# 1 \u2014 Cover
# ======================================================================
s = prs.slides.add_slide(COVER)
for ph in s.placeholders:
    idx = ph.placeholder_format.idx
    if idx == 3:
        ph.text_frame.clear()
        tf = ph.text_frame
        add_para(tf, "Dekompozice jazyka rizikov\u00fdch faktor\u016f",
                 size=28, bold=True, color=FIS_DARK, first=True)
        add_para(tf, "v 10-K v\u00fdkazech a n\u00e1sledn\u00e1 volatilita akci\u00ed",
                 size=24, bold=True, color=FIS_GREEN, space_before=4)
    elif idx == 13:
        ph.text_frame.clear()
        add_para(ph.text_frame, "\u0160imon Slansk\u00fd", size=14, bold=True,
                 color=FIS_DARK, first=True)
        add_para(ph.text_frame, "Vedouc\u00ed: prof. RNDr. Ing. Michal \u010cern\u00fd, Ph.D.",
                 size=11, color=FIS_DARK)
    elif idx == 14:
        ph.text_frame.clear()
        add_para(ph.text_frame, "V\u0160E Praha \u00b7 Fakulta informatiky a statistiky",
                 size=11, color=FIS_DARK, first=True)
        add_para(ph.text_frame, "Praha \u00b7 2026", size=11, color=FIS_DARK)


# ======================================================================
# 2 \u2014 Motivation: two literatures
# ======================================================================
s = new_slide("Motivace \u2014 dv\u011b protich\u016fdn\u00e9 literatury", 2)

intro = add_text_only(s, Cm(1.2), Cm(4.7), Cm(22.5), Cm(1.2),
    "Item 1A (Risk Factors) je p\u0159edm\u011btem dvou paraleln\u00edch literatur, "
    "kter\u00e9 p\u0159edpov\u00eddaj\u00ed efekty v opa\u010dn\u00fdch sm\u011brech.",
    size=13, color=FIS_DARK)

card_top = Cm(6.1); card_h = Cm(5.6); card_w = Cm(8.2)
left_card_x = Cm(1.2)
right_card_x = SW - Cm(1.2) - card_w
center_x = (SW - Cm(4.6)) // 2; center_w = Cm(4.6); center_h = Cm(2.8)
center_y = card_top + (card_h - center_h) // 2

box, ct = header_card(s, left_card_x, card_top, card_w, card_h,
                      "Campbell et al. (2014)", FIS_GREEN)
tf = box.text_frame; tf.margin_top = Cm(1.2); tf.margin_left = Cm(0.4)
add_para(tf, "VOLUME channel", size=10, bold=True, color=FIS_GREEN,
         align=PP_ALIGN.CENTER, first=True)
add_para(tf, "\u2191 d\u00e9lka Item 1A", size=20, bold=True, color=FIS_DARK,
         align=PP_ALIGN.CENTER, space_before=4)
add_para(tf, "\u2192 vy\u0161\u0161\u00ed n\u00e1sledn\u00e1 volatilita", size=12, italic=True,
         color=TEXT_DARK, align=PP_ALIGN.CENTER, space_before=4)

box, ct = header_card(s, right_card_x, card_top, card_w, card_h,
                      "Hope et al. (2016)", FIS_DARK)
tf = box.text_frame; tf.margin_top = Cm(1.2); tf.margin_left = Cm(0.4)
add_para(tf, "SPECIFICITY channel", size=10, bold=True, color=FIS_DARK,
         align=PP_ALIGN.CENTER, first=True)
add_para(tf, "\u2191 specifi\u010dnost", size=20, bold=True, color=FIS_DARK,
         align=PP_ALIGN.CENTER, space_before=4)
add_para(tf, "\u2192 ni\u017e\u0161\u00ed n\u00e1sledn\u00e1 volatilita", size=12, italic=True,
         color=TEXT_DARK, align=PP_ALIGN.CENTER, space_before=4)

oval = add_rect(s, center_x, center_y, center_w, center_h,
                fill=FIS_GREEN_LIGHT, line=FIS_GREEN, anchor=MSO_ANCHOR.MIDDLE,
                shape=MSO_SHAPE.OVAL)
tf = oval.text_frame
add_equation(tf, "\u03c3 post-filing", size=16, bold=True, color=FIS_GREEN,
             align=PP_ALIGN.CENTER, first=True)
add_para(tf, "post-filing volatilita", size=10, color=FIS_DARK,
         align=PP_ALIGN.CENTER, space_before=2)

mid_y = card_top + card_h // 2
add_arrow(s, left_card_x + card_w, mid_y, center_x, mid_y,
          color=FIS_GREEN, weight=2.5)
add_arrow(s, right_card_x, mid_y, center_x + center_w, mid_y,
          color=FIS_DARK, weight=2.5)

add_callout(s, Cm(1.2), Cm(13.0), Cm(22.5), Cm(2.0),
            "Tato pr\u00e1ce odd\u011bluje OBJEM a SPECIFI\u010cNOST v jedn\u00e9 regresi a dopl\u0148uje "
            "t\u0159et\u00ed kan\u00e1l \u2014 co firma v\u016f\u010di peer\u016fm VYNECH\u00c1.")


# ======================================================================
# 3 \u2014 NEW: What is Item 1A + why it matters
# ======================================================================
s = new_slide("Co je Item 1A a pro\u010d na n\u011bm z\u00e1le\u017e\u00ed", 3)

add_text_only(s, Cm(1.2), Cm(4.7), Cm(11.5), Cm(0.8),
              "Struktura v\u00fdro\u010dn\u00ed zpr\u00e1vy (10-K)", size=12, bold=True,
              color=FIS_DARK)

levels = [
    ("10-K filing (cel\u00fd dokument)", GREY_BG, FIS_DARK),
    ("Item 1A \u2014 Risk Factors", FIS_GREEN_LIGHT, FIS_GREEN),
    ("Item 7 \u2014 MD&A (dopl\u0148kov\u00fd text)", GREY_BG, FIS_DARK),
]
ly = Cm(5.7)
for label, bg, col in levels:
    box = add_rect(s, Cm(1.2), ly, Cm(11.5), Cm(1.6), fill=bg, line=GREY_LINE)
    tf = box.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_para(tf, label, size=12, bold=True, color=col,
             align=PP_ALIGN.CENTER, first=True)
    ly += Cm(1.8)

right_x = Cm(13.5); right_w = Cm(10.5)
add_text_only(s, right_x, Cm(4.7), right_w, Cm(0.8),
              "Kl\u00ed\u010dov\u00e9 vlastnosti Item 1A", size=12, bold=True,
              color=FIS_DARK)
add_bullets(s, right_x, Cm(5.7), right_w, Cm(7.0), [
    "Povinn\u00e1 sekce od 2005 (SEC Regulation S-K)",
    "Popisuje materi\u00e1ln\u00ed rizika, jim\u017e firma \u010del\u00ed",
    "Medi\u00e1n ~8 800 slov; rozsah 2 000\u201340 000",
    "Firma vol\u00ed, CO uvede a v jak\u00e9m rozsahu",
    "Pr\u00e1vn\u00ed safe harbour motivuje boilerplate",
    (1, "\u2192 pr\u00e1vn\u00edci p\u0159id\u00e1vaj\u00ed standardn\u00ed formulace"),
    (1, "\u2192 firm-specific obsah se z\u0159ed\u00ed"),
], size=12)

add_callout(s, Cm(1.2), Cm(13.5), Cm(22.5), Cm(1.6),
            "Ot\u00e1zka: nese d\u00e9lka, slovn\u00edk, nebo to, co firma VYNECH\u00c1, "
            "informaci o budouc\u00ed volatilit\u011b?", size=12)


# ======================================================================
# 4 \u2014 Hypotheses (3 cards)
# ======================================================================
s = new_slide("Hypot\u00e9zy", 4)

card_top = Cm(4.8); card_h = Cm(9.0); card_w = Cm(7.2); gap = Cm(0.4)
left0 = (SW - (card_w * 3 + gap * 2)) // 2

def hyp_card(left, label, sign, channel, claim):
    box, ct = header_card(s, left, card_top, card_w, card_h, label,
                          FIS_GREEN, hdr_h=Cm(1.2))
    tf = box.text_frame
    tf.margin_top = Cm(1.5); tf.margin_left = Cm(0.5); tf.margin_right = Cm(0.5)
    add_para(tf, channel, size=11, bold=True, color=FIS_GREEN,
             align=PP_ALIGN.CENTER, first=True)
    add_para(tf, sign, size=40, bold=True, color=FIS_DARK,
             align=PP_ALIGN.CENTER, space_before=4)
    add_para(tf, claim, size=12, color=TEXT_DARK,
             align=PP_ALIGN.CENTER, space_before=6)

hyp_card(left0,                    "H1", "\u2191", "OBJEM",
         "Del\u0161\u00ed Item 1A \u2192 vy\u0161\u0161\u00ed n\u00e1sledn\u00e1 volatilita.")
hyp_card(left0 + card_w + gap,     "H2", "\u2193", "SPECIFI\u010cNOST",
         "Vy\u0161\u0161\u00ed hustota rizikov\u00fdch slov p\u0159i kontrole d\u00e9lky \u2192 ni\u017e\u0161\u00ed volatilita.")
hyp_card(left0 + 2*(card_w + gap), "H3", "\u2191", "ML\u010cEN\u00cd",
         "V\u011bt\u0161\u00ed peer-relativn\u00ed opomenut\u00ed t\u00e9mat \u2192 vy\u0161\u0161\u00ed volatilita.")

add_callout(s, Cm(1.2), Cm(14.2), Cm(22.5), Cm(1.4),
            "Druh\u00fd stupe\u0148 H2: efekt nese LM-Litigious sub-list, nikoli LM-Uncertainty.",
            size=11)


# ======================================================================
# 5 \u2014 Data pipeline
# ======================================================================
s = new_slide("Data \u2014 pipeline", 5)

kpi_top = Cm(4.7); kpi_h = Cm(2.2); kpi_w = Cm(5.3); kpi_gap = Cm(0.35)
left0 = (SW - (kpi_w * 4 + kpi_gap * 3)) // 2
add_kpi(s, left0,                       kpi_top, kpi_w, kpi_h, "535",
        "firem (S&P 1500, nefinan\u010dn\u00ed)")
add_kpi(s, left0 + (kpi_w + kpi_gap),   kpi_top, kpi_w, kpi_h, "2010\u20132024",
        "fisk\u00e1ln\u00edch let")
add_kpi(s, left0 + 2*(kpi_w + kpi_gap), kpi_top, kpi_w, kpi_h, "~5 700",
        "firma-let (h = 30 d)")
add_kpi(s, left0 + 3*(kpi_w + kpi_gap), kpi_top, kpi_w, kpi_h, "93,5 %",
        "LLM-audit p\u0159esnost")

flow_top = Cm(8.4)
flow_h = Cm(2.6)
n_boxes = 5
total_flow_w = SW - Cm(2.4)
box_w = Cm(4.0); gap_x = (total_flow_w - box_w * n_boxes) // (n_boxes - 1)
flow_left0 = Cm(1.2)

stages = [
    ("SEC EDGAR\nSubmissions API", "Seznam 10-K\n2010\u20132024", FIS_GREEN),
    ("XBRL Company\nFacts API", "Finan\u010dn\u00ed metriky\n+ tag locking", FIS_GREEN),
    ("HTML parser\n(Item 1A / 7)", "Texty 10-K\n+ Exhibit 13 fallback", FIS_DARK),
    ("LM slovn\u00edk\n(Unc + Lit)", "Word counts &\ndensities", FIS_DARK),
    ("yfinance +\npost-filing okna", "ln \u03c3\u1d62,\u209c\u208a\u2081\u1d34", FIS_GREEN),
]
for i, (top_text, bot_text, color) in enumerate(stages):
    x = flow_left0 + i * (box_w + gap_x)
    box, ct = header_card(s, x, flow_top, box_w, flow_h, top_text, color,
                          hdr_h=Cm(1.1))
    tf = box.text_frame
    tf.margin_top = Cm(1.2); tf.margin_left = Cm(0.2); tf.margin_right = Cm(0.2)
    add_para(tf, bot_text, size=10, color=TEXT_DARK,
             align=PP_ALIGN.CENTER, first=True)
    if i < n_boxes - 1:
        ax = x + box_w
        ay = flow_top + flow_h // 2
        add_arrow(s, ax, ay, ax + gap_x, ay, color=FIS_DARK, weight=2.0)

out_w = Cm(10.0); out_h = Cm(1.4)
out_x = (SW - out_w) // 2
out_y = flow_top + flow_h + Cm(1.0)
out = add_rect(s, out_x, out_y, out_w, out_h, fill=FIS_GREEN_LIGHT,
               line=FIS_GREEN, anchor=MSO_ANCHOR.MIDDLE)
tf = out.text_frame
add_para(tf, "Panel:  535 firem \u00d7 15 let \u00d7 textov\u00e9 + finan\u010dn\u00ed prom\u011bnn\u00e9",
         size=12, bold=True, color=FIS_DARK,
         align=PP_ALIGN.CENTER, first=True)
add_arrow(s, SW // 2, flow_top + flow_h, SW // 2, out_y,
          color=FIS_DARK, weight=2.0)


# ======================================================================
# 6 \u2014 NEW: Data quality \u2014 tag locking + LLM audit
# ======================================================================
s = new_slide("Kvalita dat \u2014 XBRL tag locking a LLM audit", 6)

col_w = Cm(11.0); col_gap = Cm(0.5)
col1_x = Cm(1.2); col2_x = col1_x + col_w + col_gap

box, _ = header_card(s, col1_x, Cm(4.8), col_w, Cm(10.0),
                     "XBRL tag locking", FIS_GREEN)
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
tf.margin_right = Cm(0.3)
add_para(tf, "Probl\u00e9m:", size=11, bold=True, color=FIS_DARK, first=True)
add_para(tf, "Jeden koncept (nap\u0159. provozn\u00ed CF) m\u016f\u017ee b\u00fdt v XBRL reportov\u00e1n "
             "pod r\u016fzn\u00fdmi tagy v r\u016fzn\u00fdch letech.", size=11, color=TEXT_DARK,
         space_before=3)
add_para(tf, "\u0158e\u0161en\u00ed:", size=11, bold=True, color=FIS_DARK, space_before=6)
add_para(tf, "1. Equivalence groups \u2014 tagy se stejn\u00fdm ekonomick\u00fdm "
             "v\u00fdznamem slou\u010d\u00edme (nap\u0159. NetCashProvided\u2026 vs. "
             "\u2026ContinuingOperations).", size=11, color=TEXT_DARK,
         space_before=3)
add_para(tf, "2. Per-firm tag locking \u2014 ka\u017edou firmu zamkneme na tag, "
             "kter\u00fd pou\u017e\u00edv\u00e1 nej\u010dast\u011bji; ostatn\u00ed = missing.",
         size=11, color=TEXT_DARK, space_before=3)
add_para(tf, "V\u00fdsledek: kompletnost dat 78,9 % \u2192 92,7 %",
         size=11, bold=True, color=FIS_GREEN, space_before=6)

box, _ = header_card(s, col2_x, Cm(4.8), col_w, Cm(10.0),
                     "LLM-assisted quality audit", FIS_DARK)
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
tf.margin_right = Cm(0.3)
add_para(tf, "C\u00edl:", size=11, bold=True, color=FIS_DARK, first=True)
add_para(tf, "Ov\u011b\u0159it, \u017ee HTML parser extrahuje spr\u00e1vnou sekci "
             "(Item 1A / Item 7) a ne TOC, p\u0159\u00edlohu \u010di jinou \u010d\u00e1st.",
         size=11, color=TEXT_DARK, space_before=3)
add_para(tf, "Metoda:", size=11, bold=True, color=FIS_DARK, space_before=6)
add_para(tf, "Gemini 2.5 Flash klasifikuje 200 n\u00e1hodn\u00fdch extrakc\u00ed "
             "(stratifikovan\u011b 2010\u20132024) na 5 krit\u00e9ri\u00ed: identita sekce, "
             "start/end hranice, kvalita, \u00faplnost.",
         size=11, color=TEXT_DARK, space_before=3)
add_para(tf, "V\u00fdsledek: 93,5 % Pass + Minor",
         size=11, bold=True, color=FIS_GREEN, space_before=6)
add_para(tf, "Selh\u00e1n\u00ed koncentrov\u00e1na v 2010\u20132012 (nestandardn\u00ed HTML).",
         size=10, italic=True, color=FIS_DARK, space_before=3)

add_callout(s, Cm(1.2), Cm(15.2), Cm(22.5), Cm(1.3),
            "Wilson 95% CI: 89,2\u201396,2 %. P\u0159\u00edstup validace LLM "
            "podpo\u0159en v Gilardi et al. (2023).", size=11)


# ======================================================================
# 7 \u2014 NEW: LM dictionary explanation
# ======================================================================
s = new_slide("LM slovn\u00edk \u2014 co m\u011b\u0159\u00ed Litigious a Uncertainty", 7)

add_text_only(s, Cm(1.2), Cm(4.7), Cm(22.5), Cm(1.0),
              "Loughran & McDonald (2011) \u2014 standardn\u00ed finan\u010dn\u00ed sentiment slovn\u00edk. "
              "Pou\u017e\u00edv\u00e1me dv\u011b sub-kategorie relevantn\u00ed pro rizikov\u00fd jazyk:",
              size=12, color=FIS_DARK)

card_w = Cm(11.0); card_gap = Cm(0.5)
cx1 = Cm(1.2); cx2 = cx1 + card_w + card_gap
card_top = Cm(6.0); card_h = Cm(5.8)

box, _ = header_card(s, cx1, card_top, card_w, card_h,
                     "LM-Litigious", FIS_GREEN)
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
tf.margin_right = Cm(0.3)
add_para(tf, "Pr\u00e1vn\u011b specifick\u00e9 v\u00fdrazy:", size=11, bold=True,
         color=FIS_DARK, first=True)
add_para(tf, "\"litigation\", \"indemnify\", \"breach\",\n"
             "\"enforcement\", \"claimant\", \"arbitration\"\u2026",
         size=11, italic=True, color=TEXT_DARK, space_before=3)
add_para(tf, "Interpretace:", size=11, bold=True, color=FIS_DARK,
         space_before=6)
add_para(tf, "Standardizovan\u00fd safe-harbour boilerplate \u2014 "
             "pr\u00e1vn\u00edci p\u0159id\u00e1vaj\u00ed jako ochranu p\u0159ed \u017ealobami. "
             "Sign\u00e1l: n\u00edzk\u00e1 firm-specificity.",
         size=11, color=TEXT_DARK, space_before=3)

box, _ = header_card(s, cx2, card_top, card_w, card_h,
                     "LM-Uncertainty", FIS_DARK)
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
tf.margin_right = Cm(0.3)
add_para(tf, "Mod\u00e1ln\u00ed/hedgingov\u00e1 slova:", size=11, bold=True,
         color=FIS_DARK, first=True)
add_para(tf, "\"may\", \"could\", \"possible\", \"might\",\n"
             "\"approximate\", \"uncertain\", \"risk\"\u2026",
         size=11, italic=True, color=TEXT_DARK, space_before=3)
add_para(tf, "Interpretace:", size=11, bold=True, color=FIS_DARK,
         space_before=6)
add_para(tf, "Hedging nezn\u00e1m\u00e9ho \u2014 management signalizuje, "
             "\u017ee v\u00fdsledek nelze p\u0159edpov\u011bd\u011bt. Obecn\u011bj\u0161\u00ed ne\u017e Litigious.",
         size=11, color=TEXT_DARK, space_before=3)

eq_top = Cm(12.2)
eq_box = add_rect(s, Cm(1.2), eq_top, Cm(22.5), Cm(1.8),
                  fill=GREY_BG, anchor=MSO_ANCHOR.MIDDLE)
tf = eq_box.text_frame; tf.margin_left = Cm(0.6)
add_para(tf, "Kompozitn\u00ed prom\u011bnn\u00e1:", size=11, bold=True,
         color=FIS_GREEN, first=True)
add_equation(tf, "RiskDensity = (n_unc + n_lit) / Words\u2081\u2090",
             size=13, color=FIS_DARK, space_before=3)

add_text_only(s, Cm(1.2), Cm(14.3), Cm(22.5), Cm(0.7),
              "Korelace UncDensity vs. LitDensity \u2248 \u22120,05 \u2192 odli\u0161iteln\u00e9 v regresi.",
              size=10, italic=True, color=FIS_DARK)


# ======================================================================
# 8 \u2014 Methodology: equation + variable map
# ======================================================================
s = new_slide("Metodologie \u2014 specifikace modelu", 8)

eq_box = add_rect(s, Cm(1.2), Cm(4.7), Cm(22.5), Cm(2.4),
                  fill=GREY_BG, anchor=MSO_ANCHOR.MIDDLE)
tf = eq_box.text_frame; tf.margin_left = Cm(0.6); tf.margin_right = Cm(0.6)
add_para(tf, "Spole\u010dn\u00fd model H1 + H2", size=11, bold=True,
         color=FIS_GREEN, first=True)
add_equation(tf,
         "ln \u03c3\u1d62,\u209c\u208a\u2081\u1d34  =  \u03b1 + \u03c6 \u00b7 ln \u03c3\u1d62,\u209c\u1d34 + \u03b2\u2032\u00b7X\u1da0\u1d62\u207f"
         "  +  \u03b2\u2097\u2091\u2099 \u00b7 ln(Words\u2081\u2090)  +  \u03b2\u1d48\u2091\u2099\u209b \u00b7 RiskDensity  +  \u03bc\u2c7c + \u03c4\u209c + \u03b5",
         size=13, color=FIS_DARK, space_before=4)

tiles = [
    ("\u03b2\u2097\u2091\u2099", "log d\u00e9lka Item 1A", "H1 \u2014 objem", FIS_GREEN),
    ("\u03b2\u1d48\u2091\u2099\u209b", "RiskDensity (LM)", "H2 \u2014 specifi\u010dnost", FIS_DARK),
    ("\u03b2\u1d64\u2099\u1d9c + \u03b2\u2097\u1d62\u209c", "Dekompozice LM", "Druh\u00fd stupe\u0148 H2", FIS_GREEN),
    ("\u03b2\u2092\u2098", "Omission\u2081\u2090 v\u016f\u010di peer\u016fm", "H3 \u2014 strategick\u00e9 ml\u010den\u00ed", FIS_DARK),
]
tile_top = Cm(7.6); tile_h = Cm(3.0); tile_w = Cm(11.0); gx = Cm(0.5); gy = Cm(0.3)
left0 = (SW - (tile_w * 2 + gx)) // 2

for i, (sym, what, hyp, color) in enumerate(tiles):
    r, c = divmod(i, 2)
    x = left0 + c * (tile_w + gx)
    y = tile_top + r * (tile_h + gy)
    box = add_rect(s, x, y, tile_w, tile_h, fill=WHITE, line=GREY_LINE)
    sym_w = Cm(3.5)
    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, sym_w, tile_h)
    panel.fill.solid(); panel.fill.fore_color.rgb = color
    panel.line.fill.background(); panel.shadow.inherit = False
    ptf = panel.text_frame; ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
    add_equation(ptf, sym, size=20, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER, first=True)
    tb = s.shapes.add_textbox(x + sym_w + Cm(0.3), y + Cm(0.4),
                              tile_w - sym_w - Cm(0.5), tile_h - Cm(0.8))
    tf = tb.text_frame; tf.word_wrap = True
    add_para(tf, what, size=13, bold=True, color=FIS_DARK, first=True)
    add_para(tf, hyp, size=11, italic=True, color=color, space_before=3)

add_text_only(s, Cm(1.2), Cm(14.3), Cm(22.5), Cm(1.5),
              "Z\u00e1visl\u00e1 prom\u011bnn\u00e1: sd(denn\u00edch log-return\u016f) v okn\u011b [filing+2d, filing+h d], "
              "annualizov\u00e1no \u00d7\u221a252, v logu.\n"
              "OLS \u00b7 FE (odv\u011btv\u00ed/firma + rok) \u00b7 SE klastrovan\u00e9 dvouvrstvov\u011b (firma \u00d7 rok) \u00b7 "
              "Kontroly: Size, Leverage, ROA, Asset Growth, lagged \u03c3.",
              size=10, italic=True, color=FIS_DARK)


# ======================================================================
# 9 \u2014 Results H1+H2 with native horizon bar chart
# ======================================================================
s = new_slide("V\u00fdsledky H1 + H2 \u2014 koeficienty nap\u0159\u00ed\u010d horizonty", 9)

kpi_top = Cm(4.7); kpi_h = Cm(2.0); kpi_w = Cm(7.1); kpi_gap = Cm(0.4)
left0 = (SW - (kpi_w * 3 + kpi_gap * 2)) // 2
add_kpi(s, left0,                       kpi_top, kpi_w, kpi_h,
        "\u03b2\u2097\u2091\u2099 = +0,121", "t = +7,64   \u00b7   H1 podpo\u0159eno",
        value_color=FIS_GREEN, value_size=18)
add_kpi(s, left0 + (kpi_w + kpi_gap),   kpi_top, kpi_w, kpi_h,
        "\u03b2\u1d48\u2091\u2099\u209b = \u22123,88", "t = \u22122,68   \u00b7   H2 podpo\u0159eno",
        value_color=RED, value_size=18)
add_kpi(s, left0 + 2*(kpi_w + kpi_gap), kpi_top, kpi_w, kpi_h,
        "n = 5 691", "firma-let   \u00b7   h = 30 d",
        value_color=FIS_DARK, value_size=18)

chart_data = CategoryChartData()
chart_data.categories = ["5 d", "10 d", "30 d", "90 d", "180 d", "365 d"]
chart_data.add_series("ln(Words\u2081\u2090)  (\u03b2\u2097\u2091\u2099)",
                      (0.142, 0.132, 0.121, 0.094, 0.078, 0.061))
chart_data.add_series("LitDensity  (\u03b2\u2097\u1d62\u209c, /10)",
                      (-0.52, -0.61, -0.62, -0.55, -0.40, -0.28))
chart_data.add_series("UncDensity  (\u03b2\u1d64\u2099\u1d9c, /10)",
                      (-0.10, -0.11, -0.13, -0.09, -0.06, -0.04))

chart_w = Cm(22.5); chart_h = Cm(6.8)
chart_x = Cm(1.2); chart_y = Cm(7.2)
chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                           chart_x, chart_y, chart_w, chart_h,
                           chart_data).chart
style_chart(chart, series_colors=[FIS_GREEN, RED, FIS_DARK],
            show_legend=True, value_format='0.00')
chart.value_axis.maximum_scale = 0.25
chart.value_axis.minimum_scale = -0.75
chart.value_axis.has_major_gridlines = True

add_text_only(s, Cm(1.2), Cm(14.3), Cm(22.5), Cm(0.7),
              "Pozn.: LitDensity a UncDensity \u0161k\u00e1lov\u00e1ny /10 pro vizu\u00e1ln\u00ed srovn\u00e1n\u00ed. "
              "Oba kan\u00e1ly identifikov\u00e1ny sou\u010dasn\u011b v jedn\u00e9 regresi.",
              size=10, italic=True, color=RGBColor(0x66, 0x66, 0x66))


# ======================================================================
# 10 \u2014 LM Decomposition
# ======================================================================
s = new_slide("Dekompozice LM slovn\u00edku (druh\u00fd stupe\u0148 H2)", 10)

chart_data = CategoryChartData()
chart_data.categories = ["LM-Litigious  (\u03b2\u2097\u1d62\u209c)", "LM-Uncertainty  (\u03b2\u1d64\u2099\u1d9c)"]
chart_data.add_series("\u03b2  (h = 30 d)", (-6.16, -1.28))

chart = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,
                           Cm(1.2), Cm(4.8), Cm(13.5), Cm(8.5),
                           chart_data).chart
style_chart(chart, series_colors=[FIS_GREEN], show_legend=False,
            value_format='0.0')
chart.value_axis.minimum_scale = -8
chart.value_axis.maximum_scale = 1
plot = chart.plots[0]
plot.has_data_labels = True
dl = plot.data_labels
dl.font.size = Pt(12); dl.font.bold = True
dl.font.color.rgb = FIS_DARK
dl.number_format = '0.00'
dl.position = XL_LABEL_POSITION.OUTSIDE_END
ser = plot.series[0]
from lxml import etree
dpt_xml = (
    '<c:dPt xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    '<c:idx val="1"/><c:invertIfNegative val="0"/><c:bubble3D val="0"/>'
    '<c:spPr><a:solidFill><a:srgbClr val="4A4A49"/></a:solidFill>'
    '<a:ln><a:noFill/></a:ln></c:spPr></c:dPt>')
dpt = etree.fromstring(dpt_xml)
cat_el = ser._element.find(qn('c:cat'))
if cat_el is not None:
    ser._element.insert(list(ser._element).index(cat_el), dpt)
else:
    ser._element.append(dpt)

right_x = Cm(15.4); right_w = Cm(9.0)
card_h = Cm(4.0); card_gap = Cm(0.5)
top1 = Cm(4.8); top2 = top1 + card_h + card_gap

box, _ = header_card(s, right_x, top1, right_w, card_h,
                     "LM-Litigious", FIS_GREEN, hdr_h=Cm(0.9))
tf = box.text_frame; tf.margin_top = Cm(1.1); tf.margin_left = Cm(0.4)
add_para(tf, "t = \u22123,38   \u00b7   p < 0,001", size=11, bold=True,
         color=FIS_DARK, first=True)
add_para(tf, "Boilerplate (safe-harbour formulace) nese "
             "cel\u00fd negativn\u00ed efekt specificity.",
         size=11, color=TEXT_DARK, space_before=4)

box, _ = header_card(s, right_x, top2, right_w, card_h,
                     "LM-Uncertainty", FIS_DARK, hdr_h=Cm(0.9))
tf = box.text_frame; tf.margin_top = Cm(1.1); tf.margin_left = Cm(0.4)
add_para(tf, "t = \u22121,01   \u00b7   p \u2248 0,31", size=11, bold=True,
         color=FIS_DARK, first=True)
add_para(tf, "Hedging nezn\u00e1m\u00e9ho nem\u00e1 samostatnou s\u00edlu po "
             "kontrole d\u00e9lky a litigious wordingu.",
         size=11, color=TEXT_DARK, space_before=4)

add_callout(s, Cm(1.2), Cm(13.7), Cm(22.5), Cm(1.7),
            "Intuice: trh ignoruje pr\u00e1vn\u00ed boilerplate (opakuj\u00edc\u00ed se \"may be subject to "
            "litigation\"), ale reaguje na firm-specific obsah zachycen\u00fd d\u00e9lkou.\n"
            "Wald\u016fv test \u03b2\u1d64\u2099\u1d9c = \u03b2\u2097\u1d62\u209c:  z = +1,91 (p = 0,056).",
            size=11)


# ======================================================================
# 11 \u2014 NEW: Construction of Omission\u2081\u2090
# ======================================================================
s = new_slide("Konstrukce Omission\u2081\u2090 \u2014 peer-relativn\u00ed opomenut\u00ed", 11)

add_text_only(s, Cm(1.2), Cm(4.7), Cm(22.5), Cm(1.0),
              "Co firma NE\u0158EKNE relativn\u011b k peer\u016fm m\u016f\u017ee informovat o budouc\u00ed volatilit\u011b.",
              size=12, bold=True, color=FIS_DARK)

step_w = Cm(7.2); step_h = Cm(5.5); step_gap = Cm(0.3)
step_top = Cm(6.0)
step_left0 = (SW - (step_w * 3 + step_gap * 2)) // 2

steps = [
    ("Krok 1: Peer-cell", FIS_GREEN,
     "Pro ka\u017ed\u00fd rok definujeme peer skupinu = "
     "firmy ve stejn\u00e9m SIC2 odv\u011btv\u00ed.\n\n"
     "Min. 5 peer\u016f v bu\u0148ce;\nmedi\u00e1n 35 firem.\n\n"
     "Leave-one-out: focal firma vylou\u010dena z vlastn\u00edch peer\u016f."),
    ("Krok 2: TF\u2013IDF vektorizace", FIS_DARK,
     "Ka\u017ed\u00fd Item 1A \u2192 TF\u2013IDF vektor.\n\n"
     "TF = frekvence termu v dokumentu\n"
     "IDF = log(N / n_docs s termem)\n\n"
     "Filtry: term v \u22655 a \u226495 % filing\u016f;\nL2 normalizace."),
    ("Krok 3: Omission gap", FIS_GREEN,
     "Omission\u2081\u2090 = 1 \u2212 (pokryt\u00e1 TF\u2013IDF "
     "v\u00e1ha peer\u016f / celkov\u00e1 TF\u2013IDF v\u00e1ha peer\u016f)\n\n"
     "\u2248 0 \u2192 firma pokr\u00fdv\u00e1 v\u0161e, co p\u00ed\u0161ou peers\n"
     "\u2248 1 \u2192 firma ignoruje v\u011bt\u0161inu peer-t\u00e9mat"),
]

for i, (title, color, body) in enumerate(steps):
    x = step_left0 + i * (step_w + step_gap)
    box, _ = header_card(s, x, step_top, step_w, step_h, title, color,
                         hdr_h=Cm(1.0))
    tf = box.text_frame
    tf.margin_top = Cm(1.2); tf.margin_left = Cm(0.4); tf.margin_right = Cm(0.3)
    add_para(tf, body, size=11, color=TEXT_DARK, first=True)
    if i < 2:
        ax = x + step_w
        ay = step_top + step_h // 2
        add_arrow(s, ax, ay, ax + step_gap, ay, color=FIS_DARK, weight=2.0)

add_callout(s, Cm(1.2), Cm(12.0), Cm(22.5), Cm(1.6),
            "P\u0159\u00edklad: peer-cell p\u00ed\u0161e 50\u00d7 \u201ecybersecurity, ransomware\u201c \u2014 "
            "focal firma to nezmn\u00ed \u2192 vysok\u00e9 Omission\u2081\u2090 \u2192 vy\u0161\u0161\u00ed budouc\u00ed \u03c3.",
            size=11)

add_text_only(s, Cm(1.2), Cm(13.9), Cm(22.5), Cm(0.7),
              "Korelace Omission\u2081\u2090 vs. ln(Words\u2081\u2090) = \u22120,85 \u2192 nutn\u00e1 kontrola d\u00e9lky v regresi.",
              size=10, italic=True, color=FIS_DARK)


# ======================================================================
# 12 \u2014 H3 results: cross-section vs within-firm
# ======================================================================
s = new_slide("Strategick\u00e9 ml\u010den\u00ed (H3) \u2014 v\u00fdsledky", 12)

card_w = Cm(11.0); card_gap = Cm(0.5)
cx1 = Cm(1.2); cx2 = cx1 + card_w + card_gap
card_top = Cm(4.8); card_h = Cm(4.5)

box, _ = header_card(s, cx1, card_top, card_w, card_h,
                     "Odv\u011btv\u00ed + rok FE (cross-section)", FIS_DARK, hdr_h=Cm(1.0))
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
add_equation(tf, "\u03b2\u2092\u2098 \u2248 0", size=28, bold=True, color=FIS_DARK,
             align=PP_ALIGN.CENTER, first=True)
add_para(tf, "|t| < 1,1 na v\u0161ech horizontech", size=11,
         color=FIS_DARK, align=PP_ALIGN.CENTER, space_before=4)
add_para(tf, "Stabiln\u00ed mezifirmn\u00ed rozd\u00edly v pokryt\u00ed \u2192 "
             "nenesou informaci o \u03c3", size=11, italic=True,
         color=TEXT_DARK, align=PP_ALIGN.CENTER, space_before=4)

box, _ = header_card(s, cx2, card_top, card_w, card_h,
                     "Firma + rok FE (within-firm)", FIS_GREEN, hdr_h=Cm(1.0))
tf = box.text_frame; tf.margin_top = Cm(1.3); tf.margin_left = Cm(0.5)
add_equation(tf, "\u03b2\u2092\u2098 = +1,345", size=28, bold=True, color=FIS_GREEN,
             align=PP_ALIGN.CENTER, first=True)
add_para(tf, "t = +2,93   \u00b7   p = 0,003   (h = 30 d)", size=11,
         color=FIS_DARK, align=PP_ALIGN.CENTER, space_before=4)
add_para(tf, "Meziro\u010dn\u00ed ZM\u011aNA v opomenut\u00ed \u2192 informace o \u03c3", size=11,
         italic=True, color=TEXT_DARK, align=PP_ALIGN.CENTER, space_before=4)

expl_top = card_top + card_h + Cm(0.5)
add_rect(s, Cm(1.2), expl_top, Cm(22.5), Cm(2.8), fill=GREY_BG, line=GREY_LINE)
add_text_only(s, Cm(1.6), expl_top + Cm(0.3), Cm(21.8), Cm(2.4),
              "Pro\u010d firma FE m\u011bn\u00ed v\u00fdsledek?\n"
              "\u2022 Mezi firmami: velk\u00e9 stabiln\u00ed rozd\u00edly v \u0161\u00ed\u0159i rizikov\u00e9ho profilu "
              "(IT firma vs. utilita) \u2014 trvale r\u016fzn\u00e9 Omission \u2192 \u03b2 \u2248 0.\n"
              "\u2022 Within-firm: co se zm\u011bn\u00ed rok-od-roku v tom, co firma vynech\u00e1 "
              "relativn\u011b k vlastn\u00edmu pr\u016fm\u011bru \u2014 TAM je informace o budouc\u00ed \u03c3.",
              size=11, color=TEXT_DARK)

add_callout(s, Cm(1.2), Cm(13.2), Cm(22.5), Cm(1.4),
            "+1 within-firm SD (0,054) \u2248 +7,3 % na 30denn\u00ed volatilit\u011b.   "
            "n = 4 630 firma-let.",
            size=11)


# ======================================================================
# 13 \u2014 Robustness
# ======================================================================
s = new_slide("Robustnost \u2014 t-statistiky nap\u0159\u00ed\u010d horizonty", 13)

chart_data = CategoryChartData()
chart_data.categories = ["5 d", "10 d", "30 d", "90 d", "180 d", "365 d"]
chart_data.add_series("ln(Words\u2081\u2090)", (8.30, 8.10, 7.64, 6.20, 5.10, 4.05))
chart_data.add_series("LitDensity",   (-1.95, -2.88, -3.38, -3.72, -4.02, -3.10))
chart_data.add_series("UncDensity",   (-0.40, -0.55, -1.01, -1.10, -0.95, -0.70))

chart = s.shapes.add_chart(XL_CHART_TYPE.LINE,
                           Cm(1.2), Cm(4.7), Cm(14.5), Cm(8.5),
                           chart_data).chart
style_chart(chart, series_colors=[FIS_GREEN, RED, FIS_DARK],
            show_legend=True, value_format='0.0')
for ser, color in zip(chart.plots[0].series, [FIS_GREEN, RED, FIS_DARK]):
    ln = ser.format.line
    ln.color.rgb = color; ln.width = Pt(2.5)
chart.value_axis.minimum_scale = -5
chart.value_axis.maximum_scale = 9
chart.value_axis.has_major_gridlines = True

right_x = Cm(16.2); right_w = Cm(7.8)
tile_h = Cm(2.5); gap = Cm(0.3); top0 = Cm(4.7)

def rob_tile(top, title, body, color):
    box = add_rect(s, right_x, top, right_w, tile_h, fill=WHITE, line=GREY_LINE)
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, right_x, top, Cm(0.18), tile_h)
    stripe.fill.solid(); stripe.fill.fore_color.rgb = color
    stripe.line.fill.background(); stripe.shadow.inherit = False
    tf = box.text_frame
    tf.margin_left = Cm(0.4); tf.margin_top = Cm(0.2); tf.margin_right = Cm(0.2)
    add_para(tf, title, size=11, bold=True, color=color, first=True)
    add_para(tf, body, size=10, color=TEXT_DARK, space_before=2)

rob_tile(top0,                              "Pre-COVID vs. COVID",
         "Stejn\u00fd vzorec v obou podvzorc\u00edch (2010\u20132019 vs. 2020\u20132024).", FIS_GREEN)
rob_tile(top0 + tile_h + gap,               "Firemn\u00ed FE",
         "\u03b2\u2097\u2091\u2099 p\u0159e\u017e\u00edv\u00e1 (+0,084; t = +3,31); sub-density absorbov\u00e1ny "
         "(boilerplate = trval\u00fd firemn\u00ed rys).",
         FIS_DARK)
rob_tile(top0 + 2*(tile_h + gap),           "H3 placebo",
         "N\u00e1hodn\u00e9 p\u0159i\u0159azen\u00ed do SIC2 bun\u011bk \u2192 |t| \u2208 [0,54; 0,90] "
         "vs. re\u00e1ln\u00e9 t = +2,93. Efekt je peer-specifick\u00fd.", FIS_GREEN)
rob_tile(top0 + 3*(tile_h + gap),           "\u226520 peer\u016f",
         "Velk\u00e9 bu\u0148ky: \u03b2\u2092\u2098 = +2,52 (t = +3,09). "
         "Sign\u00e1l siln\u011bj\u0161\u00ed s p\u0159esn\u011bj\u0161\u00edm peer-agreg\u00e1tem.",
         FIS_DARK)

add_text_only(s, Cm(1.2), Cm(13.5), Cm(14.5), Cm(0.7),
              "Pozn.: Vodorovn\u00e9 p\u00e1smo |t| = 2 odpov\u00edd\u00e1 hladin\u011b 5 %.",
              size=10, italic=True, color=RGBColor(0x66, 0x66, 0x66))


# ======================================================================
# 14 \u2014 Conclusion
# ======================================================================
s = new_slide("Z\u00e1v\u011br a p\u0159\u00ednos", 14)

kpi_top = Cm(4.7); kpi_h = Cm(4.0); kpi_w = Cm(7.3); kpi_gap = Cm(0.35)
left0 = (SW - (kpi_w * 3 + kpi_gap * 2)) // 2

def takeaway(left, num, title, body, color=FIS_GREEN):
    box = add_rect(s, left, kpi_top, kpi_w, kpi_h, fill=WHITE, line=GREY_LINE)
    circ = s.shapes.add_shape(MSO_SHAPE.OVAL,
                              left + Cm(0.4), kpi_top + Cm(0.4),
                              Cm(1.0), Cm(1.0))
    circ.fill.solid(); circ.fill.fore_color.rgb = color
    circ.line.fill.background(); circ.shadow.inherit = False
    ctf = circ.text_frame; ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
    ctf.margin_left = 0; ctf.margin_right = 0
    add_para(ctf, str(num), size=14, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, first=True)
    tb = s.shapes.add_textbox(left + Cm(1.7), kpi_top + Cm(0.4),
                              kpi_w - Cm(2.1), kpi_h - Cm(0.8))
    tf = tb.text_frame; tf.word_wrap = True
    add_para(tf, title, size=12, bold=True, color=color, first=True)
    add_para(tf, body, size=11, color=TEXT_DARK, space_before=4)

takeaway(left0,                      1, "Objem \u2260 specifi\u010dnost",
         "Oba kan\u00e1ly jsou sou\u010dasn\u011b identifikovateln\u00e9 v jedn\u00e9 regresi. "
         "D\u00e9lka = riziko; boilerplate = \u0161um.")
takeaway(left0 + (kpi_w + kpi_gap),   2, "Boilerplate, ne hedging",
         "Negativn\u00ed efekt dr\u017e\u00ed LM-Litigious slovn\u00edk \u2014 ne obecn\u00e9 v\u00fdrazy nejistoty.",
         color=FIS_DARK)
takeaway(left0 + 2*(kpi_w + kpi_gap), 3, "Ml\u010den\u00ed v\u016f\u010di peer\u016fm",
         "Co firma ne\u0159ekne relativn\u011b k peer\u016fm informuje o budouc\u00ed \u03c3 (within-firm).")

add_callout(s, Cm(1.2), Cm(9.2), Cm(22.5), Cm(1.8),
            "Limity: predik\u010dn\u00ed (nikoli kauz\u00e1ln\u00ed) \u010dten\u00ed  \u00b7  vzorek velk\u00fdch americk\u00fdch firem  "
            "\u00b7  H3 z\u00e1vis\u00ed na volb\u011b peer-benchmarku (SIC2) a v decilech sl\u00e1bne.",
            fill=ORANGE_LIGHT, accent=ORANGE)

add_callout(s, Cm(1.2), Cm(11.4), Cm(22.5), Cm(1.8),
            "P\u0159\u00ednos: replikovateln\u00e1 pipeline (SEC EDGAR XBRL \u00b7 HTML parser \u00b7 LM dekompozice \u00b7 "
            "peer-omission konstrukt) \u2014 modul\u00e1rn\u00ed k\u00f3d pro dal\u0161\u00ed v\u00fdzkum.")


# ======================================================================
# 15 \u2014 Zdroje (citace pou\u017eit\u00e9 v prezentaci)
# ======================================================================
s = new_slide("Zdroje", 15)

refs = [
    ("Campbell, J. L., Chen, H., Dhaliwal, D. S., Lu, H., & Steele, L. B. (2014).",
     "The information content of mandatory risk factor disclosures in corporate filings. "
     "Review of Accounting Studies, 19(1), 396\u2013455."),
    ("Hope, O.-K., Hu, D., & Lu, H. (2016).",
     "The benefits of specific risk-factor disclosures. "
     "Review of Accounting Studies, 21(4), 1005\u20131045."),
    ("Loughran, T., & McDonald, B. (2011).",
     "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. "
     "The Journal of Finance, 66(1), 35\u201365."),
    ("Gilardi, F., Alizadeh, M., & Kubli, M. (2023).",
     "ChatGPT outperforms crowd workers for text-annotation tasks. "
     "PNAS, 120(30), e2305016120."),
]

entry_top = Cm(4.8)
entry_h = Cm(2.0)
entry_gap = Cm(0.3)
for i, (head, body) in enumerate(refs):
    y = entry_top + i * (entry_h + entry_gap)
    box = add_rect(s, Cm(1.2), y, Cm(22.5), entry_h, fill=WHITE, line=GREY_LINE)
    stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(1.2), y, Cm(0.18), entry_h)
    stripe.fill.solid(); stripe.fill.fore_color.rgb = FIS_GREEN
    stripe.line.fill.background(); stripe.shadow.inherit = False
    tf = box.text_frame
    tf.margin_left = Cm(0.5); tf.margin_top = Cm(0.25); tf.margin_right = Cm(0.4)
    add_para(tf, head, size=12, bold=True, color=FIS_DARK, first=True)
    add_para(tf, body, size=11, color=TEXT_DARK, space_before=2)

add_text_only(s, Cm(1.2), Cm(15.2), Cm(22.5), Cm(0.7),
              "Pln\u00fd seznam literatury v bakal\u00e1\u0159sk\u00e9 pr\u00e1ci (tex/bibliography.bib).",
              size=10, italic=True, color=RGBColor(0x88, 0x88, 0x88))


# ======================================================================
# 16 \u2014 Thank you
# ======================================================================
s = prs.slides.add_slide(SECTION)
for ph in s.placeholders:
    idx = ph.placeholder_format.idx
    if idx == 0:
        ph.text_frame.clear()
        p = ph.text_frame.paragraphs[0]
        r = p.add_run(); r.text = "D\u011bkuji za pozornost"
        r.font.name = FONT; r.font.size = Pt(40); r.font.bold = True
        r.font.color.rgb = FIS_GREEN
    elif idx == 1:
        ph.text_frame.clear()
        add_para(ph.text_frame, "Dotazy a diskuse", size=20,
                 color=FIS_DARK, first=True)
        add_para(ph.text_frame, "\u0160imon Slansk\u00fd \u00b7 V\u0160E FIS \u00b7 2026",
                 size=12, color=RGBColor(0x88, 0x88, 0x88), space_before=10)

prs.save(OUT)
print(f"Saved: {OUT}")
