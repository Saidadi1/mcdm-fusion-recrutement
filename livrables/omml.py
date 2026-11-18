"""Conversion LaTeX -> équation Word native (OMML) : LaTeX -> MathML (latex2mathml) -> OMML (XSLT d'Office).

* add_equation(doc, latex, number) : équation en mode « affichage » (oMathPara), centrée, numérotée à droite
  (tableau sans bordure à trois colonnes : marge | équation | numéro).
* add_inline_math(paragraph, latex) : formule dans le texte.
"""
from __future__ import annotations

import copy
import glob
import os
import re
from functools import lru_cache

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from latex2mathml.converter import convert
from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_XSL = (glob.glob(r"C:\Program Files*\Microsoft Office\**\MML2OMML.XSL", recursive=True)
        + [os.path.join(os.path.dirname(__file__), "MML2OMML.XSL")])


@lru_cache(maxsize=1)
def _transform():
    for path in _XSL:
        if os.path.exists(path):
            return etree.XSLT(etree.parse(path))
    raise FileNotFoundError("MML2OMML.XSL introuvable (fourni avec Microsoft Office)")


def _fix_nary(root, display: bool):
    """Limites des sommes sous/sur le symbole en mode affichage."""
    for nary in root.iter(f"{{{M}}}nary"):
        pr = nary.find(f"{{{M}}}naryPr")
        if pr is None:
            pr = etree.SubElement(nary, f"{{{M}}}naryPr"); nary.insert(0, pr)
        loc = pr.find(f"{{{M}}}limLoc")
        if loc is None:
            loc = etree.SubElement(pr, f"{{{M}}}limLoc")
        loc.set(f"{{{M}}}val", "undOvr" if display else "subSup")


_STOP = {"=", "+", "−", "-", ",", "<", ">", "≤", "≥", "≈", " ", " ", " "}


def _nary_arguments(root):
    """Range dans le corps du Σ les éléments qui le suivent, jusqu'au prochain opérateur."""
    for nary in list(root.iter(f"{{{M}}}nary")):
        e = nary.find(f"{{{M}}}e")
        if e is None or len(e):
            continue
        parent = nary.getparent()
        while True:
            nxt = nary.getnext()
            if nxt is None or nxt.tag == f"{{{M}}}nary":
                break
            if nxt.tag == f"{{{M}}}r":
                txt = "".join(nxt.itertext()).strip()
                if not txt or txt in _STOP or txt[0] in _STOP:
                    break
            parent.remove(nxt)
            e.append(nxt)


def _spaces(mathml: str) -> str:
    """Les espaces LaTeX (\\quad, \\;) deviennent du texte espacé (l'XSL d'Office les ignore)."""
    def rep(mo):
        w = float(mo.group(1))
        ch = " " * max(1, round(w)) if w >= 0.9 else " "
        return f"<mtext>{ch}</mtext>"
    return re.sub(r'<mspace[^>]*?width="([\d.]+)em"[^>]*/>', rep, mathml)


def _prep(latex: str) -> str:
    """Opérateurs nommés en romain (\\min, \\max…) : latex2mathml les met sinon en italique."""
    for op in ("min", "max", "arg", "exp", "log"):
        latex = re.sub(rf"\\{op}(?![a-zA-Z])", rf"\\mathrm{{{op}}}", latex)
    return latex


@lru_cache(maxsize=1024)
def _omml_xml(latex: str, display: bool) -> bytes:
    mathml = _spaces(convert(_prep(latex), display="block" if display else "inline"))
    root = _transform()(etree.fromstring(mathml.encode("utf-8"))).getroot()
    _fix_nary(root, display)
    _nary_arguments(root)
    return etree.tostring(root)


def omath(latex: str, display: bool = False):
    return copy.deepcopy(etree.fromstring(_omml_xml(latex, display)))


def add_inline_math(paragraph, latex: str):
    paragraph._p.append(omath(latex, display=False))


def _no_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); borders.append(e)
    tblPr.append(borders)


def _display_paragraph(p, latex):
    para = OxmlElement("m:oMathPara")
    para.append(omath(latex, display=True))
    p._p.append(para)


def add_equation(doc, latex: str, number: int | None = None, width_cm: float = 16.6):
    if number is None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _display_paragraph(p, latex)
        return p
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _no_borders(t)
    widths = (Cm(1.4), Cm(width_cm - 2.8), Cm(1.4))
    for cell, w in zip(t.rows[0].cells, widths):
        cell.width = w
        cell.vertical_alignment = 1  # centre
    mid = t.rows[0].cells[1].paragraphs[0]
    mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _display_paragraph(mid, latex)
    num = t.rows[0].cells[2].paragraphs[0]
    num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    num.add_run(f"({number})")
    for c in t.rows[0].cells:
        c.paragraphs[0].paragraph_format.space_after = Pt(2)
        c.paragraphs[0].paragraph_format.space_before = Pt(2)
    return t
