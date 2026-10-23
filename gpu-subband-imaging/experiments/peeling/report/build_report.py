"""Build the peeling-experiments PDF report from the study outputs in this directory."""
import json
import os
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "peeling_experiments_report.pdf")
PAGE = landscape(A4)

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=12)
B = ParagraphStyle("b", parent=P, leftIndent=12, bulletIndent=2)
SMALL = ParagraphStyle("s", parent=P, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("c", parent=P, fontSize=7.5, leading=9)


def para(t, st=P):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, B, bulletText="•") for t in items]


def table(rows, widths=None, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width=26 * cm, caption=None):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    height = width * h / w
    maxh = 15.5 * cm if width > 25 * cm else 8.2 * cm
    if height > maxh:
        width, height = width * maxh / height, maxh
    items = [Image(path, width=width, height=height)]
    if caption:
        items.append(para(caption, SMALL))
    return KeepTogether(items)


def add_figs(story, figs):
    """Charts (charts/*) flow two per page at reduced height; image mosaics get a page each."""
    for path, caption in figs:
        if path.startswith("charts/"):
            story.append(fig(os.path.join(HERE, path), width=24 * cm, caption=caption))
            story.append(Spacer(1, 10))
        else:
            story.append(PageBreak())
            story.append(fig(os.path.join(HERE, path), caption=caption))
    story.append(PageBreak())


def fmt(x, n=2):
    return "–" if x is None else f"{x:.{n}f}"


def agg_rows(rows, names, cols):
    out = [["variant"] + [c[1] for c in cols]]
    for v, label in names:
        r = rows.get(v)
        if not r:
            continue
        out.append([label] + [fmt(r.get(k)) if k not in ("n", "noop") else str(r.get(k, "–"))
                              for k, _ in cols])
    return out


def main():
    tables = json.load(open(os.path.join(HERE, "tables.json")))
    timing = {}
    if os.path.exists(os.path.join(HERE, "timing.txt")):
        for line in open(os.path.join(HERE, "timing.txt")):
            m = re.match(r"TIMING (\S+)\s+n=(\d+) mean=([\d.]+)s median=([\d.]+)s min=([\d.]+)s max=([\d.]+)s", line)
            if m:
                timing[m.group(1)] = [float(x) for x in m.groups()[1:]]
    ablation = json.load(open(os.path.join(HERE, "ablation.json"))) \
        if os.path.exists(os.path.join(HERE, "ablation.json")) else {}
    notes = json.load(open(os.path.join(HERE, "notes.json")))

    story = []
    story.append(para("GPU vs CPU peeling, GPU RFI peeling and Regularized Peeling — OVRO-LWA experiments", H1))
    story.append(para(notes["subtitle"], SMALL))
    story.append(Spacer(1, 6))
    story.append(para("Summary", H2))
    story += bullets(notes["summary"])
    story.append(para("Setup", H2))
    story += bullets(notes["setup"])
    story.append(PageBreak())

    # 1. model geometry
    story.append(para("1. Model geometry: CPU TTCal vs TTCalX (raw models, no gains)", H2))
    story += bullets(notes["geometry"])
    story.append(Spacer(1, 4))
    story.append(table(notes["geometry_table"], widths=[6.5 * cm, 4 * cm, 4 * cm, 4 * cm, 7 * cm]))
    story.append(PageBreak())

    # 2. consistency
    story.append(para("2. Sky peeling: CPU (main pipeline) vs GPU", H2))
    story += bullets(notes["sky"])
    cols = [("n", "snapshots"), ("noop", "nothing peeled"), ("rms_ratio", "rms / CPU"),
            ("corr", "removed-signal corr. with CPU"), ("CasA_ratio", "Cas A resid. / CPU"),
            ("CygA_ratio", "Cyg A resid. / CPU"), ("VirA_ratio", "Vir A resid. / CPU")]
    names = [("gpu_exp", "GPU production settings (exp code, = production images)"),
             ("gpu_geom", "GPU production settings + geometric UVW"),
             ("gpu_cpumatch", "GPU CPU-matched (rk4, maxiter 5, tol 1e-4, minuvw 5, el>0, no gate, geometric UVW)")]
    story.append(para("All 24 snapshots (medians over snapshots; residual ratios only where the CPU peel "
                      "reduced the source by >2x):", SMALL))
    story.append(table(agg_rows(tables["sky_all"], names, cols),
                       widths=[8 * cm] + [2.6 * cm] * len(cols)))
    if ablation:
        story.append(Spacer(1, 6))
        story.append(para("Ablation: one production setting changed at a time (8 snapshots)", H2))
        story.append(table(ablation["rows"], widths=ablation.get("widths")))
        story += bullets(notes.get("ablation", []))
    story.append(PageBreak())
    add_figs(story, notes["sky_figs"])

    # 3. RFI
    story.append(para("3. RFI peeling on the GPU", H2))
    story += bullets(notes["rfi"])
    names = [("gpu_cm_after", "GPU sky then RFI (main-pipeline scheme), correct emitter position"),
             ("gpu_cm_aftercpu", "GPU sky then RFI, ttcal_dev's (misread) emitter position"),
             ("gpu_cm_joint", "GPU sky + RFI zested jointly"),
             ("gpu_joint", "joint, production solver (damped, tol 1e-2, gate)"),
             ("gpu_geom_rfi", "after, production solver (damped, tol 1e-2, gate)")]
    cols = [("n", "snapshots"), ("noop", "nothing peeled"), ("rms_ratio", "rms / CPU"),
            ("corr", "removed corr. with CPU"), ("RFI_ratio", "RFI resid. / CPU"),
            ("CasA_ratio", "Cas A resid. / CPU"), ("CygA_ratio", "Cyg A resid. / CPU")]
    story.append(para("18–41 MHz, 18 snapshots, reference = CPU sky + RFI zest (main pipeline):", SMALL))
    story.append(table(agg_rows(tables["rfi_low"], names, cols),
                       widths=[8 * cm] + [2.6 * cm] * len(cols)))
    story.append(PageBreak())
    add_figs(story, notes["rfi_figs"])

    # 4. Regularized
    story.append(para("4. Regularized Peeling (Tukey phase prior)", H2))
    story += bullets(notes["reg"])
    names = [("gpu_cm_after", "after, no prior")] + \
            [(f"gpu_cm_areg{d}", f"after, prior on sky pass, max offset {d}°") for d in ("0.5", "1", "2", "5")] + \
            [("gpu_cm_joint", "joint, no prior")] + \
            [(f"gpu_cm_jreg{d}", f"joint, prior on all, max offset {d}° (5 low-band snapshots)") for d in ("0.5", "1", "2", "5")]
    story.append(table(agg_rows(tables["rfi_low"], names, cols),
                       widths=[8 * cm] + [2.6 * cm] * len(cols)))
    story.append(Spacer(1, 4))
    story.append(table(notes["gains_table"], widths=[5.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]))
    story.append(PageBreak())
    add_figs(story, notes["reg_figs"])

    # 5. timing
    story.append(para("5. GPU speed", H2))
    rows = [["mode", "MS", "mean s/MS", "median s/MS", "min", "max"]]
    for v, label in notes["timing_labels"]:
        if v in timing:
            n, mean, med, mn, mx = timing[v]
            rows.append([label, f"{int(n)}", f"{mean:.2f}", f"{med:.2f}", f"{mn:.2f}", f"{mx:.2f}"])
    story.append(table(rows, widths=[10 * cm, 1.5 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 2 * cm]))
    story += bullets(notes["timing"])
    story.append(Spacer(1, 6))
    story.append(fig(os.path.join(HERE, "charts/timing.png"), width=24 * cm,
                     caption="Median peel time per MS (first MS with Julia compilation excluded by the median)."))

    # 6. bugs + recommendations
    story.append(PageBreak())
    story.append(para("6. Bugs and issues found", H2))
    story.append(table(notes["bugs"], widths=[6 * cm, 12 * cm, 8.5 * cm]))
    story.append(para("7. Recommendations", H2))
    story += bullets(notes["recommendations"])
    story.append(para("8. Where everything is", H2))
    story += bullets(notes["where"])

    doc = SimpleDocTemplate(OUT, pagesize=PAGE, leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title="Peeling experiments: GPU vs CPU, RFI, Regularized Peeling")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
