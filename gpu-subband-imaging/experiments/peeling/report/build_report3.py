"""Peeling experiments report v3: v2 + speed-ups of the CPU-exact model, TTCalX on CPU, math/code map."""
import json
import os
import re

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "peeling_experiments_report.pdf")
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# Arial (TTF) has the Greek and accented glyphs used in the text; the built-in Helvetica does not.
_F = "/System/Library/Fonts/Supplemental/"
for name, f in (("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf"), ("Arial-Italic", "Arial Italic.ttf"),
                ("Arial-BoldItalic", "Arial Bold Italic.ttf")):
    pdfmetrics.registerFont(TTFont(name, _F + f))
for b, i, name in ((0, 0, "Arial"), (1, 0, "Arial-Bold"), (0, 1, "Arial-Italic"), (1, 1, "Arial-BoldItalic")):
    addMapping("Arial", b, i, name)
ss = getSampleStyleSheet()
for st in ss.byName.values():
    st.fontName = "Arial-Bold" if "Heading" in st.name or st.name == "Title" else "Arial"
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9, leading=12)
B = ParagraphStyle("b", parent=P, leftIndent=12, bulletIndent=2)
SMALL = ParagraphStyle("s", parent=P, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("c", parent=P, fontSize=7.3, leading=8.8)
N = json.load(open(os.path.join(HERE, "notes3.json")))
SP = json.load(open(os.path.join(HERE, "speed_timing.json")))


def para(t, st=P):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, B, bulletText="•") for t in items]


def table(rows, widths=None):
    t = Table([[Paragraph(str(c), CELL) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def fig(path, width=26 * cm, maxh=15.5 * cm, caption=None):
    w, h = ImageReader(path).getSize()
    height = width * h / w
    if height > maxh:
        width, height = width * maxh / height, maxh
    items = [Image(path, width=width, height=height)]
    if caption:
        items.append(para(caption, SMALL))
    return KeepTogether(items)


def figs(story, items):
    for path, cap in items:
        full = os.path.join(HERE, path)
        if not os.path.exists(full):
            story.append(para(f"[missing figure {path}]", SMALL))
            continue
        if path.startswith("charts/"):
            story.append(fig(full, width=25 * cm, maxh=8.2 * cm, caption=cap))
            story.append(Spacer(1, 8))
        else:
            story.append(PageBreak())
            story.append(fig(full, caption=cap))
    story.append(PageBreak())


def fmt(x):
    return "–" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def agg_table(rows, names, cols):
    out = [["variant"] + [c for _, c in cols]]
    for v, lab in names:
        r = rows.get(v)
        if r:
            out.append([lab] + [fmt(r.get(k)) for k, _ in cols])
    return out


def divergence_table(summaries, variants):
    out = [["variant", "ref", "snapshots", "diverged (removed > 105% of data power)",
            "rms > 1.3x ref", "worst rms / ref", "where"]]
    for v, ref, lab in variants:
        n = div = bad = 0
        worst, where = 0.0, []
        for date, d in summaries.items():
            for s, r in d.items():
                if v not in r or ref not in r:
                    continue
                n += 1
                ratio = r[v]["rms"] / r[ref]["rms"]
                worst = max(worst, ratio)
                if r[v].get("removed", 0) > 1.05:
                    div += 1
                    where.append(f"{date[-4:]}/{s}")
                if ratio > 1.3:
                    bad += 1
        if n:
            out.append([lab, ref, n, div, bad, f"{worst:.2f}", ", ".join(where[:6]) + (" …" if len(where) > 6 else "")])
    return out


def main():
    T = json.load(open(os.path.join(HERE, "tables_final.json")))
    S = {"2024-12-20": json.load(open(os.path.join(HERE, "full_1220.json"))),
         "2024-12-27": json.load(open(os.path.join(HERE, "full_1227.json")))}
    story = [para(N["title"], H1), para(N["subtitle"], SMALL), Spacer(1, 4),
             para("Answers in short", H2)]
    story += bullets(N["answers"])
    story.append(para("Setup", H2))
    story += bullets(N["setup"])
    story.append(PageBreak())

    story.append(para("1. Bugs and discrepancies found (TTCalX GPU vs CPU TTCal, daemon, main pipeline)", H2))
    story.append(table(N["bugs"], widths=[5.2 * cm, 9.5 * cm, 3.2 * cm, 8.6 * cm]))
    story.append(para(N["bugs_note"], SMALL))
    story.append(PageBreak())

    story.append(para("2. Sky peeling: CPU main pipeline vs GPU, 2 dates × 16 subbands × 3 hours", H2))
    story += bullets(N["sky"])
    cols = [("n", "snapshots"), ("noop", "nothing peeled"), ("rms_ratio", "median rms / CPU"),
            ("corr", "removed corr. w/ CPU"), ("CasA_ratio", "Cas A resid. / CPU"),
            ("CygA_ratio", "Cyg A resid. / CPU"), ("VirA_ratio", "Vir A resid. / CPU")]
    names = [("gpu_prod", "GPU production (deployed settings)"),
             ("gpu_prodfix", "production + no gate + elev. 0° + model flags"),
             ("gpu_prodfix_guard", "  … + divergence guard"),
             ("gpu_cpumatch", "GPU CPU-matched (rk4, maxiter 5, tol 1e-4, minuvw 5), kernel model, NO model-flag fix"),
             ("gpu_cm_exact", "GPU CPU-matched + model flags + CPU-exact model (ZEST_GENVIS=exact)")]
    for key, lab in (("sky_all", "all 96 snapshots"), ("sky_low", "13–46 MHz"), ("sky_high", "50–82 MHz")):
        story.append(para(f"Medians over {lab} (residual ratios where CPU reduced the source > 2×):", SMALL))
        story.append(table(agg_table(T[key], names, cols), widths=[8.4 * cm] + [2.55 * cm] * len(cols)))
        story.append(Spacer(1, 4))
    story.append(para("Robustness (medians hide outliers): diverged = peel added power instead of removing it.", SMALL))
    story.append(table(divergence_table(S, [
        ("gpu_prod", "cpu_sky", "GPU production"), ("gpu_prodfix", "cpu_sky", "production + 3 fixes"),
        ("gpu_prodfix_guard", "cpu_sky", "production + 3 fixes + guard"),
        ("gpu_cpumatch", "cpu_sky", "CPU-matched, no flag fix"), ("gpu_cm_exact", "cpu_sky", "CPU-matched + flags + exact model")]),
        widths=[5.4 * cm, 2 * cm, 2 * cm, 3.6 * cm, 2.4 * cm, 2.2 * cm, 9 * cm]))
    story.append(PageBreak())
    figs(story, N["sky_figs"])

    story.append(para("3. Sky + RFI peeling on the GPU vs the main pipeline (sky zest, then RFI zest)", H2))
    story += bullets(N["rfi"])
    cols = [("n", "snapshots"), ("rms_ratio", "median rms / CPU"), ("corr", "removed corr. w/ CPU"),
            ("RFI_ratio", "RFI resid. / CPU"), ("CasA_ratio", "Cas A resid. / CPU"), ("CygA_ratio", "Cyg A resid. / CPU")]
    names = [("gpu_cm_after", "GPU sky, then RFI (main-pipeline scheme)"),
             ("gpu_cm_after_guard", "  … + guard"),
             ("gpu_cm_joint", "GPU sky + RFI jointly"), ("gpu_cm_joint_guard", "  … + guard"),
             ("gpu_cm_exact_joint", "GPU joint, exact model"), ("gpu_cm_jreg1", "GPU joint + Tukey prior 1°"),
             ("gpu_prodfix_rfi", "production solver (damped) + fixes, joint RFI")]
    for key, lab in (("rfi_low", "13–46 MHz"), ("rfi_high", "50–82 MHz")):
        story.append(para(f"{lab}, both dates:", SMALL))
        story.append(table(agg_table(T[key], names, cols), widths=[8.4 * cm] + [2.9 * cm] * len(cols)))
        story.append(Spacer(1, 4))
    story.append(table(divergence_table(S, [
        ("gpu_cm_after", "cpu_sky_rfi", "GPU sky, then RFI"), ("gpu_cm_after_guard", "cpu_sky_rfi", "  … + guard"),
        ("gpu_cm_joint", "cpu_sky_rfi", "GPU joint"), ("gpu_cm_joint_guard", "cpu_sky_rfi", "  … + guard"),
        ("gpu_cm_jreg0.25", "cpu_sky_rfi", "GPU joint + prior 0.25°"), ("gpu_cm_jreg1", "cpu_sky_rfi", "GPU joint + prior 1°"),
        ("gpu_cm_jreg3", "cpu_sky_rfi", "GPU joint + prior 3°"), ("gpu_prodfix_rfi", "cpu_sky_rfi", "production solver, joint RFI")]),
        widths=[5.4 * cm, 2 * cm, 2 * cm, 3.6 * cm, 2.4 * cm, 2.2 * cm, 9 * cm]))
    story.append(PageBreak())
    figs(story, N["rfi_figs"])

    story.append(para("4. Speed: CPU TTCal vs GPU TTCalX, sky + RFI", H2))
    story += bullets(N["speed"])
    bj = json.load(open(os.path.join(HERE, "bench.json")))
    rows = [["mode", "per snapshot (median)", "range", "8 snapshots wall-clock incl. start-up"]]
    cpu = [a + b for a, b in bj["cpu"]]
    rows.append(["CPU TTCal, main pipeline (sky ~78 s + RFI ~52 s; new Julia process per MS)",
                 f"{np.median(cpu):.1f} s", f"{min(cpu):.0f}–{max(cpu):.0f} s", f"{sum(cpu):.0f} s"])
    labs = {"prod_sky": "GPU production, sky only", "prodfix_rfi_joint": "GPU production solver + fixes, joint RFI",
            "cm_after": "GPU CPU-matched, sky then RFI", "cm_joint": "GPU CPU-matched, joint RFI",
            "cm_exact_joint": "GPU CPU-matched, joint RFI, exact model"}
    for k, lab in labs.items():
        if k in bj["gpu"]:
            g = bj["gpu"][k]
            rows.append([lab, f"{np.median(g):.1f} s", f"{min(g):.1f}–{max(g):.1f} s", f"{bj['walls'][k]:.0f} s"])
    story.append(table(rows, widths=[11 * cm, 4 * cm, 4 * cm, 6 * cm]))
    story.append(Spacer(1, 6))
    figs(story, [("charts/bench.png", "Median time per snapshot (log scale).")])

    story.append(para("5. Speeding up the CPU-exact model, and TTCalX on the CPU", H2))
    story += bullets(N["speed2_intro"])
    story.append(Spacer(1, 4))
    rows = [["configuration (same CPU-matched sky+RFI zest unless noted)", "MS read", "peel", "MS write", "per snapshot", "output vs CPU-exact GPU"]]
    rows += SP["table"]
    story.append(table(rows, widths=[9.6 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm, 3.0 * cm, 5.4 * cm]))
    story.append(Spacer(1, 4))
    story += [para(t, SMALL) for t in N["speed2_notes"]]
    story.append(Spacer(1, 4))
    rows = [["snapshot", "removed signal: GPU optimized vs CPU TTCal (rel. diff / corr.)", "image rms CPU TTCal", "image rms GPU optimized",
             "image rms TTCalX CPU", "GPU optimized vs earlier exact run", "TTCalX CPU vs GPU optimized"]]
    rows += SP["equality"]
    story.append(table(rows, widths=[2.4 * cm, 6.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 4.2 * cm, 4.2 * cm]))
    story.append(para("Visibility-level comparison of what was removed (prep − output, unflagged cross-correlations, I = (XX+YY)/2) and image "
                      "rms from simple uniform-weighted dirty images. The GPU−CPU TTCal difference at 41/69 MHz is the RFI pass: "
                      "ttcal_dev uses the misread emitter position (section 1).", SMALL))
    story.append(PageBreak())
    figs(story, N["speed2_figs"])

    story.append(para("6. Regularized Peeling (Tukey phase prior): where does it help?", H2))
    story += bullets(N["reg"])
    figs(story, N["reg_figs"])

    story.append(para("7. Recommendations", H2))
    story += bullets(N["recommendations"])
    story.append(para("8. Where everything is", H2))
    story += bullets(N["where"])
    story.append(PageBreak())
    story.append(para("Appendix A: the math, and where it is in the code", H2))
    story.append(para("Line numbers refer to the local experimental branches (TTCalX exp/peeling-experiments @ a72f4e5: src/gpu/…; "
                      "distributed-pipeline exp/gpu-peeling-experiments: gpu-subband-imaging/workers/zest_daemon.jl) and to the "
                      "main pipeline's CPU TTCal (julia060 build, src/…).", SMALL))
    for title, eqs, text, code in N["math"]:
        block = [para(f"<b>{title}</b>", P)]
        for e in eqs:
            path = os.path.join(HERE, "eq", f"{e}.png")
            w, h = ImageReader(path).getSize()
            width = min(24 * cm, w / 200 * 2.54 * cm * 0.95)
            block.append(Image(path, width=width, height=width * h / w))
        block.append(para(text, P))
        block.append(para("<font color='#1f5fa8'>Code:</font> " + code, SMALL))
        block.append(Spacer(1, 6))
        story.append(KeepTogether(block))
    story.append(PageBreak())
    story.append(para("Appendix B: first-round results (2024-12-20, 8 bands; before the model-flag fix)", H2))
    story += bullets(N["appendix"])
    figs(story, N["appendix_figs"])

    doc = SimpleDocTemplate(OUT, pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title="Peeling experiments: GPU vs CPU, RFI, Regularized Peeling, speed-ups (v3)")
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
