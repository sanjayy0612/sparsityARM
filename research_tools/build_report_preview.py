"""Build the non-canonical, layout-checked PDF preview from verified summary data."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "research/package/summary.json").read_text())
OUTPUT = ROOT / "paper/arm-sparse-report.pdf"
NAVY, BLUE, RED, GREEN, GREY = colors.HexColor("#172554"), colors.HexColor("#2563eb"), colors.HexColor("#e11d48"), colors.HexColor("#16a34a"), colors.HexColor("#64748b")


def font_setup():
    candidates = [
        ("Inter", "/System/Library/Fonts/SFNS.ttf"),
        ("Inter", "/System/Library/Fonts/Helvetica.ttc"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            pass
    return "Helvetica"


FONT = font_setup()
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontName=FONT, fontSize=25, leading=29, textColor=NAVY, spaceAfter=16))
styles.add(ParagraphStyle(name="Deck", fontName=FONT, fontSize=12, leading=17, textColor=GREY, spaceAfter=18))
styles.add(ParagraphStyle(name="H1x", fontName=FONT, fontSize=19, leading=23, textColor=NAVY, spaceAfter=12))
styles.add(ParagraphStyle(name="H2x", fontName=FONT, fontSize=12, leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name="Bodyx", fontName=FONT, fontSize=9.4, leading=13.5, textColor=colors.HexColor("#1f2937"), spaceAfter=7))
styles.add(ParagraphStyle(name="Smallx", fontName=FONT, fontSize=7.8, leading=10.5, textColor=GREY, spaceAfter=5))
styles.add(ParagraphStyle(name="Pull", fontName=FONT, fontSize=14, leading=19, textColor=NAVY, leftIndent=20, rightIndent=20, borderColor=colors.HexColor("#bfdbfe"), borderWidth=1, borderPadding=12, backColor=colors.HexColor("#eff6ff"), spaceBefore=10, spaceAfter=12))


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def table(data, widths):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def line_chart(kind):
    w, h = 470, 245
    d = Drawing(w, h)
    l, r, top, bottom = 52, 15, 28, 42
    pw, ph = w-l-r, h-top-bottom
    if kind == "quality":
        ymin, ymax, xmax, threshold = 0, 24, 30, 5
        title, ylab = "Quality cost of B8 sparsity", "relative perplexity increase (%)"
        series = {name: [(q["sparsity_pct"], q["relative_ppl_pct"]) for q in model["quality"]] for name, model in SUMMARY["models"].items()}
    else:
        ymin, ymax, xmax, threshold = -35, 55, 60, 0
        title, ylab = "B8 wall-clock speedup on Apple M2", "speedup vs Accelerate dense (%)"
        series = {name: [(q["sparsity_pct"], q["mean_speedup_pct"]) for q in model["runtime"] if q["sparsity_pct"] <= 60] for name, model in SUMMARY["models"].items()}
    x = lambda v: l + v/xmax*pw
    y = lambda v: bottom + (v-ymin)/(ymax-ymin)*ph
    d.add(String(l, h-15, title, fontName=FONT, fontSize=13, fillColor=NAVY))
    for i in range(6):
        val = ymin + (ymax-ymin)*i/5
        yy = y(val)
        d.add(Line(l, yy, w-r, yy, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=.5))
        d.add(String(l-6, yy-3, f"{val:.0f}", fontName=FONT, fontSize=7, textAnchor="end", fillColor=GREY))
    for val in range(0, xmax+1, 10):
        d.add(String(x(val), bottom-15, str(val), fontName=FONT, fontSize=7, textAnchor="middle", fillColor=GREY))
    d.add(Line(l, y(threshold), w-r, y(threshold), strokeColor=NAVY, strokeWidth=1.1, strokeDashArray=[4, 3]))
    palette = {"Llama 3.2 1B": BLUE, "TinyLlama 1.1B": RED}
    for index, (name, points) in enumerate(series.items()):
        pts = [(x(a), y(b)) for a, b in points]
        d.add(PolyLine(pts, strokeColor=palette[name], strokeWidth=2.2))
        for xx, yy in pts:
            d.add(Circle(xx, yy, 3.2, fillColor=palette[name], strokeColor=colors.white, strokeWidth=.7))
        lx = l + index*150
        d.add(Line(lx, 12, lx+18, 12, strokeColor=palette[name], strokeWidth=2.2))
        d.add(String(lx+23, 9, name, fontName=FONT, fontSize=7.5, fillColor=colors.HexColor("#1f2937")))
    d.add(String(8, bottom+ph/2, ylab, fontName=FONT, fontSize=7, fillColor=GREY, angle=90, textAnchor="middle"))
    return d


def frontier():
    w, h = 470, 190
    d = Drawing(w, h)
    l, r = 112, 20
    x = lambda v: l + v/60*(w-l-r)
    d.add(String(20, h-18, "No measured quality-speed intersection", fontName=FONT, fontSize=13, fillColor=NAVY))
    for tick in range(0, 61, 10):
        xx = x(tick)
        d.add(Line(xx, 35, xx, 150, strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=.5))
        d.add(String(xx, 22, f"{tick}%", fontName=FONT, fontSize=7, textAnchor="middle", fillColor=GREY))
    for idx, (name, model) in enumerate(SUMMARY["models"].items()):
        yy = 118-idx*58
        q, s = model["quality_ceiling_pct"], model["first_speedup_grid_pct"]
        d.add(String(l-8, yy-3, name, fontName=FONT, fontSize=8, textAnchor="end", fillColor=NAVY))
        d.add(Line(x(q), yy, x(s), yy, strokeColor=colors.HexColor("#94a3b8"), strokeWidth=4))
        d.add(Circle(x(q), yy, 6, fillColor=GREEN, strokeColor=colors.white))
        d.add(Circle(x(s), yy, 6, fillColor=RED, strokeColor=colors.white))
        d.add(String(x(q), yy+11, f"quality {q}%", fontName=FONT, fontSize=7, textAnchor="middle", fillColor=GREEN))
        d.add(String(x(s), yy+11, f"speed {s}%", fontName=FONT, fontSize=7, textAnchor="middle", fillColor=RED))
    return d


def page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#dbeafe")); canvas.line(0.65*inch, 0.52*inch, 7.85*inch, 0.52*inch)
    canvas.setFont(FONT, 7.5); canvas.setFillColor(GREY)
    canvas.drawString(0.65*inch, 0.32*inch, "ARM-SPARSE • VERIFIED TECHNICAL REPORT • SEPTEMBER 2026")
    canvas.drawRightString(7.85*inch, 0.32*inch, str(doc.page))
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(str(OUTPUT), pagesize=letter, rightMargin=.65*inch, leftMargin=.65*inch, topMargin=.58*inch, bottomMargin=.65*inch,
                          title="ARM-Sparse Technical Report", author="Sanjay Elango")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page))
    s = []
    s += [Spacer(1, .42*inch), p("ARM-SPARSE", "Smallx"), p("A Reproducible Investigation of the Quality–Latency Break-Even for Dynamic FFN Block Sparsity on Apple M2", "ReportTitle"), p("Technical report • September 2026 • Sanjay Elango", "Deck"), p("BLOCKS DID ACCELERATE THE ISOLATED FFN — BUT ONLY AFTER QUALITY HAD ALREADY FAILED", "Pull"), Spacer(1, .15*inch), frontier(), Spacer(1, .16*inch), p("Bottom line", "H2x"), p("Llama 3.2 1B preserved the screening quality gate through 10% B8 sparsity but first beat dense at 40%. TinyLlama preserved quality through 20% but first beat dense at 50%. No tested point satisfied both requirements."), p("This PDF is a layout-checked preview derived from committed evidence. The canonical manuscript is <font name='Courier'>paper/main.tex</font>.", "Smallx"), PageBreak()]
    s += [p("1. Research question and contribution", "H1x"), p("Dynamic FFN sparsity promises fewer operations, but a CPU must still pay for sparse indices, fragmented weight access, branches, dispatch, cache misses, and weak vector utilization. ARM-Sparse asks whether converting token-dependent neuron selections into fixed contiguous blocks can make those skipped operations visible in wall-clock time."), p("The compromise", "H2x"), p("A B8 executor operates on eight adjacent intermediate neurons at once. It may calculate neurons that an ideal irregular mask would skip, but it can reuse prepacked gate/up rows and down-projection columns with regular addressing. Weight packing happens once—not once per token."), p("What this report contributes", "H2x"), table([["Contribution", "Evidence boundary"], ["Measured quality–latency frontier", "Two ~1B Llama-family models on Apple M2"], ["Selector/executor separation", "Oracle quality masks; replayed kernel timings"], ["Strong dense comparison", "Apple Accelerate FP32 dense path"], ["Negative-result provenance", "18 retained experiments; five decisive"], ["Reproducible package", "Validators, raw artifacts, figures, LaTeX, tests"]], [2.1*inch, 4.8*inch]), Spacer(1, 8), p("Not claimed", "H2x"), p("ARM-Sparse does not invent contextual sparsity, neuron clustering, or contiguous packing. It does not demonstrate a deployable selector, end-to-end token-generation acceleration, or universal failure of block sparsity."), PageBreak()]
    s += [p("2. Related systems and the remaining gap", "H1x"), p("DejaVu established contextual sparsity and lightweight input-dependent predictors. LLM in a Flash showed why larger contiguous row/column transfers matter when memory movement dominates. PowerInfer separated frequently active and context-dependent neurons across CPU/GPU resources. PowerInfer-2 used neuron clusters as scheduling and storage units on smartphones. Dynamic Input Pruning combined SwiGLU pruning with cache-aware masks."), p("What remains worth measuring", "H2x"), p("Those systems prevent a novelty claim based merely on grouping neurons. The defensible question is narrower: at equal hardware, model geometry, and sparsity budget, where does block execution cross optimized dense latency, and does model quality survive to that same point?"), p("Prior-work relationship", "H2x"), table([["Work", "Relevant idea", "ARM-Sparse distinction"], ["DejaVu", "Contextual sparsity", "Executor frontier, not predictor novelty"], ["LLM in a Flash", "Bundled contiguous access", "RAM-resident CPU latency"], ["PowerInfer", "Hot/cold neurons", "CPU-only controlled executor"], ["PowerInfer-2", "Neuron clusters", "Quality–latency break-even"], ["Dynamic Input Pruning", "Cache-aware SwiGLU masks", "Apple M2 reproducibility"]], [1.25*inch, 2.25*inch, 3.4*inch]), Spacer(1, 8), p("Research policy", "H2x"), p("Hypotheses, observations, interpretations, and conclusions remain separate. Quantitative statements map to manifests; literature statements map to verified primary sources. Negative experiments remain visible."), PageBreak()]
    s += [p("3. Methodology and measurement boundary", "H1x"), p("A SwiGLU FFN computes y = Wd(SiLU(Wg x) * Wu x), where * is element-wise multiplication. The intermediate dimension is divided into contiguous groups. An active group executes its packed gate and up rows, then accumulates its packed down columns."), p("Quality protocol", "H2x"), p("Post-SwiGLU output-contribution oracle masks estimate the attainable quality boundary. Conditions use 16 deterministic WikiText-2 excerpts and 2,032 predicted tokens. A sparse point passes when perplexity is no more than 5% above the same-run dense condition."), p("Runtime protocol", "H2x"), p("Single-token synthetic FFNs match the model geometries: 2048×8192×2048 and 2048×5632×2048. Native packed FP32 B8 is compared with one-thread Accelerate dense. Every grid point retains 200 post-warmup measurements and is repeated across three seeds. Speedup is 100(tdense/tblock − 1)."), p("Threat-reducing choices", "H2x"), table([["Choice", "Why"], ["Oracle masks", "Measure attainable quality before predictor research"], ["Replay masks", "Measure executor independently from selector overhead"], ["Optimized dense baseline", "Avoid an academically weak PyTorch comparison"], ["Discrete grid, no interpolation", "Do not fabricate an exact crossover"], ["Per-run medians and ranges", "Expose repeatability without calling ranges confidence intervals"]], [2.35*inch, 4.55*inch]), p("The runtime setup favors sparsity because mask selection is excluded and clustered masks reduce fragmentation. A failure here is meaningful for this executor, although it does not generalize to trained block-aware models."), PageBreak()]
    s += [p("4. Quality frontier", "H1x"), line_chart("quality"), Spacer(1, 9), table([["Model / condition", "Relative perplexity", "Gate"], ["Llama B8 10%", "+2.520%", "Pass"], ["Llama B8 20%", "+8.773%", "Fail"], ["Llama B8 30%", "+20.168%", "Fail"], ["TinyLlama B8 10%", "+1.336%", "Pass"], ["TinyLlama B8 20%", "+3.157%", "Pass"], ["TinyLlama B8 30%", "+6.874%", "Fail"], ["TinyLlama B32 10%", "+3.620%", "Pass"], ["TinyLlama B32 20%", "+10.306%", "Fail"]], [3.3*inch, 2.0*inch, 1.3*inch]), Spacer(1, 8), p("Observation", "H2x"), p("TinyLlama tolerates more B8 sparsity than Llama, but larger B32 groups lose quality faster. Useful unstructured neuron sparsity therefore does not automatically imply useful contiguous block sparsity."), PageBreak()]
    s += [p("5. Runtime frontier", "H1x"), line_chart("runtime"), Spacer(1, 9), table([["Geometry", "Last slower point", "First faster point", "Speedup range"], ["Llama 2048×8192×2048", "30%", "40%", "+1.6 to +3.1%"], ["TinyLlama 2048×5632×2048", "40%", "50%", "+10.7 to +17.1%"]], [2.55*inch, 1.25*inch, 1.35*inch, 1.55*inch]), Spacer(1, 8), p("Observation", "H2x"), p("Blocks mechanically accelerate when enough groups are skipped. The smaller TinyLlama FFN needs more sparsity before overhead is amortized. At Llama's real-mask B8/10% passing point, block execution was 37.26–38.77% slower than Accelerate dense."), p("Interpretation", "H2x"), p("The speedup is real within the microbenchmark, but it is not usable under the chosen quality rule. Calling ARM-Sparse simply ‘accelerated’ would omit the most important half of the result."), PageBreak()]
    s += [p("6. Combined result and failed alternatives", "H1x"), frontier(), Spacer(1, 8), p("Five decisive experiments", "H2x"), table([["ID", "Role", "Outcome"], ["EXP-004", "Llama quality", "B8 ceiling: 10%"], ["EXP-005", "Passing-mask replay", "37–39% slower"], ["EXP-012", "Llama runtime", "First win: 40%"], ["EXP-016", "TinyLlama quality", "B8 ceiling: 20%"], ["EXP-018", "TinyLlama runtime", "First win: 50%"]], [.75*inch, 2.35*inch, 3.55*inch]), Spacer(1, 7), p("Why the other 13 experiments remain", "H2x"), p("They establish baselines, test mask expansion and scoring, evaluate neuron reordering and hot/cold layouts, profile phases, try explicit NEON and coalescing, and probe INT8/B32 execution. These attempts did not produce a quality-preserving win. Keeping them prevents cherry-picking; excluding them from the headline prevents an unreadable 18-experiment narrative."), p("Quantization boundary", "H2x"), p("The production llama.cpp Q8_0 result is a dense context baseline. The minimal B32 Q8_0 executor did not reproducibly beat its matched dense control through 80% sparsity. Custom INT8 wins against a weak custom dense control do not supersede the optimized Accelerate result."), PageBreak()]
    s += [p("7. Conclusion, next decision, and reproducibility", "H1x"), p("ARM-Sparse completed its bounded investigation. Contiguous blocks can accelerate isolated FFN execution, but the tested models lose acceptable quality before enough blocks can be skipped. Llama's quality ceiling and first speed point are 10% and 40%; TinyLlama's are 20% and 50%. No measured point satisfies both."), p("What should happen next", "H2x"), p("Do not continue tuning the same executor without a mechanism capable of moving the quality frontier by roughly 30 percentage points. A scientifically stronger continuation would train for block-aligned activation, jointly learn selector and layout, or evaluate a model architecture with native grouped structure. Each should have a preregistered stop condition."), p("One-command audit", "H2x"), p("After installing the declared development dependencies, run <font name='Courier'>make verify-package</font>. It validates core manifests, checks deterministic package artifacts, and runs the full test suite. <font name='Courier'>make package</font> regenerates the machine-readable summary, three SVG figures, and LaTeX table without touching raw results."), p("Evidence map", "H2x"), table([["Artifact", "Purpose"], ["research/claims.yaml", "Canonical quantitative claims and limitations"], ["research/package/summary.json", "Generated figure/table data"], ["research/package/claim-audit.md", "Claim and citation mapping"], ["paper/main.tex", "Canonical manuscript"], ["docs/article.md", "Public technical narrative"], ["REPRODUCING.md", "Environment and command boundary"]], [2.65*inch, 4.15*inch]), Spacer(1, 10), p("Final answer", "Pull"), p("Under the recorded Apple M2 setup, block-based dynamic FFN sparsity worked as a high-sparsity kernel optimization, but did not work as a quality-preserving LLM inference method."), p("Primary literature: Liu et al. (ICML 2023); Alizadeh et al. (ACL 2024); Song et al. (SOSP 2024); Xue et al. (2024); Sudarshan et al. (MLSys 2025). Full references are in paper/references.bib.", "Smallx")]
    doc.build(s)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
