"""Build deterministic summary data, figures, and LaTeX tables from verified artifacts."""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text())


def quality_points(relative):
    data = load(relative)
    dense = data["results"][0]["perplexity"]
    return [{"sparsity_pct": round(row["sparsity"] * 100),
             "relative_ppl_pct": (row["perplexity"] / dense - 1) * 100}
            for row in data["results"]
            if row["condition"].startswith("block_output_norm-B8")]


def runtime_points(directory):
    base = ROOT / directory
    manifest = json.loads((base / "manifest.json").read_text())
    grouped = {}
    for item in manifest["cases"]:
        case = json.loads((base / item["path"]).read_text())
        dense = case["summary"]["dense_accelerate"]["median_ms"]
        block = case["summary"]["block_native"]["median_ms"]
        neuron = case["summary"]["irregular_native"]["median_ms"]
        grouped.setdefault(round(case["target_sparsity"] * 100), []).append(
            ((dense / block - 1) * 100, (dense / neuron - 1) * 100))
    points = []
    for sparsity, pairs in sorted(grouped.items()):
        block_values, neuron_values = [p[0] for p in pairs], [p[1] for p in pairs]
        points.append({"sparsity_pct": sparsity, "mean_speedup_pct": statistics.mean(block_values),
                       "min_speedup_pct": min(block_values), "max_speedup_pct": max(block_values),
                       "neuron_mean_speedup_pct": statistics.mean(neuron_values),
                       "neuron_min_speedup_pct": min(neuron_values),
                       "neuron_max_speedup_pct": max(neuron_values)})
    return points


def frontier(quality, runtime, gate):
    """Derive the decision-relevant grid points instead of hardcoding them."""
    ceiling = max(p["sparsity_pct"] for p in quality if p["relative_ppl_pct"] <= gate)
    first = min(p["sparsity_pct"] for p in runtime if p["min_speedup_pct"] > 0)
    at_ceiling = next(p for p in runtime if p["sparsity_pct"] == ceiling)
    return {"quality_ceiling_pct": ceiling, "first_speedup_grid_pct": first,
            "neuron_speedup_at_ceiling_min_pct": at_ceiling["neuron_min_speedup_pct"],
            "neuron_speedup_at_ceiling_max_pct": at_ceiling["neuron_max_speedup_pct"]}


def build_summary():
    llama_quality = quality_points("research/results/EXP-004/m2-oracle-screening-20260909/manifest.json")
    tiny_quality = quality_points("research/results/EXP-016/m2-tinyllama-screening-20260910-r1/manifest.json")
    llama_runtime = runtime_points("research/results/EXP-012/m2-b8-break-even-20260910")
    tiny_runtime = runtime_points("research/results/EXP-018/m2-tinyllama-b8-break-even-20260911-r1")
    return {
        "title": "ARM-Sparse quality-latency break-even on Apple M2",
        "quality_gate_relative_ppl_pct": 5.0,
        "models": {
            "Llama 3.2 1B": {"quality": llama_quality, "runtime": llama_runtime,
                             **frontier(llama_quality, llama_runtime, 5.0)},
            "TinyLlama 1.1B": {"quality": tiny_quality, "runtime": tiny_runtime,
                               **frontier(tiny_quality, tiny_runtime, 5.0)},
        },
        "decisive_experiments": ["EXP-004", "EXP-005", "EXP-012", "EXP-016", "EXP-018"],
        "conclusion": ("The packed B8 executor has no quality-speed intersection; the per-neuron "
                       "executor on the same B8 masks is faster than dense at the quality ceiling."),
    }


COLORS = {"Llama 3.2 1B": "#2563eb", "TinyLlama 1.1B": "#e11d48"}


def svg_chart(summary, kind):
    width, height = 920, 540
    left, right, top, bottom = 92, 34, 56, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    if kind == "quality":
        title, ylabel, ymin, ymax = "Quality cost of B8 sparsity", "Perplexity increase (%)", 0, 24
        xs = [0, 10, 20, 30]
        series = {name: [(p["sparsity_pct"], p["relative_ppl_pct"]) for p in values["quality"]]
                  for name, values in summary["models"].items()}
        threshold = 5
    else:
        title, ylabel, ymin, ymax = "B8 wall-clock speedup on Apple M2", "Speedup vs Accelerate dense (%)", -35, 55
        # Keep the publication figure on the decision-relevant 20--60% sweep.
        # EXP-012 also contains 70/80% stress points, but including them would
        # compress the quality-relevant region and visually overstate extremes.
        xs = list(range(0, 61, 10))
        series = {name: [(p["sparsity_pct"], p["mean_speedup_pct"]) for p in values["runtime"]
                         if p["sparsity_pct"] <= 60]
                  for name, values in summary["models"].items()}
        threshold = 0
    xmax = max(xs)
    px = lambda x: left + x / xmax * plot_w
    py = lambda y: top + (ymax - y) / (ymax - ymin) * plot_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{left}" y="30" font-family="Helvetica,Arial" font-size="22" font-weight="700" fill="#111827">{title}</text>']
    ticks = range(ymin, ymax + 1, 5 if kind == "quality" else 10)
    for value in ticks:
        y = py(value)
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Helvetica,Arial" font-size="13" fill="#4b5563">{value}</text>']
    for value in xs:
        x = px(value)
        out += [f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="Helvetica,Arial" font-size="13" fill="#4b5563">{value}</text>']
    y = py(threshold)
    out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="7 6"/>')
    label = "5% quality gate" if kind == "quality" else "dense parity"
    out.append(f'<text x="{width-right-4}" y="{y-8:.1f}" text-anchor="end" font-family="Helvetica,Arial" font-size="12" fill="#111827">{label}</text>')
    for index, (name, points) in enumerate(series.items()):
        color = COLORS[name]
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, value in points:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(value):.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="2"/>')
        lx, ly = left + index * 220, height - 18
        out += [f'<line x1="{lx}" y1="{ly-5}" x2="{lx+30}" y2="{ly-5}" stroke="{color}" stroke-width="4"/>',
                f'<text x="{lx+40}" y="{ly}" font-family="Helvetica,Arial" font-size="14" fill="#111827">{escape(name)}</text>']
    out += [f'<text x="{left+plot_w/2:.1f}" y="{height-40}" text-anchor="middle" font-family="Helvetica,Arial" font-size="15" fill="#111827">B8 sparsity (%)</text>',
            f'<text transform="translate(24 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Helvetica,Arial" font-size="15" fill="#111827">{ylabel}</text>',
            '</svg>']
    return "\n".join(out) + "\n"


def frontier_svg(summary):
    width, height = 920, 420
    left, right = 190, 50
    plot_w = width - left - right
    px = lambda x: left + x / 60 * plot_w
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<text x="55" y="42" font-family="Helvetica,Arial" font-size="22" font-weight="700" fill="#111827">The quality-speed gap</text>',
           '<text x="55" y="70" font-family="Helvetica,Arial" font-size="14" fill="#4b5563">Highest passing B8 point versus first reproducible speedup grid point</text>']
    for tick in range(0, 61, 10):
        x = px(tick)
        out += [f'<line x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="330" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="360" text-anchor="middle" font-family="Helvetica,Arial" font-size="13" fill="#4b5563">{tick}%</text>']
    for index, (name, model) in enumerate(summary["models"].items()):
        y = 160 + index * 120
        q, s = model["quality_ceiling_pct"], model["first_speedup_grid_pct"]
        out += [f'<text x="{left-18}" y="{y+5}" text-anchor="end" font-family="Helvetica,Arial" font-size="15" font-weight="700" fill="#111827">{escape(name)}</text>',
                f'<line x1="{px(q):.1f}" y1="{y}" x2="{px(s):.1f}" y2="{y}" stroke="#9ca3af" stroke-width="6"/>',
                f'<circle cx="{px(q):.1f}" cy="{y}" r="11" fill="#16a34a"/>',
                f'<circle cx="{px(s):.1f}" cy="{y}" r="11" fill="#dc2626"/>',
                f'<text x="{px(q):.1f}" y="{y-20}" text-anchor="middle" font-family="Helvetica,Arial" font-size="13" fill="#166534">quality {q}%</text>',
                f'<text x="{px(s):.1f}" y="{y-20}" text-anchor="middle" font-family="Helvetica,Arial" font-size="13" fill="#991b1b">speed {s}%</text>']
    out += ['<circle cx="260" cy="397" r="6" fill="#16a34a"/><text x="274" y="402" font-family="Helvetica,Arial" font-size="13">highest measured point passing quality gate</text>',
            '<circle cx="610" cy="397" r="6" fill="#dc2626"/><text x="624" y="402" font-family="Helvetica,Arial" font-size="13">first measured speedup point</text>', '</svg>']
    return "\n".join(out) + "\n"


def csv_text(summary):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["model", "metric", "sparsity_pct", "value_pct", "min_pct", "max_pct"])
    for name, model in summary["models"].items():
        for point in model["quality"]:
            writer.writerow([name, "relative_perplexity_increase", point["sparsity_pct"],
                             f'{point["relative_ppl_pct"]:.6f}', "", ""])
        for point in model["runtime"]:
            writer.writerow([name, "wall_clock_speedup", point["sparsity_pct"],
                             f'{point["mean_speedup_pct"]:.6f}',
                             f'{point["min_speedup_pct"]:.6f}', f'{point["max_speedup_pct"]:.6f}'])
            writer.writerow([name, "wall_clock_speedup_per_neuron", point["sparsity_pct"],
                             f'{point["neuron_mean_speedup_pct"]:.6f}',
                             f'{point["neuron_min_speedup_pct"]:.6f}', f'{point["neuron_max_speedup_pct"]:.6f}'])
    return stream.getvalue()


def pgf_data(summary, kind):
    # Whitespace-separated tables read by pgfplots in paper/figures/*.tex.
    lines = []
    if kind == "quality":
        lines.append("model sparsity value")
        for index, model in enumerate(summary["models"].values()):
            for point in [{"sparsity_pct": 0, "relative_ppl_pct": 0.0}] + model["quality"]:
                lines.append(f'{index} {point["sparsity_pct"]} {point["relative_ppl_pct"]:.6f}')
    else:
        lines.append("model sparsity value min max neuron neuronmin neuronmax")
        for index, model in enumerate(summary["models"].values()):
            for point in model["runtime"]:
                lines.append(f'{index} {point["sparsity_pct"]} {point["mean_speedup_pct"]:.6f} '
                             f'{point["min_speedup_pct"]:.6f} {point["max_speedup_pct"]:.6f} '
                             f'{point["neuron_mean_speedup_pct"]:.6f} '
                             f'{point["neuron_min_speedup_pct"]:.6f} {point["neuron_max_speedup_pct"]:.6f}')
    return "\n".join(lines) + "\n"


def latex_grid_table(summary):
    gate = summary["quality_gate_relative_ppl_pct"]
    models = list(summary["models"].values())
    grid = sorted({p["sparsity_pct"] for m in models for key in ("quality", "runtime") for p in m[key]})

    def quality_cell(model, sparsity):
        point = next((p for p in model["quality"] if p["sparsity_pct"] == sparsity), None)
        if point is None:
            return "--"
        mark = "\\checkmark" if point["relative_ppl_pct"] <= gate else "$\\times$"
        return f'{point["relative_ppl_pct"]:+.2f} {mark}'

    def signed(value):
        return f"${value:+.1f}$"

    def speed_cell(prefix):
        def cell(model, sparsity):
            point = next((p for p in model["runtime"] if p["sparsity_pct"] == sparsity), None)
            if point is None:
                return "--"
            return (f'{signed(point[prefix + "mean_speedup_pct"])} '
                    f'[{signed(point[prefix + "min_speedup_pct"])}, {signed(point[prefix + "max_speedup_pct"])}]')
        return cell

    cells = (quality_cell, speed_cell(""), speed_cell("neuron_"))
    rows = [f"{s}\\% & " + " & ".join(cell(m, s) for m in models for cell in cells) + " \\\\"
            for s in grid]
    names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name in summary["models"])
    return ("\\begin{tabular}{r" + "rrr" * len(models) + "}\n\\toprule\n"
            f" & {names} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "Sparsity" + " & $\\Delta$PPL & Packed B8 & Per-neuron" * len(models) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def latex_table(summary):
    rows = []
    for name, model in summary["models"].items():
        rows.append(f"{name} & {model['quality_ceiling_pct']}\\% & {model['first_speedup_grid_pct']}\\% & "
                    f"${model['neuron_speedup_at_ceiling_min_pct']:+.1f}$ to "
                    f"${model['neuron_speedup_at_ceiling_max_pct']:+.1f}$\\% \\\\")
    return """\\begin{tabular}{lrrr}
\\toprule
 & Quality & First B8 & Per-neuron at \\\\
Model & ceiling & speedup & ceiling \\\\
\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = build_summary()
    outputs = {
        ROOT / "research/package/summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
        ROOT / "research/package/summary.csv": csv_text(summary),
        ROOT / "paper/figures/quality-vs-sparsity.svg": svg_chart(summary, "quality"),
        ROOT / "paper/figures/speedup-vs-sparsity.svg": svg_chart(summary, "runtime"),
        ROOT / "paper/figures/quality-speed-frontier.svg": frontier_svg(summary),
        ROOT / "paper/figures/quality.dat": pgf_data(summary, "quality"),
        ROOT / "paper/figures/speedup.dat": pgf_data(summary, "runtime"),
        ROOT / "paper/tables/key-results.tex": latex_table(summary),
        ROOT / "paper/tables/full-grid.tex": latex_grid_table(summary),
    }
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"stale generated artifact: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(("Validated" if args.check else "Generated"), len(outputs), "package artifacts")


if __name__ == "__main__":
    main()
