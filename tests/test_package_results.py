import json

from research_tools import package_results


def test_summary_preserves_decisive_frontier():
    summary = package_results.build_summary()

    assert summary["decisive_experiments"] == [
        "EXP-004", "EXP-005", "EXP-012", "EXP-016", "EXP-018"
    ]
    assert summary["models"]["Llama 3.2 1B"]["quality_ceiling_pct"] == 10
    assert summary["models"]["Llama 3.2 1B"]["first_speedup_grid_pct"] == 40
    assert summary["models"]["TinyLlama 1.1B"]["quality_ceiling_pct"] == 20
    assert summary["models"]["TinyLlama 1.1B"]["first_speedup_grid_pct"] == 50


def test_committed_summary_matches_source_artifacts():
    committed = json.loads((package_results.ROOT / "research/package/summary.json").read_text())
    assert committed == package_results.build_summary()


def test_runtime_figure_excludes_stress_only_points():
    svg = package_results.svg_chart(package_results.build_summary(), "runtime")
    assert ">70<" not in svg
    assert ">80<" not in svg
    assert "B8 wall-clock speedup" in svg
