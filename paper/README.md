# Paper package

`main.tex` is the canonical manuscript. Generated tables and SVG figures come
from `research_tools/package_results.py` and committed experiment artifacts.

The repository also includes `arm-sparse-report.pdf`, a layout-checked preview
generated from the same evidence because the packaging host had no TeX engine.
The PDF is not the canonical source. For final publication, compile `main.tex`
with a TeX distribution supporting `svg` and shell escape.
