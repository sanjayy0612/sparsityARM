.PHONY: test package verify-package

test:
	PYTHONPATH=. .venv/bin/python -m pytest -q

package:
	.venv/bin/python research_tools/package_results.py

verify-package:
	PYTHONPATH=. .venv/bin/python research_tools/verify_core.py
	PYTHONPATH=. .venv/bin/python -m pytest -q
