StockPig pyproject.toml hotfix

This corrects the initial cleanup patch's uv index configuration:
1. AMD ROCm index is explicit, so generic dependencies resolve from PyPI.
2. Python support begins at 3.12 to match the current ROCm 10 / NumPy dependency set.
3. AMD-only helper packages from the project's prior working environment are explicitly pinned to the AMD index.

From the StockPig repo root:

    unzip -o stockpig_pyproject_hotfix.zip -d .
    rm -f uv.lock
    uv lock
    uv sync --extra amd-gfx1201

Then verify:

    uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
