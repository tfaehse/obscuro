# Copilot Hints (Obscuro)

## TL;DR
- Python 3.12+, managed with `uv`. No `pip install .`, no `venv`.
- Core pipeline lives in `src/anonymizer/` (detect → track → blur). State flows through Polars DataFrames and `AnonymizerConfig`.
- Interfaces:
  - CLI: `src/blur_cli/cli.py`
  - FastAPI backend: `src/blur_api/serve.py`
  - Electron renderer: `src/frontend/app/renderer`
- Logging, progress, and cancellation are already plumbed. Reuse the helpers in `src/anonymizer/utils`.

## When editing
- Prefer `pathlib.Path` over `os.path`.
- Keep DataFrame schemas primitive (ints/floats/bools/strings).
- For async HTTP handlers stick to FastAPI patterns already in `serve.py`.
- Frontend is TypeScript + vanilla DOM helpers (no React). Update the TS sources under `src/frontend/app/renderer` and run `npm run build` as needed.

## Builds & Tests
- Python deps: `uv sync --group test`, run tests via `uv run pytest`.
- Frontend: `cd src/frontend && npm install && npm run dev|build`.
- Docker: only `linux/amd64` + `linux/arm64` CPU images are published (`Dockerfile`). GPU image is currently disabled.

## Misc
- SAHI tiling is controlled via `config.detection.use_sahi`; CLI exposes `--use-sahi/--no-use-sahi`.
- Only blur types: `gaussian`, `pixelate`, `blackout`, `debug`.
