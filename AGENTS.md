# Agent Instructions for Obscuro

## Overview
Obscuro is a desktop-first video anonymization tool (FastAPI backend + CLI + Electron UI) built around an ONNX/SAHI detector pipeline, tracking, and configurable anonymization targets. Primary language is Python 3.12; frontend is Electron + TypeScript.

## Repo layout
- `src/anonymizer/`: core anonymization pipeline, models, IO, tracking, segmentation.
- `src/blur_api/`: FastAPI backend.
- `src/blur_cli/`: CLI entrypoints.
- `src/frontend/`: Electron + TypeScript desktop app.
- `config/`: default configuration files.
- `models/`: detection/tracking model artifacts.
- `tests/`: pytest suite.
- `docs/`: architecture + API/CLI/config docs.

## Environment setup
- Python: 3.12+
- Create env + install deps:
  - `uv sync`
  - GPU deps (NVIDIA only): `uv sync --extra gpu`
- Frontend deps (Electron):
  - `cd src/frontend && npm install`

## Common commands
- Run API: `uv run blur-api`
- Run CLI: `uv run blur-cli`
- Run Electron app (from `src/frontend`): `npm run dev`

## Tests and quality
- Run tests: `uv run pytest`
- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Type checks (Python subset): `uv run ty check`
- Frontend type check (from `src/frontend`): `npm run type-check`

## Project conventions
- Python formatting: Ruff (line length 100, double quotes)
- Avoid large model file changes unless explicitly requested.
- Use existing configuration files in `config/` rather than duplicating settings.
- Prefer `pathlib` over `os.path`.
- Prefer `polars` over `pandas`.
- Inference speed is critical; avoid on-the-fly mask decoding.
- Keep logical steps (tracking, etc.) in relative coordinates so parameters generalize across input sizes.
- Do not guard against missing dependencies; crash if a required dependency is missing.

## Notes
- The Electron build bundles backend assets from the repo; see `src/frontend/package.json` `build.extraResources`.
- Models and trackers can require external weights; consult `docs/` if a change touches tracking or detection.
