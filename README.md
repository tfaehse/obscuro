# Obscuro

<p align="center">
  <img src="docs/assets/obscuro-icon.png" alt="Obscuro logo" width="140" height="140" />
</p>

Desktop-first tooling for anonymizing dashcam footage: optimized ONNX/SAHI detector pipeline, FastAPI backend, CLI utilities, and an Electron UI with live previews.

## Documentation

- [Overview & architecture](docs/index.md)
- [FastAPI endpoints](docs/api-reference.md)
- [CLI commands](docs/cli-reference.md)
- [Python API guide](docs/python-api.md)
- [Configuration reference](docs/configuration.md)

## Quick start

![Obscuro desktop application](docs/assets/application.png)

*Example of the Electron desktop frontend with live video preview and anonymization controls*


```bash
# Create the dev environment (Python 3.12+)
uv sync

# Run the backend API
uv run blur-api

# Launch the Electron desktop frontend (in src/frontend/)
npm install
npm run dev
```

Need CUDA? Only enable the GPU dependencies on NVIDIA-enabled hosts:

```bash
uv sync --extra gpu
```

Container images (CPU and GPU) can be built with `docker build -f Dockerfile[.gpu] .` or consumed from GitHub Container Registry. See the docs linked above for volumes, ports, and auto-start configuration.

## Contributing

Open issues and pull requests on [GitHub](https://github.com/tfaehse/obscuro). Run `uv run pytest` and `uv run ruff check` locally before submitting. Artwork and screenshots live under `docs/assets/`.

## License & attribution

Licensed under **AGPL-3.0-or-later**. Detector checkpoints are based on Ultralytics YOLO pretrained weights—credit Ultralytics if you redistribute derived models. See `LICENSE` and the documentation for full attribution guidance.
