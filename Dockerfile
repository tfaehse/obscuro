# Backend runtime image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and project dependencies first for better caching
# Copy project metadata and sources for installation
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir uv \
 && uv pip install --system .

# Copy models (optional; mount alternate volumes in production if desired)
COPY models ./models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "blur_api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
