FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install ".[caso]"

FROM base AS dev
RUN pip install ".[dev,caso]"
COPY tests/ tests/
CMD ["pytest", "-v"]

FROM base AS runtime
ENTRYPOINT ["psi-lab1"]
CMD ["--salida", "/out"]