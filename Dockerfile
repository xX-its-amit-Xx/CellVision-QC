FROM python:3.11-slim

LABEL maintainer="Amit Shenoy <shenoy.am@husky.neu.edu>"
LABEL description="CellVision-QC: fluorescence microscopy quality control"
LABEL org.opencontainers.image.source="https://github.com/ashenoy/CellVision-QC"

WORKDIR /workspace

# System dependencies for scipy / scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install the package
COPY pyproject.toml LICENSE README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir jupyter nbformat

# Copy examples and notebooks
COPY examples/ examples/
COPY notebooks/ notebooks/

# Default: show CLI help
ENTRYPOINT ["cellvision-qc"]
CMD ["--help"]
