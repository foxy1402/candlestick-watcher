# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Build the TA-Lib C library from source (required by the `ta-lib` Python wheel).
ARG TA_LIB_VERSION=0.6.4
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential wget ca-certificates; \
    wget -q "https://github.com/ta-lib/ta-lib/releases/download/v${TA_LIB_VERSION}/ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    tar -xzf "ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    cd "ta-lib-${TA_LIB_VERSION}"; \
    ./configure --prefix=/usr; \
    make -j"$(nproc)"; \
    make install; \
    cd ..; \
    rm -rf "ta-lib-${TA_LIB_VERSION}" "ta-lib-${TA_LIB_VERSION}-src.tar.gz"; \
    ldconfig

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Drop build-only packages to keep the runtime image small
# (the compiled libta-lib shared library stays in /usr/lib).
RUN set -eux; \
    apt-get purge -y --auto-remove build-essential wget; \
    rm -rf /var/lib/apt/lists/*

COPY . .

# PORT is honored by most PaaS (Render, Railway, Portainer, etc.).
# Override it to change the listening port without rebuilding.
ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,sys,urllib.request; url='http://127.0.0.1:'+os.environ.get('PORT','8501')+'/_stcore/health'; sys.exit(0 if urllib.request.urlopen(url, timeout=4).status==200 else 1)" || exit 1

# Shell form so ${PORT} is expanded at runtime.
CMD streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0 --browser.gatherUsageStats=false
