# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ta-lib >= 0.6 ships a self-contained manylinux wheel, so the C library is
# bundled and no compiler / source build is needed in the image.
#
# Install and prune must share one layer: deleting files in a later RUN leaves
# their bytes behind in the earlier layer and does not shrink the image.
COPY requirements.txt ./
RUN set -eux; \
    pip install --no-cache-dir -r requirements.txt; \
    SP=/usr/local/lib/python3.11/site-packages; \
    # Streamlit display extras this app never uses (no st.image / st.altair_chart / st.pydeck_chart).
    pip uninstall -y pydeck altair pillow; \
    # Test suites and headers shipped inside runtime wheels.
    rm -rf "$SP"/pandas/tests "$SP"/numpy/tests "$SP"/pyarrow/tests "$SP"/pyarrow/include; \
    # Arrow Flight is unused; keep libarrow_substrait/libparquet (libarrow links against them).
    rm -rf "$SP"/pyarrow/flight* "$SP"/pyarrow/libarrow_flight.so.*; \
    # Keep package_data/templates: plotly_dark is loaded from there at runtime.
    rm -rf "$SP"/plotly/package_data/datasets "$SP"/plotly/package_data/*.js \
           "$SP"/plotly/labextension; \
    # Drop the package manager from the runtime image.
    python -m pip uninstall -y pip setuptools wheel

COPY app.py ./
COPY .streamlit/ ./.streamlit/

# The app only reads its files; run it as an unprivileged user.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 appuser
USER appuser

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
CMD streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0 --server.fileWatcherType=none --browser.gatherUsageStats=false
