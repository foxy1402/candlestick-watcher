"""Modal deployment for the Crypto Candlestick Pattern Watcher (Streamlit app).

Deploy with:
    modal deploy modal_app.py

The app is served via Modal's `web_server` on port 8501. API keys are read
from Modal secrets (also falls back to env vars) -- see get_secret() in app.py.
Create them with:
    modal secret create coinalyze-secret COINALYZE_API_KEY=your-key
    modal secret create cryptocompare-secret CRYPTOCOMPARE_API_KEY=your-key   # optional
"""

import modal
from pathlib import Path

# Resolve everything relative to this file so deploys work from any cwd.
LOCAL_DIR = Path(__file__).parent

# Modal does not pick up .modalignore on its own; load it explicitly so VCS
# metadata, local caches and secrets never enter the image or its mounts.
ignore = modal.FilePatternMatcher.from_file(LOCAL_DIR / ".modalignore")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ta-lib >= 0.6 ships a self-contained manylinux wheel (bundled C library),
    # so no system packages or source build are required.
    .pip_install_from_requirements(str(LOCAL_DIR / "requirements.txt"))
    .add_local_dir(LOCAL_DIR, remote_path="/app", ignore=ignore)
)

app = modal.App("candlestick-pattern-watcher", image=image)


@app.function(
    timeout=3600,
    # Modal's default web-function allocation (0.125 CPU / 128 MiB) is far too
    # small for Streamlit + pandas + Plotly; the app OOMs or crawls.
    cpu=1.0,
    memory=1024,
    secrets=[
        modal.Secret.from_name("coinalyze-secret"),
        modal.Secret.from_name("cryptocompare-secret"),
    ],
)
# Without this, every parallel request (Streamlit loads dozens of JS chunks)
# gets its own container; one page load spawned 6+ Streamlit servers.
@modal.concurrent(max_inputs=100)
@modal.web_server(8501, startup_timeout=120)
def serve():
    import subprocess

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "/app/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ],
        cwd="/app",
    )
