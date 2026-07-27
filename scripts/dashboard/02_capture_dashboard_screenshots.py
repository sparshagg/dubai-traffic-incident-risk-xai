from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "dashboard" / "app.py"
OUTPUT_DIR = ROOT / "reports" / "dashboard" / "screenshots"
VIEWS = {
    "dashboard_rolling_replay.png": "Rolling replay",
    "dashboard_zone_explanation.png": "Zone explanation",
    "dashboard_historical_check.png": "Historical check",
    "dashboard_model_evidence.png": "Model evidence",
}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"Streamlit server did not start within {timeout_seconds} seconds")


def capture_views(base_url: str) -> None:
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            raise SystemExit(
                "Could not launch the installed Chrome browser through Playwright. "
                "Install Google Chrome or run `python -m playwright install chromium`."
            ) from exc
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
        try:
            for filename, view in VIEWS.items():
                encoded = urllib.parse.quote(view)
                url = f"{base_url}/?view={encoded}"
                output_path = OUTPUT_DIR / filename
                page.goto(url, wait_until="domcontentloaded", timeout=90_000)
                page.get_by_text("Dubai incident-risk replay").wait_for(timeout=90_000)
                page.wait_for_timeout(8_000)
                page.screenshot(path=str(output_path), full_page=True)
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise SystemExit(f"Screenshot was not written: {output_path}")
                print(f"Wrote {output_path}")
        finally:
            browser.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.headless=true",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        wait_for_server(base_url)
        capture_views(base_url)
    finally:
        if server.poll() is None:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)


if __name__ == "__main__":
    main()
