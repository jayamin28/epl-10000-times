"""The live page, opened in a real browser against the committed feed.

Skipped where Playwright is not installed (the weekly CI job only needs the model).
"""
from __future__ import annotations

import functools
import http.server
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

SITE = Path(__file__).resolve().parents[1] / "site"
CARD = "#0B0D19"


def lum(hex_: str) -> float:
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    x, y = lum(a), lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


@pytest.fixture(scope="module")
def server():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # pragma: no cover
            pytest.skip(f"no browser: {e}")
        yield b
        b.close()


def open_page(browser, url, width):
    page = browser.new_page(viewport={"width": width, "height": 900})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/*", lambda r: r.continue_() if r.request.url.startswith(url) else r.abort())
    page.goto(url)
    page.wait_for_selector("#content:not(.hidden)", timeout=15000)
    return page, errors


@pytest.mark.parametrize("width", [390, 1280])
def test_page_renders_without_errors_or_sideways_scroll(server, browser, width):
    page, errors = open_page(browser, server, width)
    assert errors == []
    assert page.evaluate("document.documentElement.scrollWidth") <= width
    page.close()


def test_page_carries_the_launch_brand_and_no_preview_tag(server, browser):
    page, _ = open_page(browser, server, 1280)
    assert "10,000 different universes" in page.text_content("h1")
    assert page.title() == "The EPL in 10,000 different universes"
    assert "after Gameweek" in page.inner_text("#stamp")
    assert "preview" not in page.content().lower()
    page.close()


def test_every_club_colour_reads_on_the_dark_page(server, browser):
    page, _ = open_page(browser, server, 1280)
    teams = page.eval_on_selector_all("#pick option", "os => os.map(o => o.value)")
    assert len(teams) == 20
    failures = []
    for t in teams:
        page.select_option("#pick", t)
        accent = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()")
        ink = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--accent-ink').trim()")
        if contrast(accent, CARD) < 3:
            failures.append((t, accent, "mark"))
        if contrast(accent, ink) < 4.5:
            failures.append((t, accent, ink, "chip text"))
    assert failures == []
    page.close()
