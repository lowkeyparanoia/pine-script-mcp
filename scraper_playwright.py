"""
scraper_playwright.py - TradingView Pine Script Reference Scraper
=================================================================
Uses Playwright (headless Chromium) to fully render the SPA reference manual
at https://www.tradingview.com/pine-script-reference/v5/ and extract every
function/variable with its complete signature, parameters, description,
and code examples.

Strategy (single page load, no per-function navigation):
  1. Intercept network responses to capture any JSON API data
  2. Navigate to reference page, wait for full render
  3. Extract all sidebar anchor links (full function list)
  4. Scroll the page to trigger lazy rendering of all sections
  5. Parse every function entry from the rendered DOM in one pass

Run standalone:
    python scraper_playwright.py                    # scrape + save JSON
    python scraper_playwright.py --out custom.json  # custom output path
    python scraper_playwright.py --debug            # show browser, verbose

Called by indexer.py automatically when playwright is available.
"""

import asyncio
import json
import re
import sys
import argparse
import time
from pathlib import Path

REF_URL   = "https://www.tradingview.com/pine-script-reference/v5/"
OUT_FILE  = Path(__file__).parent / "index_cache" / "reference_scraped.json"


# ── DOM extraction (runs inside the browser page via page.evaluate) ──────────

_EXTRACT_JS = """
() => {
    const results = [];

    // Each documented item has a wrapper with a data-id or id attribute
    // TradingView reference uses sections with class patterns for functions/vars

    // Collect all anchor-targetable sections
    const sections = document.querySelectorAll(
        '[id^="fun_"], [id^="var_"], [id^="op_"], [id^="type_"], [id^="const_"], [id^="ann_"]'
    );

    sections.forEach(el => {
        const id      = el.id || "";
        const kind    = id.startsWith("fun_")   ? "function"  :
                        id.startsWith("var_")   ? "variable"  :
                        id.startsWith("op_")    ? "operator"  :
                        id.startsWith("type_")  ? "type"      :
                        id.startsWith("const_") ? "constant"  :
                        id.startsWith("ann_")   ? "annotation": "other";

        // Name: strip prefix
        const name = id.replace(/^(fun_|var_|op_|type_|const_|ann_)/, "");

        // Look for the closest containing article/section/div
        const container = el.closest("article, section, .tv-pine-reference-item, div[class*='item'], div[class*='entry']") || el.parentElement;

        // Signature: <code> or <pre> near the top of container
        let signature = "";
        const sigEl = container && (
            container.querySelector("code.tv-pine-reference-syntax, pre.syntax, .signature code, h3 + p code, .function-signature") ||
            container.querySelector("code")
        );
        if (sigEl) signature = sigEl.textContent.trim();

        // Description: first <p> after signature
        let description = "";
        const descEls = container ? container.querySelectorAll("p") : [];
        if (descEls.length > 0) {
            description = Array.from(descEls)
                .slice(0, 3)
                .map(p => p.textContent.trim())
                .filter(t => t.length > 5 && !t.startsWith("//"))
                .join(" ");
        }

        // Parameters: <dl>, <table>, or list items describing params
        const params = [];
        const paramEls = container ? container.querySelectorAll("dt, th, .param-name, [class*='param']") : [];
        paramEls.forEach(dt => {
            const name_ = dt.textContent.trim();
            const dd = dt.nextElementSibling;
            const pdesc = dd ? dd.textContent.trim() : "";
            if (name_ && name_.length < 60) {
                params.push(name_ + (pdesc ? ": " + pdesc.slice(0, 200) : ""));
            }
        });

        // Returns: look for "Returns" heading
        let returns = "";
        const headers = container ? container.querySelectorAll("h4, h5, strong, b") : [];
        headers.forEach(h => {
            if (/returns?/i.test(h.textContent)) {
                const next = h.nextElementSibling || h.parentElement.nextElementSibling;
                if (next) returns = next.textContent.trim().slice(0, 200);
            }
        });

        // Code examples: <pre><code> blocks
        const examples = [];
        const preEls = container ? container.querySelectorAll("pre code, pre") : [];
        preEls.forEach(pre => {
            const code = pre.textContent.trim();
            if (code.length > 10 && code.length < 2000) examples.push(code);
        });

        // Full text of the section for BM25
        const fullText = container ? container.innerText || container.textContent : "";

        if (name && (signature || description || fullText.length > 30)) {
            results.push({
                id,
                kind,
                name,
                signature,
                description,
                params,
                returns,
                examples,
                text: fullText.slice(0, 3000)
            });
        }
    });

    return results;
}
"""

_GET_ANCHORS_JS = """
() => {
    // Get all anchor hrefs from sidebar/nav that point to # fragments
    const links = document.querySelectorAll('a[href^="#fun_"], a[href^="#var_"], a[href^="#op_"], a[href^="#type_"], a[href^="#const_"]');
    return Array.from(links).map(a => ({
        href: a.getAttribute("href"),
        text: a.textContent.trim()
    }));
}
"""

_SCROLL_AND_EXPAND_JS = """
async () => {
    // Scroll the page gradually to trigger lazy rendering
    const scrollStep = window.innerHeight * 0.8;
    const totalHeight = document.body.scrollHeight;
    let pos = 0;
    while (pos < totalHeight) {
        window.scrollTo(0, pos);
        await new Promise(r => setTimeout(r, 80));
        pos += scrollStep;
    }
    // Scroll back up
    window.scrollTo(0, 0);
    await new Promise(r => setTimeout(r, 500));
    return document.body.scrollHeight;
}
"""


# ── Main scraper ─────────────────────────────────────────────────────────────

async def scrape(debug: bool = False, out: Path = OUT_FILE) -> list[dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("[playwright] Not installed. Run: pip install playwright && python -m playwright install chromium")
        return []

    out.parent.mkdir(parents=True, exist_ok=True)
    captured_json: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not debug)
        ctx     = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        )
        page = await ctx.new_page()

        # ── Intercept network: capture any JSON docs API response ────────────
        async def on_response(resp):
            url = resp.url
            if resp.status == 200 and "pine" in url.lower():
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    try:
                        data = await resp.json()
                        if isinstance(data, (dict, list)) and len(str(data)) > 500:
                            captured_json.append({"url": url, "data": data})
                            if debug:
                                print(f"  [API] Captured: {url} ({len(str(data))} chars)")
                    except Exception:
                        pass

        page.on("response", on_response)

        # ── Load the reference page ──────────────────────────────────────────
        print(f"[playwright] Loading {REF_URL}")
        try:
            await page.goto(REF_URL, wait_until="networkidle", timeout=45000)
        except Exception:
            await page.goto(REF_URL, wait_until="domcontentloaded", timeout=45000)

        await page.wait_for_timeout(3000)

        # ── Get sidebar anchor list (tells us all documented items) ──────────
        anchors = await page.evaluate(_GET_ANCHORS_JS)
        print(f"[playwright] Found {len(anchors)} sidebar anchors")

        # ── Scroll to trigger lazy loading ───────────────────────────────────
        print("[playwright] Scrolling page to trigger lazy rendering...")
        try:
            page_height = await page.evaluate_handle(_SCROLL_AND_EXPAND_JS)
            await page.wait_for_timeout(1000)
        except Exception as e:
            if debug:
                print(f"  [WARN] Scroll error: {e}")

        # ── Extract all sections from DOM ────────────────────────────────────
        print("[playwright] Extracting DOM sections...")
        raw_sections = await page.evaluate(_EXTRACT_JS)
        print(f"[playwright] DOM extraction: {len(raw_sections)} sections")

        # ── If DOM extraction missed items, navigate to anchors that have names
        #    but no matching DOM section (fill gaps) ───────────────────────────
        dom_names = {s["name"] for s in raw_sections}
        missing   = [a for a in anchors if a["href"].lstrip("#").replace("fun_","").replace("var_","") not in dom_names]

        if missing and len(missing) < 200:
            print(f"[playwright] Filling {len(missing)} gaps via anchor navigation...")
            for anchor_info in missing[:150]:   # cap to avoid timeout
                href = anchor_info["href"]
                try:
                    await page.evaluate(f"document.querySelector('{href}') && document.querySelector('{href}').scrollIntoView()")
                    await page.wait_for_timeout(120)
                    # Re-check DOM after scroll
                    sect = await page.evaluate(f"""
                    () => {{
                        const el = document.querySelector('{href}');
                        if (!el) return null;
                        const cont = el.closest('article, section, div') || el.parentElement;
                        return cont ? cont.innerText.slice(0, 2000) : null;
                    }}
                    """)
                    if sect and len(sect) > 30:
                        name = href.lstrip("#").replace("fun_","").replace("var_","").replace("op_","")
                        raw_sections.append({
                            "id":          href.lstrip("#"),
                            "kind":        "function" if "fun_" in href else "variable",
                            "name":        name,
                            "signature":   "",
                            "description": "",
                            "params":      [],
                            "returns":     "",
                            "examples":    [],
                            "text":        sect
                        })
                except Exception:
                    pass

        # ── Process captured API JSON (bonus data if TV has an endpoint) ─────
        api_docs = []
        for cap in captured_json:
            data = cap["data"]
            items = data if isinstance(data, list) else data.get("items", data.get("functions", data.get("docs", [])))
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and ("name" in item or "id" in item):
                        api_docs.append(item)

        if api_docs:
            print(f"[playwright] Bonus: {len(api_docs)} items from intercepted API")

        await browser.close()

    # ── Build final doc list in indexer format ───────────────────────────────
    final_docs: list[dict] = []
    seen: set[str] = set()

    for s in raw_sections:
        name = s.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)

        kind = s.get("kind", "function")
        sig  = s.get("signature", "")
        desc = s.get("description", "")
        text = s.get("text", "")

        # Build rich text for BM25
        rich = f"Pine Script {kind}: {name}"
        if sig:
            rich += f"\nSignature: {sig}"
        if desc:
            rich += f"\nDescription: {desc}"
        params = s.get("params", [])
        if params:
            rich += "\nParameters:\n  " + "\n  ".join(params[:10])
        ret = s.get("returns", "")
        if ret:
            rich += f"\nReturns: {ret}"
        if text and len(text) > len(rich):
            rich += "\n" + text[:1000]

        anchor = f"fun_{name}" if kind == "function" else f"var_{name}"
        final_docs.append({
            "url":      f"{REF_URL}#{anchor}",
            "title":    name,
            "text":     rich,
            "headings": [],
            "code":     s.get("examples", []),
            "source":   "playwright_reference",
            "kind":     kind,
            "signature":sig,
        })

    # Merge API docs
    for item in api_docs:
        name = item.get("name") or item.get("id", "")
        if not name or name in seen:
            continue
        seen.add(name)
        final_docs.append({
            "url":      f"{REF_URL}#fun_{name}",
            "title":    name,
            "text":     f"Pine Script function: {name}. {item.get('description','')} {item.get('syntax','')}",
            "headings": [],
            "code":     [item.get("example", "")],
            "source":   "playwright_reference",
            "kind":     "function",
            "signature":item.get("syntax", ""),
        })

    print(f"[playwright] Final: {len(final_docs)} unique reference docs")

    # Save
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final_docs, f, indent=2, ensure_ascii=False)
    print(f"[playwright] Saved -> {out}")

    return final_docs


def run(debug: bool = False, out: Path = OUT_FILE) -> list[dict]:
    """Sync entry point for calling from indexer.py."""
    return asyncio.run(scrape(debug=debug, out=out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playwright scraper for TradingView Pine Script reference")
    parser.add_argument("--debug", action="store_true", help="Show browser window and verbose output")
    parser.add_argument("--out",   default=str(OUT_FILE), help="Output JSON file path")
    args = parser.parse_args()
    docs = run(debug=args.debug, out=Path(args.out))
    print(f"\nDone: {len(docs)} docs. Integrate with: python indexer.py --force")
