"""
indexer.py - TradingView Pine Script Documentation Indexer
==========================================================
Scrapes TradingView Pine Script docs and builds a BM25 search index.

Two sources:
  1. User Manual (pine-script-docs/) - static HTML, fully scrapable
  2. Reference Manual (pine-script-reference/v5/) - SPA, scraped via
     static entry point + supplemented by GitHub reference repo

Run:
    python indexer.py           # build index (saves to index_cache/)
    python indexer.py --force   # force re-scrape even if cache exists
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# ── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "index_cache"
INDEX_FILE = CACHE_DIR / "bm25.pkl"
DOCS_FILE  = CACHE_DIR / "docs.json"

BASE_URL_MANUAL = "https://www.tradingview.com/pine-script-docs"
BASE_URL_REF    = "https://www.tradingview.com/pine-script-reference/v5/"

# GitHub raw reference (plain text, LLM-friendly format)
# This supplements the SPA reference with a pre-built text corpus
GITHUB_REF_URLS = [
    "https://raw.githubusercontent.com/codenamedevan/pinescriptv6/main/pinescript_v6_reference.md",
    "https://raw.githubusercontent.com/pinecoders/pine-script-docs/master/pine_primer.md",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PineScriptDocsMCP/1.0; +https://github.com/)"
}

MANUAL_SECTIONS = [
    # (path, title)
    ("",                          "Pine Script Home"),
    ("language/basics/",          "Language Basics"),
    ("language/execution-model/","Execution Model"),
    ("language/time-series/",     "Time Series"),
    ("language/script-structure/","Script Structure"),
    ("language/identifiers/",     "Identifiers"),
    ("language/operators/",       "Operators"),
    ("language/variable-declarations/","Variable Declarations"),
    ("language/conditional-structures/","Conditional Structures"),
    ("language/loops/",           "Loops"),
    ("language/types/",           "Types"),
    ("language/built-ins/",       "Built-in Variables and Functions"),
    ("language/user-defined-functions/","User-Defined Functions"),
    ("language/methods/",         "Methods"),
    ("language/objects/",         "Objects"),
    ("language/arrays/",          "Arrays"),
    ("language/matrices/",        "Matrices"),
    ("language/maps/",            "Maps"),
    ("language/libraries/",       "Libraries"),
    ("concepts/alerts/",          "Alerts"),
    ("concepts/backgrounds/",     "Backgrounds"),
    ("concepts/bar-coloring/",    "Bar Coloring"),
    ("concepts/bar-plotting/",    "Bar Plotting"),
    ("concepts/bar-states/",      "Bar States"),
    ("concepts/chart-information/","Chart Information"),
    ("concepts/colors/",          "Colors"),
    ("concepts/fills/",           "Fills"),
    ("concepts/inputs/",          "Inputs"),
    ("concepts/levels/",          "Levels"),
    ("concepts/lines-and-boxes/", "Lines and Boxes"),
    ("concepts/non-standard-charts/","Non-Standard Charts"),
    ("concepts/plots/",           "Plots"),
    ("concepts/sessions/",        "Sessions"),
    ("concepts/strategies/",      "Strategies"),
    ("concepts/tables/",          "Tables"),
    ("concepts/text-and-shapes/", "Text and Shapes"),
    ("concepts/timeframes/",      "Timeframes"),
    ("writing-scripts/debugging/","Debugging"),
    ("writing-scripts/limitations/","Limitations"),
    ("writing-scripts/optimization/","Optimization"),
    ("writing-scripts/profiling/","Profiling"),
    ("writing-scripts/publishing/","Publishing Scripts"),
    ("writing-scripts/style-guide/","Style Guide"),
    ("faq/",                      "FAQ"),
    ("migration-guides/",         "Migration Guides"),
    ("release-notes/",            "Release Notes"),
]

# Known Pine Script namespaces for the reference scraper
NAMESPACES = [
    "ta", "math", "strategy", "request", "ticker", "input",
    "color", "plot", "label", "line", "box", "table", "array",
    "matrix", "map", "str", "int", "float", "bool", "string",
    "chart", "syminfo", "timeframe", "session", "dayofweek",
    "alert", "barstate", "currency", "display", "dividends",
    "earnings", "extend", "fill", "font", "format", "hline",
    "indicator", "library", "log", "order", "polyline", "position",
    "scale", "shape", "size", "splits", "text",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_.]+", text.lower())


def fetch(url: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 404:
                return None
            else:
                print(f"  [{r.status_code}] {url}")
                time.sleep(2 ** attempt)
        except requests.RequestException as exc:
            print(f"  [ERR] {url}: {exc}")
            time.sleep(2 ** attempt)
    return None


# ── Scrapers ─────────────────────────────────────────────────────────────────

def scrape_manual_page(path: str, title: str) -> Optional[dict]:
    """Scrape a Pine Script User Manual page (static HTML)."""
    url = f"{BASE_URL_MANUAL}/{path}"
    html = fetch(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "lxml")

    # Remove nav, footer, scripts, styles
    for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    # Extract main content
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(class_=re.compile(r"content|main|docs", re.I)) or
        soup.body
    )

    if not main:
        return None

    # Extract code blocks separately
    code_blocks = []
    for pre in main.find_all("pre"):
        code = pre.get_text()
        if code.strip():
            code_blocks.append(code.strip())
        pre.decompose()  # remove from main text to avoid duplication

    text = clean_text(main.get_text())
    if len(text) < 50:
        return None

    # Extract all headings for sub-section awareness
    headings = [h.get_text().strip() for h in main.find_all(["h1", "h2", "h3", "h4"])]

    return {
        "url":        url,
        "title":      title,
        "text":       text,
        "headings":   headings,
        "code":       code_blocks,
        "source":     "user_manual",
    }


def scrape_reference_spa() -> list[dict]:
    """
    Scrape the Pine Script Reference Manual.
    The reference is a SPA, so we try to:
    1. Extract initial HTML content (function list in script tags / initial state)
    2. Parse the visible text
    3. Supplement with known function patterns
    """
    docs = []
    html = fetch(BASE_URL_REF)
    if not html:
        print("  [WARN] Could not fetch reference SPA entry point")
        return docs

    soup = BeautifulSoup(html, "lxml")

    # Try to extract JSON state from script tags (common SPA pattern)
    for script in soup.find_all("script"):
        src = script.string or ""
        # Look for JSON blobs containing function documentation
        matches = re.findall(r'(\{["\']?name["\']?\s*:\s*["\'][\w.]+["\'].*?\})', src, re.DOTALL)
        for m in matches[:50]:
            try:
                obj = json.loads(m)
                if "name" in obj and "description" in obj:
                    docs.append({
                        "url":    f"{BASE_URL_REF}#fun_{obj['name'].replace('.','.')}",
                        "title":  obj.get("name", ""),
                        "text":   obj.get("description", "") + " " + str(obj.get("params", "")),
                        "code":   [obj.get("example", "")],
                        "source": "reference_spa",
                    })
            except (json.JSONDecodeError, KeyError):
                pass

    # Also extract visible text from the initial render
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    visible = clean_text(soup.get_text())
    if len(visible) > 200:
        docs.append({
            "url":    BASE_URL_REF,
            "title":  "Pine Script v5 Reference Manual",
            "text":   visible[:10000],
            "code":   [],
            "source": "reference_main",
        })

    return docs


def fetch_github_references() -> list[dict]:
    """Fetch plain-text Pine Script references from GitHub repos."""
    docs = []
    for url in GITHUB_REF_URLS:
        print(f"  Fetching GitHub reference: {url}")
        text = fetch(url)
        if not text:
            continue

        # Split markdown into sections by heading
        sections = re.split(r"(?m)^#{1,3} ", text)
        for section in sections:
            if len(section) < 100:
                continue
            lines = section.strip().split("\n")
            title = lines[0].strip() if lines else "Pine Script Reference"
            body  = "\n".join(lines[1:]).strip()
            docs.append({
                "url":    url,
                "title":  title,
                "text":   clean_text(body[:3000]),
                "code":   re.findall(r"```(?:pine|pinescript)?\s*(.*?)```", body, re.DOTALL),
                "source": "github_reference",
            })
        time.sleep(0.5)
    return docs


def build_synthetic_reference() -> list[dict]:
    """
    Build synthetic reference docs for all known Pine Script functions/variables
    using pattern-based URL construction + fetch.
    This supplements what the SPA scraper misses.
    """
    docs = []
    # Common functions per namespace — pre-seeded knowledge
    known_functions = {
        "ta": [
            "sma", "ema", "rma", "wma", "vwma", "dema", "tema", "hma",
            "rsi", "macd", "bbands", "kc", "stoch", "cci", "mfi", "atr",
            "tr", "atr", "highest", "lowest", "highestbars", "lowestbars",
            "cross", "crossover", "crossunder", "rising", "falling", "change",
            "mom", "roc", "correlation", "linreg", "pivot_point_levels",
            "valuewhen", "barssince", "cum", "dev", "variance", "stdev",
            "percentrank", "percentile_linear_interpolation", "supertrend",
            "vwap", "obv", "nvi", "pvi", "pvt",
        ],
        "math": [
            "abs", "ceil", "floor", "round", "max", "min", "pow", "sqrt",
            "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos",
            "atan", "sign", "random", "sum", "avg", "todegrees", "toradians",
        ],
        "strategy": [
            "entry", "exit", "close", "order", "cancel", "cancel_all",
            "risk_max_intraday_loss", "risk_max_intraday_filled_orders",
            "risk_max_cons_loss_days", "risk_max_drawdown",
        ],
        "request": [
            "security", "security_lower_tf", "dividends", "earnings",
            "splits", "quandl", "financial",
        ],
        "input": [
            "bool", "color", "float", "int", "price", "session", "source",
            "string", "symbol", "text_area", "time", "timeframe",
        ],
        "array": [
            "new", "from", "push", "pop", "shift", "unshift", "get", "set",
            "remove", "insert", "size", "slice", "join", "concat", "copy",
            "includes", "indexof", "sort", "sort_indices", "reverse", "clear",
            "sum", "min", "max", "avg", "variance", "stdev", "median", "mode",
            "range", "percentile_linear_interpolation", "percentrank",
            "fill", "new_bool", "new_float", "new_int", "new_string",
        ],
        "str": [
            "tostring", "tonumber", "length", "contains", "startswith",
            "endswith", "replace", "replace_all", "split", "lower", "upper",
            "format", "format_time", "substring", "trim", "match",
        ],
        "color": [
            "new", "r", "g", "b", "t", "from_gradient", "rgb",
        ],
        "line": [
            "new", "delete", "set_x1", "set_x2", "set_y1", "set_y2",
            "set_xy1", "set_xy2", "set_color", "set_style", "set_width",
            "set_extend", "get_x1", "get_x2", "get_y1", "get_y2",
            "get_price", "copy", "all",
        ],
        "label": [
            "new", "delete", "set_x", "set_y", "set_xy", "set_text",
            "set_color", "set_textcolor", "set_style", "set_size",
            "set_tooltip", "get_x", "get_y", "get_text", "copy", "all",
        ],
        "table": [
            "new", "delete", "cell", "cell_set_text", "cell_set_bgcolor",
            "cell_set_text_color", "cell_set_text_size", "merge_cells",
            "set_bgcolor", "set_border_color", "set_border_width",
            "set_frame_color", "set_frame_width", "set_position", "all",
        ],
        "box": [
            "new", "delete", "set_top", "set_bottom", "set_left", "set_right",
            "set_border_color", "set_border_width", "set_border_style",
            "set_bgcolor", "set_extend", "set_text", "set_text_color",
            "get_top", "get_bottom", "get_left", "get_right", "copy", "all",
        ],
    }

    # Rich signatures for the most commonly used functions
    rich_signatures = {
        "ta.sma":       ("ta.sma(source, length) -> series float", "Simple moving average of source over length bars."),
        "ta.ema":       ("ta.ema(source, length) -> series float", "Exponential moving average. More weight on recent values."),
        "ta.rma":       ("ta.rma(source, length) -> series float", "Rolling moving average (Wilder's MA). Used in RSI and ATR."),
        "ta.wma":       ("ta.wma(source, length) -> series float", "Weighted moving average — linearly weighted, recent bars have more weight."),
        "ta.vwma":      ("ta.vwma(source, length) -> series float", "Volume-weighted moving average."),
        "ta.rsi":       ("ta.rsi(source, length) -> series float", "Relative Strength Index. Returns value 0-100."),
        "ta.macd":      ("ta.macd(source, fast_length, slow_length, signal_smoothing) -> [macd, signal, hist]", "MACD indicator. Returns tuple: [macd line, signal line, histogram]."),
        "ta.bbands":    ("[upper, basis, lower] = ta.bb(source, length, mult) -> [float, float, float]", "Bollinger Bands. Returns upper band, basis (SMA), and lower band."),
        "ta.kc":        ("[upper, lower] = ta.kc(source, length, mult, use_true_range) -> [float, float]", "Keltner Channels. Returns upper and lower bands."),
        "ta.atr":       ("ta.atr(length) -> series float", "Average True Range over length bars."),
        "ta.tr":        ("ta.tr(handle_na) -> series float", "True Range. handle_na=true replaces na with 0."),
        "ta.stoch":     ("ta.stoch(source, high, low, length) -> series float", "Stochastic oscillator value 0-100."),
        "ta.cci":       ("ta.cci(source, length) -> series float", "Commodity Channel Index."),
        "ta.mfi":       ("ta.mfi(source, length) -> series float", "Money Flow Index (0-100)."),
        "ta.supertrend":("ta.supertrend(factor, atr_period) -> [supertrend, direction]", "SuperTrend indicator. direction: 1=uptrend, -1=downtrend."),
        "ta.vwap":      ("ta.vwap(source, anchor, stdev_mult) -> series float", "Volume Weighted Average Price. Resets at anchor period."),
        "ta.crossover": ("ta.crossover(source1, source2) -> series bool", "True on the bar where source1 crosses above source2."),
        "ta.crossunder":("ta.crossunder(source1, source2) -> series bool", "True on the bar where source1 crosses below source2."),
        "ta.cross":     ("ta.cross(source1, source2) -> series bool", "True when source1 and source2 cross (either direction)."),
        "ta.highest":   ("ta.highest(source, length) -> series float", "Highest value of source in the last length bars."),
        "ta.lowest":    ("ta.lowest(source, length) -> series float", "Lowest value of source in the last length bars."),
        "ta.highestbars":("ta.highestbars(source, length) -> series int", "Number of bars since highest value. Returns 0 if current bar is highest."),
        "ta.lowestbars":("ta.lowestbars(source, length) -> series int", "Number of bars since lowest value."),
        "ta.barssince": ("ta.barssince(condition) -> series int", "Number of bars since condition was last true. Returns na if never true."),
        "ta.valuewhen": ("ta.valuewhen(condition, source, occurrence) -> series float", "Value of source on the Nth most recent bar where condition was true."),
        "ta.rising":    ("ta.rising(source, length) -> series bool", "True if source has been rising for length bars consecutively."),
        "ta.falling":   ("ta.falling(source, length) -> series bool", "True if source has been falling for length bars consecutively."),
        "ta.change":    ("ta.change(source, length) -> series float", "Difference between current value and length bars ago. Default length=1."),
        "ta.mom":       ("ta.mom(source, length) -> series float", "Momentum: source - source[length]."),
        "ta.roc":       ("ta.roc(source, length) -> series float", "Rate of change as percentage: (source - source[length]) / source[length] * 100."),
        "ta.linreg":    ("ta.linreg(source, length, offset) -> series float", "Linear regression value. offset=0 gives current bar's regression value."),
        "ta.correlation":("ta.correlation(source1, source2, length) -> series float", "Pearson correlation coefficient between source1 and source2 over length bars."),
        "ta.stdev":     ("ta.stdev(source, length, biased) -> series float", "Standard deviation of source over length bars."),
        "ta.variance":  ("ta.variance(source, length, biased) -> series float", "Variance of source over length bars."),
        "ta.dev":       ("ta.dev(source, length) -> series float", "Mean absolute deviation."),
        "ta.cum":       ("ta.cum(source) -> series float", "Cumulative sum of source from the first bar."),
        "ta.sum":       ("ta.sum(source, length) -> series float", "Rolling sum of source over length bars."),
        "ta.pivot_point_levels":("ta.pivot_point_levels(type, anchor) -> float[]", "Returns array of pivot point levels. type: 'Traditional','Fibonacci','Woodie','Classic','DM','Camarilla'."),
        "ta.percentrank":("ta.percentrank(source, length) -> series float", "Percent rank of current value within last length values (0-100)."),
        "ta.obv":       ("ta.obv -> series float", "On Balance Volume. Built-in series, no function call needed."),
        "strategy.entry":("strategy.entry(id, direction, qty, limit, stop, oca_name, oca_type, comment, alert_message, disable_alert)", "Place a market, limit, or stop entry order. direction: strategy.long or strategy.short."),
        "strategy.exit":("strategy.exit(id, from_entry, qty, qty_percent, profit, limit, loss, stop, trail_price, trail_offset, oca_name, comment, alert_message, disable_alert)", "Exit an open position or order. Set profit/loss in ticks or limit/stop in price."),
        "strategy.close":("strategy.close(id, comment, qty, qty_percent, alert_message, disable_alert, immediately)", "Close a position opened with strategy.entry(id)."),
        "strategy.close_all":("strategy.close_all(comment, alert_message, disable_alert, immediately)", "Close all open positions."),
        "strategy.order":("strategy.order(id, direction, qty, limit, stop, oca_name, oca_type, comment, alert_message, disable_alert)", "Place any order type (entry or exit) unconditionally."),
        "request.security":("request.security(symbol, timeframe, expression, gaps, lookahead, ignore_invalid_symbol, currency, calc_bars_count) -> series", "Fetch data from another symbol or timeframe. IMPORTANT: lookahead=barmerge.lookahead_off by default (no future leak)."),
        "request.security_lower_tf":("request.security_lower_tf(symbol, timeframe, expression, ignore_invalid_symbol, currency, ignore_invalid_timeframe, calc_bars_count) -> array", "Returns array of values from a lower timeframe (one element per lower-TF bar within the current bar)."),
        "input.float":  ("input.float(defval, title, minval, maxval, step, tooltip, inline, group, display, confirm, options) -> input float", "Create a float input widget in the indicator settings panel."),
        "input.int":    ("input.int(defval, title, minval, maxval, step, tooltip, inline, group, display, confirm, options) -> input int", "Create an integer input widget."),
        "input.bool":   ("input.bool(defval, title, tooltip, inline, group, display, confirm) -> input bool", "Create a boolean checkbox input."),
        "input.string": ("input.string(defval, title, options, tooltip, inline, group, display, confirm) -> input string", "Create a string dropdown or text input."),
        "input.color":  ("input.color(defval, title, tooltip, inline, group, display, confirm) -> input color", "Create a color picker input."),
        "input.source": ("input.source(defval, title, tooltip, inline, group, display) -> series float", "Create a source selector (close, open, hl2, etc.)."),
        "input.symbol": ("input.symbol(defval, title, tooltip, inline, group, display, confirm) -> input string", "Create a symbol search input."),
        "input.timeframe":("input.timeframe(defval, title, options, tooltip, inline, group, display, confirm) -> input string", "Create a timeframe selector."),
        "array.new":    ("array.new<type>(size, initial_value) -> array<type>", "Create a new typed array. E.g. array.new<float>(10, 0.0)"),
        "array.from":   ("array.from(value1, value2, ...) -> array", "Create array from literal values. E.g. array.from(1.0, 2.0, 3.0)"),
        "array.push":   ("array.push(id, value) -> void", "Append value to the end of the array."),
        "array.pop":    ("array.pop(id) -> value", "Remove and return the last element."),
        "array.get":    ("array.get(id, index) -> value", "Get value at index. Negative index counts from end."),
        "array.set":    ("array.set(id, index, value) -> void", "Set value at index."),
        "array.size":   ("array.size(id) -> series int", "Number of elements in the array."),
        "array.sort":   ("array.sort(id, order) -> void", "Sort in place. order: order.ascending or order.descending."),
        "array.sum":    ("array.sum(id) -> series float", "Sum of all array elements."),
        "array.avg":    ("array.avg(id) -> series float", "Mean of all array elements."),
        "array.min":    ("array.min(id) -> series float", "Minimum value in the array."),
        "array.max":    ("array.max(id) -> series float", "Maximum value in the array."),
        "color.new":    ("color.new(color, transp) -> color", "Create color with transparency. transp: 0=opaque, 100=invisible."),
        "color.rgb":    ("color.rgb(r, g, b, transp) -> color", "Create color from RGB components (0-255) with optional transparency."),
        "color.from_gradient":("color.from_gradient(value, bottom_value, top_value, bottom_color, top_color) -> color", "Interpolate color between two values."),
        "line.new":     ("line.new(x1, y1, x2, y2, xloc, extend, color, style, width) -> line", "Create a new line object. xloc: xloc.bar_index (default) or xloc.bar_time."),
        "label.new":    ("label.new(x, y, text, xloc, yloc, color, style, textcolor, size, textalign, tooltip, text_font_family) -> label", "Create a new label object."),
        "box.new":      ("box.new(left, top, right, bottom, border_color, border_width, border_style, extend, xloc, bgcolor, text, text_size, text_color, text_halign, text_valign, text_wrap, text_font_family) -> box", "Create a new box object."),
        "table.new":    ("table.new(position, columns, rows, bgcolor, frame_color, frame_width, border_color, border_width) -> table", "Create a new table. position: position.top_left, position.top_right, etc."),
        "table.cell":   ("table.cell(table_id, column, row, text, width, height, text_color, text_halign, text_valign, text_size, bgcolor, tooltip, text_font_family) -> void", "Set cell content and styling."),
        "math.abs":     ("math.abs(number) -> int/float", "Absolute value."),
        "math.max":     ("math.max(val1, val2, ...) -> int/float", "Maximum of multiple values or a series."),
        "math.min":     ("math.min(val1, val2, ...) -> int/float", "Minimum of multiple values or a series."),
        "math.pow":     ("math.pow(base, exponent) -> float", "Raise base to the power of exponent."),
        "math.sqrt":    ("math.sqrt(number) -> float", "Square root."),
        "math.round":   ("math.round(number, precision) -> float", "Round to precision decimal places."),
        "math.floor":   ("math.floor(number) -> int", "Round down to nearest integer."),
        "math.ceil":    ("math.ceil(number) -> int", "Round up to nearest integer."),
        "math.log":     ("math.log(number) -> float", "Natural logarithm."),
        "math.log10":   ("math.log10(number) -> float", "Base-10 logarithm."),
        "math.sin":     ("math.sin(angle_radians) -> float", "Sine of angle in radians."),
        "math.cos":     ("math.cos(angle_radians) -> float", "Cosine of angle in radians."),
        "math.random":  ("math.random(min, max, seed) -> float", "Random float in [min, max)."),
        "str.tostring": ("str.tostring(value, format) -> string", "Convert to string. format: '#.##', 'HH:mm', etc."),
        "str.format":   ("str.format(formatString, arg0, arg1, ...) -> string", "Format string with placeholders. E.g. str.format('{0} crossed {1}', close, ta.sma(close,20))"),
        "str.length":   ("str.length(string) -> int", "Number of characters in string."),
        "str.contains": ("str.contains(source, str) -> bool", "True if source contains str."),
        "str.split":    ("str.split(source, separator) -> string[]", "Split string into array by separator."),
        "str.lower":    ("str.lower(source) -> string", "Convert to lowercase."),
        "str.upper":    ("str.upper(source) -> string", "Convert to uppercase."),
        "str.replace":  ("str.replace(source, target, replacement, occurrence) -> string", "Replace occurrence of target in source with replacement."),
        "str.format_time":("str.format_time(time, format, timezone) -> string", "Format Unix ms timestamp. E.g. str.format_time(time, 'yyyy-MM-dd HH:mm', 'UTC')"),
    }

    for ns, funcs in known_functions.items():
        for fn in funcs:
            full_name = f"{ns}.{fn}"
            if full_name in rich_signatures:
                sig, desc = rich_signatures[full_name]
                text = f"Pine Script function: {full_name}\nSignature: {sig}\nDescription: {desc}"
            else:
                text = f"Pine Script function: {full_name}. Namespace: {ns}. See: {BASE_URL_REF}#fun_{full_name}"
            docs.append({
                "url":    f"{BASE_URL_REF}#fun_{full_name}",
                "title":  full_name,
                "text":   text,
                "code":   [],
                "source": "synthetic_reference",
            })

    # Key built-in variables
    builtins = [
        ("open", "Current bar's opening price"),
        ("high", "Current bar's highest price"),
        ("low",  "Current bar's lowest price"),
        ("close","Current bar's closing price"),
        ("volume","Current bar's volume"),
        ("hl2",  "Average of high and low"),
        ("hlc3", "Average of high, low, and close"),
        ("ohlc4","Average of open, high, low, and close"),
        ("hlcc4","Average of high, low, and close*2"),
        ("bar_index","Zero-based index of current bar"),
        ("time", "Time of current bar open in Unix milliseconds"),
        ("time_close","Time of current bar close"),
        ("timenow","Current time in Unix milliseconds"),
        ("syminfo.ticker","Symbol ticker (no exchange prefix)"),
        ("syminfo.tickerid","Full ticker with exchange prefix"),
        ("syminfo.prefix","Exchange prefix"),
        ("syminfo.currency","Symbol currency"),
        ("syminfo.mintick","Minimum tick size"),
        ("syminfo.pointvalue","Point value"),
        ("syminfo.timezone","Timezone of exchange"),
        ("syminfo.type","Instrument type: stock, futures, forex, crypto"),
        ("barstate.isfirst","True on the first bar"),
        ("barstate.islast","True on the last bar"),
        ("barstate.isrealtime","True on a real-time bar"),
        ("barstate.ishistory","True on a historical bar"),
        ("barstate.isconfirmed","True when bar is confirmed (closed)"),
        ("barstate.isnew","True on a new bar"),
        ("timeframe.period","Current chart timeframe string"),
        ("timeframe.multiplier","Current timeframe multiplier"),
        ("timeframe.isdaily","True if daily timeframe"),
        ("timeframe.isweekly","True if weekly timeframe"),
        ("timeframe.ismonthly","True if monthly timeframe"),
        ("timeframe.isintraday","True if intraday timeframe"),
        ("dayofweek","Day of week of bar's open time (1=Sunday)"),
        ("dayofmonth","Day of month"),
        ("month","Month number"),
        ("year","Year"),
        ("hour","Hour of the bar"),
        ("minute","Minute of the bar"),
        ("second","Second of the bar"),
        ("na",   "Not a number (missing value)"),
        ("true", "Boolean true"),
        ("false","Boolean false"),
    ]

    for var_name, desc in builtins:
        anchor = f"var_{var_name.replace('.', '.')}"
        docs.append({
            "url":    f"{BASE_URL_REF}#{anchor}",
            "title":  var_name,
            "text":   f"Pine Script built-in variable: {var_name}. {desc}.",
            "code":   [],
            "source": "builtin_variable",
        })

    return docs


# ── Main build function ───────────────────────────────────────────────────────

def build_index(force: bool = False) -> tuple[BM25Okapi, list[dict]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and INDEX_FILE.exists() and DOCS_FILE.exists():
        print("[indexer] Loading cached index...")
        with open(INDEX_FILE, "rb") as f:
            bm25 = pickle.load(f)
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"[indexer] Loaded {len(docs)} documents from cache.")
        return bm25, docs

    print("[indexer] Building fresh index...")
    all_docs: list[dict] = []

    # 1. Scrape User Manual pages
    print(f"\n[1/5] Scraping User Manual ({len(MANUAL_SECTIONS)} sections)...")
    for i, (path, title) in enumerate(MANUAL_SECTIONS, 1):
        print(f"  [{i:02d}/{len(MANUAL_SECTIONS)}] {title}")
        doc = scrape_manual_page(path, title)
        if doc:
            all_docs.append(doc)
        time.sleep(0.5)

    # 2. SPA Reference main page
    print("\n[2/5] Scraping Reference SPA entry point...")
    spa_docs = scrape_reference_spa()
    all_docs.extend(spa_docs)
    print(f"  Got {len(spa_docs)} docs from SPA")

    # 3. GitHub reference repos
    print("\n[3/5] Fetching GitHub reference sources...")
    gh_docs = fetch_github_references()
    all_docs.extend(gh_docs)
    print(f"  Got {len(gh_docs)} sections from GitHub")

    # 4. Playwright SPA reference (full function signatures from rendered DOM)
    print("\n[4/5] Running Playwright SPA scraper...")
    playwright_cache = CACHE_DIR / "reference_scraped.json"
    try:
        from scraper_playwright import run as playwright_run
        pw_docs = playwright_run(out=playwright_cache)
        if pw_docs:
            # Deduplicate against already-seen titles
            existing_titles = {d["title"] for d in all_docs}
            pw_new = [d for d in pw_docs if d["title"] not in existing_titles]
            all_docs.extend(pw_new)
            print(f"  Got {len(pw_docs)} Playwright docs, {len(pw_new)} new after dedup")
        else:
            print("  [WARN] Playwright returned 0 docs")
    except ImportError:
        print("  [SKIP] scraper_playwright.py not found")
    except Exception as e:
        print(f"  [WARN] Playwright scraper error: {e}")
        # Try loading cached output if scraper fails
        if playwright_cache.exists():
            with open(playwright_cache, "r", encoding="utf-8") as f:
                pw_docs = json.load(f)
            existing_titles = {d["title"] for d in all_docs}
            pw_new = [d for d in pw_docs if d["title"] not in existing_titles]
            all_docs.extend(pw_new)
            print(f"  Loaded {len(pw_new)} docs from Playwright cache")

    # 5. Synthetic reference for known functions
    print("\n[5/5] Building synthetic function reference...")
    syn_docs = build_synthetic_reference()
    # Only add synthetic entries not already covered by Playwright
    existing_titles = {d["title"] for d in all_docs}
    syn_new = [d for d in syn_docs if d["title"] not in existing_titles]
    all_docs.extend(syn_new)
    print(f"  Built {len(syn_docs)} synthetic entries, {len(syn_new)} new after dedup")

    # Filter empty docs
    all_docs = [d for d in all_docs if d.get("text") and len(d["text"]) > 20]

    print(f"\n[indexer] Total documents: {len(all_docs)}")

    # Build BM25
    print("[indexer] Building BM25 index...")
    corpus = []
    for doc in all_docs:
        tokens = tokenise(doc["title"] + " " + doc["text"])
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)

    # Save
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    print(f"[indexer] Saved index ({INDEX_FILE.stat().st_size // 1024} KB)")
    return bm25, all_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TradingView Pine Script docs index")
    parser.add_argument("--force", action="store_true", help="Force re-scrape even if cache exists")
    args = parser.parse_args()

    bm25, docs = build_index(force=args.force)
    print(f"\n[done] Index ready: {len(docs)} docs indexed.")
    print(f"  Cache: {DOCS_FILE}")
    print(f"  Run server: python server.py")
