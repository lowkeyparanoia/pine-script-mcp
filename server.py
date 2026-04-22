"""
server.py - TradingView Pine Script Documentation MCP Server
=============================================================
FastMCP server that exposes Pine Script docs to any MCP-compatible LLM.

Tools:
  search_pine_docs   - BM25 search across all indexed documentation
  get_pine_function  - Look up a specific function or variable by name
  get_pine_page      - Get full content of a docs page by URL path
  list_pine_sections - List all indexed sections with their URLs
  pine_quick_ref     - Get a quick reference card for a Pine Script topic

Usage:
    python server.py          # start MCP server (stdio transport)
    python server.py --http   # start HTTP transport on port 8080

First run:
    python indexer.py         # build the docs index (one-time setup)
"""

import json
import sys
import re
import os
from pathlib import Path
from typing import Optional

# ── Lazy-load heavy deps ──────────────────────────────────────────────────────
try:
    import pickle
    from rank_bm25 import BM25Okapi
    _HAVE_BM25 = True
except ImportError:
    _HAVE_BM25 = False

try:
    from fastmcp import FastMCP
except ImportError:
    print("ERROR: fastmcp not installed. Run: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  = Path(__file__).parent / "index_cache"
INDEX_FILE = CACHE_DIR / "bm25.pkl"
DOCS_FILE  = CACHE_DIR / "docs.json"

# ── Global state ──────────────────────────────────────────────────────────────
_bm25: Optional[object] = None
_docs: list[dict]       = []
_loaded                 = False

def _ensure_loaded():
    global _bm25, _docs, _loaded
    if _loaded:
        return

    if not INDEX_FILE.exists() or not DOCS_FILE.exists():
        _loaded = True
        return  # will be caught in tool handlers

    with open(INDEX_FILE, "rb") as f:
        _bm25 = pickle.load(f)
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        _docs = json.load(f)
    _loaded = True


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_.]+", text.lower())


def _format_doc(doc: dict, include_code: bool = True, max_text: int = 1200) -> str:
    parts = []
    parts.append(f"# {doc.get('title', 'Untitled')}")
    parts.append(f"URL: {doc.get('url', '')}")
    parts.append(f"Source: {doc.get('source', 'unknown')}")
    parts.append("")
    text = doc.get("text", "")
    if len(text) > max_text:
        text = text[:max_text] + "... [truncated]"
    parts.append(text)
    if include_code and doc.get("code"):
        parts.append("\n**Code Examples:**")
        for i, code in enumerate(doc["code"][:3], 1):
            if code.strip():
                parts.append(f"```pine\n{code.strip()[:500]}\n```")
    headings = doc.get("headings", [])
    if headings:
        parts.append(f"\n**Sub-sections:** {', '.join(headings[:8])}")
    return "\n".join(parts)


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("tradingview-pine-docs")


@mcp.tool()
def search_pine_docs(query: str, max_results: int = 8) -> str:
    """
    Search TradingView Pine Script documentation using full-text BM25 search.

    Args:
        query:       Search query. Examples: "ta.sma moving average", "strategy entry",
                     "array push", "plot color", "input.float", "barstate.islast"
        max_results: Maximum results to return (default: 8, max: 20)

    Returns:
        Ranked list of matching documentation sections with content excerpts.
    """
    _ensure_loaded()
    if not _docs:
        return (
            "ERROR: Documentation index not found.\n"
            "Run: python indexer.py\nto build the index first."
        )

    max_results = min(max_results, 20)
    tokens = _tokenise(query)
    if not tokens:
        return "ERROR: Empty query."

    scores = _bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top    = [(i, s) for i, s in ranked if s > 0][:max_results]

    if not top:
        return f"No results found for: '{query}'\nTry different keywords."

    results = [f"Search results for: '{query}' ({len(top)} found)\n{'='*50}\n"]
    for rank, (idx, score) in enumerate(top, 1):
        doc = _docs[idx]
        snippet = doc.get("text", "")[:300].replace("\n", " ")
        results.append(
            f"**[{rank}] {doc.get('title', 'Untitled')}**\n"
            f"URL: {doc.get('url', '')}\n"
            f"Score: {score:.2f} | Source: {doc.get('source', '')}\n"
            f"{snippet}...\n"
        )

    return "\n".join(results)


@mcp.tool()
def get_pine_function(function_name: str) -> str:
    """
    Get full documentation for a specific Pine Script function or variable.

    Args:
        function_name: Function or variable name. Examples:
                       "ta.sma", "strategy.entry", "array.push",
                       "input.float", "close", "barstate.islast",
                       "request.security", "color.new", "line.new"

    Returns:
        Full documentation including signature, description, parameters, and examples.
    """
    _ensure_loaded()
    if not _docs:
        return "ERROR: Index not found. Run: python indexer.py"

    name_lower = function_name.lower().strip()

    # Exact title match first
    for doc in _docs:
        if doc.get("title", "").lower() == name_lower:
            return _format_doc(doc, include_code=True, max_text=3000)

    # Partial match in title
    candidates = [d for d in _docs if name_lower in d.get("title", "").lower()]
    if candidates:
        return _format_doc(candidates[0], include_code=True, max_text=3000)

    # Fall back to BM25 search
    tokens = _tokenise(function_name)
    scores = _bm25.get_scores(tokens)
    best_idx = int(scores.argmax())
    if scores[best_idx] > 0:
        return _format_doc(_docs[best_idx], include_code=True, max_text=3000)

    return (
        f"No documentation found for '{function_name}'.\n"
        f"Check the TradingView reference: "
        f"https://www.tradingview.com/pine-script-reference/v5/#fun_{function_name}"
    )


@mcp.tool()
def get_pine_page(section: str) -> str:
    """
    Get the full content of a Pine Script User Manual section.

    Args:
        section: Section path or keyword. Examples:
                 "language/types", "concepts/strategies", "language/arrays",
                 "concepts/plots", "writing-scripts/debugging", "faq"

    Returns:
        Full page content including all sub-sections and code examples.
    """
    _ensure_loaded()
    if not _docs:
        return "ERROR: Index not found. Run: python indexer.py"

    section_lower = section.lower().strip().strip("/")

    # Match by URL path
    for doc in _docs:
        url_path = doc.get("url", "").lower()
        if section_lower in url_path and doc.get("source") == "user_manual":
            return _format_doc(doc, include_code=True, max_text=5000)

    # Match by title
    for doc in _docs:
        if section_lower in doc.get("title", "").lower():
            return _format_doc(doc, include_code=True, max_text=5000)

    # BM25 fallback limited to user_manual
    manual_docs = [(i, d) for i, d in enumerate(_docs) if d.get("source") == "user_manual"]
    if not manual_docs:
        return f"Section '{section}' not found."

    tokens = _tokenise(section)
    scores = _bm25.get_scores(tokens)
    best   = max(manual_docs, key=lambda x: scores[x[0]])
    if scores[best[0]] > 0:
        return _format_doc(best[1], include_code=True, max_text=5000)

    return (
        f"Section '{section}' not found.\n"
        f"Try: search_pine_docs('{section}')\n"
        f"Or browse: https://www.tradingview.com/pine-script-docs/{section}/"
    )


@mcp.tool()
def list_pine_sections() -> str:
    """
    List all indexed Pine Script documentation sections with their URLs.

    Returns:
        Categorized list of all available documentation sections.
    """
    _ensure_loaded()
    if not _docs:
        return "ERROR: Index not found. Run: python indexer.py"

    categories: dict[str, list[str]] = {}
    for doc in _docs:
        source = doc.get("source", "other")
        title  = doc.get("title", "Untitled")
        url    = doc.get("url", "")
        cat    = {
            "user_manual":       "User Manual",
            "reference_spa":     "Reference (SPA)",
            "reference_main":    "Reference Main",
            "github_reference":  "GitHub Reference",
            "synthetic_reference":"Function Reference",
            "builtin_variable":  "Built-in Variables",
        }.get(source, source)

        categories.setdefault(cat, [])
        categories[cat].append(f"  - {title}  ({url})")

    lines = [f"Pine Script Documentation Index — {len(_docs)} total documents\n"]
    for cat, items in sorted(categories.items()):
        lines.append(f"\n## {cat} ({len(items)} items)")
        lines.extend(items[:30])
        if len(items) > 30:
            lines.append(f"  ... and {len(items) - 30} more")

    return "\n".join(lines)


@mcp.tool()
def pine_quick_ref(topic: str) -> str:
    """
    Get a Pine Script quick reference card for a topic.

    Args:
        topic: Topic keyword. Examples:
               "indicators", "strategy", "plotting", "arrays", "inputs",
               "operators", "time", "types", "loops", "functions", "colors"

    Returns:
        Concise cheat-sheet style reference for the topic.
    """
    quick_refs = {
        "operators": """
## Pine Script Operators
**Arithmetic:** + - * / % (modulo)
**Comparison:** == != > < >= <=
**Logical:** and, or, not
**Ternary:** condition ? value_if_true : value_if_false
**Assignment:** =, :=  (note: := reassigns a previously declared variable)
**Type cast:** int(x), float(x), bool(x), str.tostring(x)
**Series operator:** [] (e.g., close[1] = previous bar's close)
""",
        "types": """
## Pine Script Types
int     — integer: 42, -10
float   — decimal: 3.14, na
bool    — true / false
string  — "hello"
color   — color.red, color.new(#FF0000, 50)
line    — line.new()
label   — label.new()
box     — box.new()
table   — table.new()
array   — array.new<float>() or array.from(1,2,3)
matrix  — matrix.new<float>(rows, cols)
map     — map.new<string, float>()
""",
        "indicators": """
## Common Technical Indicators
ta.sma(source, length)          — Simple Moving Average
ta.ema(source, length)          — Exponential Moving Average
ta.wma(source, length)          — Weighted Moving Average
ta.rma(source, length)          — Rolling Moving Average (used in RSI)
ta.vwma(source, length)         — Volume Weighted MA
ta.rsi(source, length)          — RSI (0-100)
ta.macd(source, fast, slow, sig)— MACD [macd, signal, hist]
[upper, mid, lower] = ta.bb(source, length, mult)  — Bollinger Bands
[upper, lower] = ta.kc(source, length, mult)        — Keltner Channel
ta.atr(length)                  — Average True Range
ta.tr(handle_na)                — True Range
ta.stoch(source, high, low, length) — Stochastic
ta.cci(source, length)          — CCI
ta.mfi(source, length)          — Money Flow Index
ta.obv                          — On Balance Volume
ta.vwap                         — VWAP (daily reset)
ta.supertrend(factor, length)   — SuperTrend [supertrend, direction]
ta.pivot_point_levels(type, anchor) — Pivot Points
""",
        "plotting": """
## Plotting
plot(series, "title", color, linewidth, style, display)
plotshape(series, "title", shape.circle, location.belowbar, color)
plotarrow(series, "title", colorup, colordown)
plotbar(open, high, low, close, "title", color)
plotcandle(open, high, low, close, "title", color, wickcolor)
bgcolor(color, offset, title)
hline(price, "title", color, linestyle, linewidth)
fill(plot1, plot2, color, title)

// Styles
plot.style_line, plot.style_area, plot.style_histogram
plot.style_circles, plot.style_cross, plot.style_stepline
plot.style_columns, plot.style_linebr (break on na)

// Shapes
shape.circle, shape.square, shape.diamond, shape.triangleup/down
shape.arrowup/down, shape.flag, shape.xcross, shape.cross, shape.labelup/down
""",
        "strategy": """
## Strategy Functions
strategy(title, overlay, initial_capital, currency, default_qty_type,
         default_qty_value, commission_type, commission_value, pyramiding)

strategy.entry(id, direction, qty, limit, stop, comment)
  direction: strategy.long | strategy.short

strategy.exit(id, from_entry, qty, limit, stop, trail_price,
              trail_offset, profit, loss, comment)

strategy.close(id, comment, qty)
strategy.close_all(comment)
strategy.order(id, direction, qty, limit, stop)
strategy.cancel(id)
strategy.cancel_all()

// Position info
strategy.position_size          — current position size (+long, -short)
strategy.position_avg_price     — average entry price
strategy.netprofit              — net profit
strategy.grossprofit / grossloss
strategy.wintrades / losstrades / eventrades
strategy.max_drawdown
""",
        "time": """
## Time & Date
time                            — bar open time (Unix ms)
time_close                      — bar close time (Unix ms)
timenow                         — current time (Unix ms)
year, month, dayofmonth         — date components
hour, minute, second            — time components
dayofweek                       — 1=Sunday, 7=Saturday

// Useful functions
time(timeframe, session, timezone) — time of bar open in given TF
timestamp(year, month, day, hour, minute, second) — Unix ms
str.format_time(unix_ms, format, timezone)
weekofyear(time)

// Example: check if bar is in session
in_session = not na(time(timeframe.period, "0930-1600", "America/New_York"))

// J2000 epoch (for astronomy): 946728000000 ms
""",
        "arrays": """
## Arrays
a = array.new<float>(size, initial_value)
a = array.from(1.0, 2.0, 3.0)

array.push(a, value)            — append to end
array.pop(a)                    — remove & return last
array.shift(a)                  — remove & return first
array.unshift(a, value)         — prepend
array.insert(a, index, value)
array.remove(a, index)
array.get(a, index)
array.set(a, index, value)
array.size(a)
array.slice(a, start, end)
array.copy(a)
array.sort(a, order.ascending)

// Stats
array.sum(a), array.avg(a), array.min(a), array.max(a)
array.stdev(a), array.variance(a), array.median(a)
""",
        "inputs": """
## Inputs
input.bool(defval, title, tooltip)
input.int(defval, title, minval, maxval, step, options)
input.float(defval, title, minval, maxval, step, options)
input.string(defval, title, options)
input.color(defval, title)
input.source(defval, title)     — e.g., input.source(close, "Source")
input.symbol(defval, title)
input.timeframe(defval, title)
input.session(defval, title)
input.time(defval, title)
input.text_area(defval, title)
input.price(defval, title)

// Group inputs
input.float(1.0, "My Input", group="Settings", inline="row1")
""",
        "colors": """
## Colors
// Built-in colors
color.red, color.green, color.blue, color.white, color.black
color.yellow, color.orange, color.purple, color.gray, color.navy
color.lime, color.aqua, color.fuchsia, color.silver, color.maroon
color.teal, color.olive

// Transparency: 0=opaque, 100=invisible
color.new(base_color, transparency)     — e.g., color.new(color.red, 50)
color.rgb(r, g, b, transparency)        — from components 0-255
color.from_gradient(value, bottom_val, top_val, bottom_color, top_color)

// Get components
color.r(c), color.g(c), color.b(c), color.t(c)
""",
        "loops": """
## Loops in Pine Script
// for loop
for i = 0 to 9
    // runs 10 times (0,1,...,9)
    array.push(a, i)

for i = 10 to 1 by -1
    // countdown

// for-in (iterate array)
for val in myArray
    sum += val

for [i, val] in myArray   // with index
    if val > 0
        count += 1

// while loop
i = 0
while i < 10
    i += 1

// break / continue work inside loops

// NOTE: loops execute on every bar — keep them short
// Max iterations: 500 per loop call
""",
        "functions": """
## User-Defined Functions
// Single-line
f(x) => x * x

// Multi-line
myFunc(source, length) =>
    s = ta.sma(source, length)
    d = source - s
    [s, d]  // return tuple

// Call
[avg, dev] = myFunc(close, 20)

// Methods (attached to types, v5+)
type MyPoint
    float x
    float y

MyPoint.distance(this, other) =>
    math.sqrt(math.pow(this.x - other.x, 2) + math.pow(this.y - other.y, 2))

p1 = MyPoint.new(0.0, 0.0)
p2 = MyPoint.new(3.0, 4.0)
dist = p1.distance(p2)  // 5.0

// Functions are series-aware — they see the same bar as the calling scope
// Recursive functions NOT supported
""",
        "request": """
## request.security — Multi-timeframe & Symbol Data
// Basic usage
htf_close = request.security(syminfo.tickerid, "D", close)
weekly_high = request.security(syminfo.tickerid, "W", high)

// Different symbol
spy_close = request.security("AMEX:SPY", "D", close)

// IMPORTANT: lookahead parameter
// lookahead=barmerge.lookahead_off  (DEFAULT, no future leak)
// lookahead=barmerge.lookahead_on   (uses future data — only for non-repainting calcs)

// gaps parameter
// gaps=barmerge.gaps_off  (fill gaps with last known value — DEFAULT)
// gaps=barmerge.gaps_on   (use na for missing bars)

// Expression (calculate ON the security's bars)
htf_rsi = request.security(syminfo.tickerid, "D", ta.rsi(close, 14))

// Tuple
[htf_open, htf_close] = request.security(syminfo.tickerid, "D", [open, close])

// Lower timeframe (get all intraday bars within current bar)
// request.security_lower_tf(symbol, timeframe, expression)
m1_closes = request.security_lower_tf(syminfo.tickerid, "1", close)
// Returns array of values (one per lower-TF bar within current bar)

// COMMON PITFALL: calling request.security inside if blocks causes issues
// Always call at the top level of your script
""",
        "alerts": """
## Alerts
// alert() — fires programmatically during bar
alert("Price crossed MA!", alert.freq_once_per_bar)
alert("Signal: " + str.tostring(close), alert.freq_all)

// Frequencies:
// alert.freq_all              — every tick
// alert.freq_once_per_bar     — once per bar (on first trigger)
// alert.freq_once_per_bar_close — only on bar close

// alertcondition() — creates an alert condition in the UI
alertcondition(ta.crossover(close, ta.sma(close, 20)),
               title="Price Cross MA",
               message="{{ticker}} crossed above 20 SMA at {{close}}")

// Template variables in message:
// {{ticker}}, {{exchange}}, {{close}}, {{open}}, {{high}}, {{low}}
// {{volume}}, {{time}}, {{timenow}}, {{interval}}
// {{plot_0}}, {{plot_1}}  — values of named plots

// Strategy alerts
// Use strategy.entry/exit with alert_message parameter
strategy.entry("Long", strategy.long, alert_message="ENTER LONG {{ticker}}")
""",
        "drawing": """
## Drawing Objects — line, box, label, polyline
// LINE
l = line.new(x1=bar_index[5], y1=low[5], x2=bar_index, y2=low,
             color=color.blue, width=2, style=line.style_solid)
line.set_y2(l, close)          // update
line.delete(l)                 // remove

// Styles: line.style_solid, line.style_dashed, line.style_dotted
// Extend: extend.none, extend.left, extend.right, extend.both

// BOX
b = box.new(left=bar_index[10], top=high, right=bar_index, bottom=low,
            border_color=color.red, bgcolor=color.new(color.red, 80))
box.set_right(b, bar_index)    // update right edge

// LABEL
lb = label.new(bar_index, high, "Signal",
               color=color.yellow, textcolor=color.black,
               style=label.style_label_down, size=size.normal)
label.set_text(lb, "Updated")

// Label styles: label.style_label_up/down/left/right/center
//               label.style_arrowup/down, label.style_circle, label.style_cross

// POLYLINE (v5.3+)
var pts = array.new<chart.point>()
array.push(pts, chart.point.from_index(bar_index, close))
// polyline.new(pts, curved=false, closed=false, line_color=color.blue)

// LIMITS (max objects on chart):
// lines: 500,  boxes: 500,  labels: 500  (use .delete() to manage)
""",
        "limitations": """
## Pine Script Limitations (Critical)
// Execution model
// - Script runs once per bar, left to right, history first then realtime
// - No true parallelism, no external calls during execution

// Object limits (per script)
max_bars_back default = 500 (set higher in indicator())
max labels    = 500
max lines     = 500
max boxes     = 500
max tables    = 1

// String limits
max string length ≈ 4096 chars

// Loop limits
max iterations per loop call = 500
// Recursive functions: NOT supported

// request.security limits
// Max 40 unique request.security() calls per script
// Cannot call inside loops or conditional blocks reliably

// Array limits
// Max array size: 100,000 elements
// matrix: no hard limit but performance degrades

// Plot limits
// Max 64 plot() calls per script
// Max 1 plotcandle() or plotbar() per script

// Indicator vs Strategy
// indicator() — display only, no order execution
// strategy() — can place orders, has commission/slippage settings

// Common errors
// "Pine cannot determine the referencing length" — use max_bars_back
// "Script has too many local variables" — refactor into functions
// "Operation not available for series" — use series-compatible operations
// "Cannot call alert() in a local scope" — move to global scope
""",
        "matrices": """
## Matrices (2D arrays)
m = matrix.new<float>(rows, cols, initial_value)

// Read/Write
matrix.get(m, row, col)
matrix.set(m, row, col, value)

// Dimensions
matrix.rows(m)
matrix.columns(m)

// Operations
matrix.copy(m)
matrix.transpose(m)       // returns new matrix
matrix.mult(m1, m2)       // matrix multiplication
matrix.sum(m1, m2)        // element-wise sum
matrix.diff(m1, m2)       // element-wise difference
matrix.pow(m, power)
matrix.pinv(m)            // pseudo-inverse
matrix.det(m)             // determinant
matrix.inv(m)             // inverse

// Add/remove rows/cols
matrix.add_row(m, row, array)
matrix.add_col(m, col, array)
matrix.remove_row(m, row)
matrix.remove_col(m, col)

// Convert
matrix.row(m, row)        // returns array
matrix.col(m, col)        // returns array
matrix.to_array(m)        // flatten to 1D array
""",
        "maps": """
## Maps (key-value store, v5.2+)
// Create
m = map.new<string, float>()

// Add / update
map.put(m, "key", 42.0)

// Read
val = map.get(m, "key")      // returns na if missing
contains = map.contains(m, "key")

// Remove
map.remove(m, "key")

// Iterate
for [k, v] in m
    // k = key (string), v = value (float)

// Size
map.size(m)

// Keys / Values
keys = map.keys(m)      // returns array of keys
vals = map.values(m)    // returns array of values

// Copy
m2 = map.copy(m)

// Clear
map.clear(m)

// Common key types: string, int
// Common value types: float, int, bool, string, line, label, box, array
""",
    }

    topic_lower = topic.lower().strip()

    # Exact match
    if topic_lower in quick_refs:
        return quick_refs[topic_lower]

    # Partial match
    for key, content in quick_refs.items():
        if topic_lower in key or key in topic_lower:
            return content

    # Search in docs as fallback
    return search_pine_docs(topic, max_results=5)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradingView Pine Script Docs MCP Server")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport instead of stdio")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP host")
    args = parser.parse_args()

    # Pre-load index at startup
    _ensure_loaded()
    if not _docs:
        print("WARNING: No docs indexed. Run: python indexer.py", file=sys.stderr)
    else:
        print(f"[tv-docs-mcp] Loaded {len(_docs)} documents", file=sys.stderr)

    if args.http:
        print(f"[tv-docs-mcp] Starting HTTP server on {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        print("[tv-docs-mcp] Starting stdio MCP server", file=sys.stderr)
        mcp.run()
