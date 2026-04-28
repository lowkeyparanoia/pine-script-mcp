# pine-script-mcp

An MCP (Model Context Protocol) server that gives any LLM full access to TradingView Pine Script v5/v6 documentation — searchable, structured, and fast.

Works with **Claude Code, Claude Desktop, Cursor, Windsurf, GPT-4, Gemini**, or any MCP-compatible client.

---

## What it does

Indexes the complete TradingView Pine Script documentation (User Manual + Reference) and exposes it through 5 MCP tools. Instead of an LLM hallucinating Pine Script syntax, it can query this server for accurate function signatures, examples, and conceptual docs.

## Tools

| Tool | Description |
|------|-------------|
| `search_pine_docs` | BM25 full-text search across all docs |
| `search_pine_examples` | Search and return only Pine Script **code examples** — use when writing code |
| `get_pine_function` | Look up any function/variable by name (e.g. `ta.sma`, `strategy.entry`) |
| `get_pine_page` | Retrieve a complete docs section (e.g. `concepts/strategies`) |
| `list_pine_sections` | Browse all indexed content |
| `pine_quick_ref` | Instant cheat-sheet cards — 23 topics: `indicators`, `strategy`, `plotting`, `arrays`, `inputs`, `colors`, `operators`, `types`, `time`, `loops`, `functions`, `request`, `alerts`, `drawing`, `limitations`, `matrices`, `maps`, `var_varip`, `na`, `type_qualifiers`, `libraries`, `objects`, `debugging` |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jrenothmisquith/pine-script-mcp
cd pine-script-mcp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Playwright for full reference scraping (816 function docs)
python -m playwright install chromium

# 4. Build the docs index (one-time, ~3 minutes including Playwright)
python indexer.py

# 5. Start the server
python server.py
```

## Integrate with Claude Code

**Option A — Project-level** (add to `.mcp.json` in your project root):
```json
{
  "mcpServers": {
    "pine-docs": {
      "command": "python",
      "args": ["/absolute/path/to/pine-script-mcp/server.py"]
    }
  }
}
```

**Option B — Global** (available in all Claude Code sessions):
```bash
claude mcp add pine-docs python /absolute/path/to/pine-script-mcp/server.py
```

## Integrate with Claude Desktop

Edit the config file for your OS:
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pine-docs": {
      "command": "python",
      "args": ["/absolute/path/to/pine-script-mcp/server.py"]
    }
  }
}
```

## Integrate with Cursor / Windsurf

Add to your IDE's MCP settings (usually `mcp.json`):
```json
{
  "servers": {
    "pine-docs": {
      "command": "python",
      "args": ["/absolute/path/to/pine-script-mcp/server.py"],
      "transport": "stdio"
    }
  }
}
```

## HTTP Mode (for remote LLMs or custom integrations)

```bash
python server.py --http --port 8080
# Server available at http://localhost:8080
```

Point any OpenAI-compatible MCP client at `http://localhost:8080`.

## What's indexed

| Source | Count | Content |
|--------|-------|---------|
| Pine Script User Manual | ~34 pages | Language, concepts, strategies, visuals, FAQ |
| Playwright SPA reference | 816 entries | Full function/variable/operator/type/constant docs scraped from the live reference SPA |
| Synthetic reference | ~8 entries | Gap-fill entries for functions not caught by the SPA scraper |
| Quick reference cards | 17 topics | Hardcoded cheat sheets: indicators, strategy, plotting, arrays, inputs, colors, operators, types, time, loops, functions, request, alerts, drawing, limitations, matrices, maps |

**Total: ~858 documents**

## Refresh the index

```bash
python indexer.py --force   # re-scrape everything fresh
```

The TradingView docs update occasionally. Re-run the indexer monthly or after Pine Script version updates.

## Requirements

- Python 3.10+
- Internet access for initial indexing (not required at runtime)
- ~5MB disk for the index cache

## Coverage gaps & known limitations

- The Playwright scraper captures what is rendered in the DOM on page load + scroll. A small number of entries may have partial descriptions if TradingView lazy-loads them beyond the scroll pass.
- Pine Script v6 features may not be fully indexed if TradingView hasn't updated the reference page.
- `request.security` nuances (lookahead, barmerge modes) are documented in both the scraped reference and the user manual pages.

## Optional: Playwright for full reference scraping

The indexer automatically runs the Playwright scraper if `playwright` is installed. To install:

```bash
pip install playwright
python -m playwright install chromium
python indexer.py --force
```

Without Playwright, the index still works but has ~42 docs (user manual only) instead of 858.

## Contributing

PRs welcome. The most impactful improvements:
1. Fix gap-filling in `scraper_playwright.py` for entries that miss descriptions
2. Add v6 migration guide content once TradingView publishes it
3. Add more `pine_quick_ref` topics: `pine_quick_ref("pine6_changes")`

## License

MIT
