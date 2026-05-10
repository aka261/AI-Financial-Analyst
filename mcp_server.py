"""
mcp_server.py
-------------
Author : Learner transitioning into Agentic AI Engineering
Purpose: MCP (Model Context Protocol) server that exposes the corporate
         financial analyst tools as standard MCP tools.

What is MCP?
------------
MCP (Model Context Protocol) is an open standard created by Anthropic
that defines how AI agents communicate with external tools and data sources.
It separates the AI model from the tools it uses, so the same tools can
be used by any MCP-compatible agent or client.

Think of MCP as a USB standard for AI tools:
  - The MCP server exposes tools (like a USB device)
  - The agent connects and uses them (like a USB host)
  - Any agent that speaks MCP can use any MCP server

Why add MCP to this project?
------------------------------
Without MCP: the agent calls Python functions directly (tightly coupled)
With MCP:    the agent calls tools via a standard protocol (loosely coupled)

This means:
  - The tools can run in a separate process or even a separate machine
  - Any other MCP-compatible client (Claude Desktop, etc.) can use these tools
  - The architecture is production-ready and extensible

How this file works
--------------------
1. Creates an MCP Server instance
2. Registers financial analysis tools using the @server.list_tools decorator
3. Handles tool calls using the @server.call_tool decorator
4. Runs over stdio (standard input/output) -- the MCP standard transport

Tools exposed
-------------
  parse_financial_file    - Parse an Excel/CSV file, return detected statements
  calculate_yoy_growth    - Year-on-year growth calculation
  calculate_ratio         - Ratio calculation (e.g. current ratio)
  calculate_variance      - Budget vs actual variance
  calculate_margin        - Margin percentage
  calculate_cagr          - Compound annual growth rate
  build_bridge            - EBITDA or variance bridge
  benchmark_metric        - Compare metric against industry norm
  analyse_working_capital - DSO, DPO, DIO, cash conversion cycle
  retrieve_rag_context    - Query the RAG knowledge base

How to run
----------
Standalone test:
    python mcp_server.py

Used by server.py automatically -- no need to run manually.

How to connect Claude Desktop to this MCP server
-------------------------------------------------
Add to your Claude Desktop config (claude_desktop_config.json):

{
  "mcpServers": {
    "financial-analyst": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/your/project"
    }
  }
}
"""

import asyncio
import json
import math
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


# ---------------------------------------------------------------------------
# MCP Server instance
# ---------------------------------------------------------------------------

server = Server("corporate-financial-analyst")


# ---------------------------------------------------------------------------
# Tool definitions
# Each tool has a name, description, and JSON Schema for its inputs.
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Register all financial analyst tools with the MCP server."""
    return [

        types.Tool(
            name="parse_financial_file",
            description=(
                "Parse an Excel (.xlsx, .xls) or CSV financial file. "
                "Automatically detects P&L, Balance Sheet, Cash Flow, "
                "Budget vs Actual, and KPI sheets. Derives key margins "
                "and growth rates. Returns a structured summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Absolute or relative path to the financial file"
                    }
                },
                "required": ["filepath"]
            }
        ),

        types.Tool(
            name="calculate_yoy_growth",
            description=(
                "Calculate year-on-year (or period-on-period) growth rate. "
                "Returns direction and percentage change."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "current": {
                        "type": "number",
                        "description": "Current period value"
                    },
                    "prior": {
                        "type": "number",
                        "description": "Prior period value"
                    },
                    "label": {
                        "type": "string",
                        "description": "Name of the metric (e.g. Revenue)"
                    }
                },
                "required": ["current", "prior"]
            }
        ),

        types.Tool(
            name="calculate_ratio",
            description=(
                "Calculate a financial ratio (numerator divided by denominator). "
                "Use for current ratio, quick ratio, debt-to-equity, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "numerator":   {"type": "number"},
                    "denominator": {"type": "number"},
                    "label":       {"type": "string", "description": "Name of the ratio"}
                },
                "required": ["numerator", "denominator"]
            }
        ),

        types.Tool(
            name="calculate_variance",
            description=(
                "Calculate budget vs actual variance. "
                "Returns absolute variance, percentage variance, "
                "and whether it is FAVOURABLE or ADVERSE."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "actual": {"type": "number", "description": "Actual result"},
                    "budget": {"type": "number", "description": "Budget or target"},
                    "label":  {"type": "string", "description": "Name of the line item"}
                },
                "required": ["actual", "budget"]
            }
        ),

        types.Tool(
            name="calculate_margin",
            description=(
                "Calculate a margin percentage (numerator / denominator * 100). "
                "Use for gross margin, EBITDA margin, net margin, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "numerator":   {"type": "number"},
                    "denominator": {"type": "number"},
                    "label":       {"type": "string"}
                },
                "required": ["numerator", "denominator"]
            }
        ),

        types.Tool(
            name="calculate_cagr",
            description=(
                "Calculate Compound Annual Growth Rate (CAGR) "
                "between a start value and end value over a number of years."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Starting value"},
                    "end":   {"type": "number", "description": "Ending value"},
                    "years": {"type": "number", "description": "Number of years"},
                    "label": {"type": "string"}
                },
                "required": ["start", "end", "years"]
            }
        ),

        types.Tool(
            name="build_bridge",
            description=(
                "Build a financial bridge (e.g. EBITDA bridge from budget to actual). "
                "Takes a list of items with labels and values, returns a formatted bridge "
                "showing each component and the net total."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the bridge (e.g. EBITDA Bridge: Budget vs Actual)"
                    },
                    "items": {
                        "type": "array",
                        "description": "List of [label, value] pairs. Negative = adverse.",
                        "items": {
                            "type": "array",
                            "items": [
                                {"type": "string"},
                                {"type": "number"}
                            ]
                        }
                    }
                },
                "required": ["title", "items"]
            }
        ),

        types.Tool(
            name="benchmark_metric",
            description=(
                "Compare a financial metric against industry benchmarks. "
                "Supported metrics: gross_margin_pct, ebitda_margin_pct, "
                "operating_margin_pct, net_margin_pct, current_ratio, debt_to_equity. "
                "Supported industries: manufacturing, saas, retail, services, general."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric name (e.g. ebitda_margin_pct)"
                    },
                    "value": {
                        "type": "number",
                        "description": "Actual value of the metric"
                    },
                    "industry": {
                        "type": "string",
                        "description": "Industry for benchmarking (default: general)"
                    }
                },
                "required": ["metric", "value"]
            }
        ),

        types.Tool(
            name="analyse_working_capital",
            description=(
                "Calculate working capital metrics: DSO, DIO, DPO, "
                "and Cash Conversion Cycle (CCC)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "receivables": {"type": "number", "description": "Accounts receivable balance"},
                    "payables":    {"type": "number", "description": "Accounts payable balance"},
                    "revenue":     {"type": "number", "description": "Annual revenue"},
                    "inventory":   {"type": "number", "description": "Inventory balance (optional)"},
                    "cogs":        {"type": "number", "description": "Cost of goods sold (optional)"}
                },
                "required": ["receivables", "payables", "revenue"]
            }
        ),

        types.Tool(
            name="retrieve_rag_context",
            description=(
                "Query the RAG (Retrieval-Augmented Generation) financial knowledge base. "
                "Returns the most relevant benchmarks, frameworks, and financial norms "
                "for the given query. Use this before analysing any financial metric."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The financial topic or question to search for"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
    ]


# ---------------------------------------------------------------------------
# Tool execution
# Each tool call is routed here and dispatched to the appropriate function.
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """
    Execute a tool call and return the result as MCP TextContent.

    Learning note: MCP tool results must be returned as a list of
    content objects. TextContent is the simplest -- plain text.
    In production you could also return ImageContent or EmbeddedResource.
    """
    try:
        result = _dispatch(name, arguments)
        return [types.TextContent(type="text", text=result)]
    except Exception as e:
        error_msg = f"Tool '{name}' failed: {e}"
        return [types.TextContent(type="text", text=error_msg)]


def _dispatch(name: str, args: dict) -> str:
    """Route tool name to the correct handler function."""
    handlers = {
        "parse_financial_file":   _tool_parse_file,
        "calculate_yoy_growth":   _tool_yoy_growth,
        "calculate_ratio":        _tool_ratio,
        "calculate_variance":     _tool_variance,
        "calculate_margin":       _tool_margin,
        "calculate_cagr":         _tool_cagr,
        "build_bridge":           _tool_bridge,
        "benchmark_metric":       _tool_benchmark,
        "analyse_working_capital":_tool_working_capital,
        "retrieve_rag_context":   _tool_rag,
    }
    if name not in handlers:
        return f"Unknown tool: '{name}'. Available: {list(handlers.keys())}"
    return handlers[name](args)


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def _tool_parse_file(args: dict) -> str:
    """Parse a financial Excel or CSV file."""
    filepath = args.get("filepath", "")
    if not filepath:
        return "Error: filepath is required"
    if not Path(filepath).exists():
        return f"Error: file not found at '{filepath}'"

    # Import here to avoid circular dependency issues at module load
    try:
        from financial_parser import parse_file, summarise_for_agent
        parsed  = parse_file(filepath)
        summary = summarise_for_agent(parsed)

        detected = []
        if parsed.get("pnl"):             detected.append("P&L")
        if parsed.get("balance_sheet"):   detected.append("Balance Sheet")
        if parsed.get("cash_flow"):       detected.append("Cash Flow")
        if parsed.get("budget_variance"): detected.append("Budget vs Actual")
        if parsed.get("kpis"):            detected.append("KPIs")

        header = (
            f"FILE PARSED: {parsed['filename']}\n"
            f"Sheets found: {', '.join(parsed.get('sheets_found', []))}\n"
            f"Detected: {', '.join(detected) or 'none'}\n"
            f"Errors: {', '.join(parsed.get('parse_errors', [])) or 'none'}\n\n"
        )
        return header + summary

    except ImportError:
        return "Error: financial_parser.py not found. Ensure it is in the same directory."
    except Exception as e:
        return f"Error parsing file: {e}"


def _tool_yoy_growth(args: dict) -> str:
    cur   = float(args["current"])
    pri   = float(args["prior"])
    label = args.get("label", "Metric")
    if pri == 0:
        return f"{label}: prior period is zero, growth rate undefined"
    g   = (cur - pri) / abs(pri) * 100
    dir = "up" if g > 0 else "down"
    return f"{label} year-on-year: {pri:,.0f} -> {cur:,.0f} ({dir} {abs(g):.1f}%)"


def _tool_ratio(args: dict) -> str:
    n     = float(args["numerator"])
    d     = float(args["denominator"])
    label = args.get("label", "Ratio")
    if d == 0:
        return f"{label}: denominator is zero, ratio undefined"
    return f"{label}: {n / d:.2f}x"


def _tool_variance(args: dict) -> str:
    act   = float(args["actual"])
    bud   = float(args["budget"])
    label = args.get("label", "Item")
    var   = act - bud
    pct   = (var / abs(bud) * 100) if bud != 0 else 0
    status = "FAVOURABLE" if var > 0 else "ADVERSE"
    return (
        f"{label}\n"
        f"  Actual:   {act:>16,.0f}\n"
        f"  Budget:   {bud:>16,.0f}\n"
        f"  Variance: {var:>+16,.0f}  ({pct:+.1f}%)  {status}"
    )


def _tool_margin(args: dict) -> str:
    n     = float(args["numerator"])
    d     = float(args["denominator"])
    label = args.get("label", "Margin")
    if d == 0:
        return f"{label}: denominator is zero"
    return f"{label}: {n / d * 100:.1f}%  ({n:,.0f} / {d:,.0f})"


def _tool_cagr(args: dict) -> str:
    s     = float(args["start"])
    e     = float(args["end"])
    y     = float(args["years"])
    label = args.get("label", "CAGR")
    if s <= 0 or y <= 0:
        return f"{label}: start value must be positive and years must be > 0"
    cagr  = ((e / s) ** (1 / y) - 1) * 100
    return f"{label}: {cagr:.1f}% per year over {y:.0f} years  ({s:,.0f} -> {e:,.0f})"


def _tool_bridge(args: dict) -> str:
    title = args.get("title", "Bridge")
    items = args.get("items", [])
    if not items:
        return f"{title}: no items provided"
    total = sum(float(v) for _, v in items)
    rows  = "\n".join(
        f"  {lbl:<44} {float(val):>+14,.0f}"
        for lbl, val in items
    )
    sep = "  " + "-" * 60
    return f"{title}\n{rows}\n{sep}\n  {'Net impact':<44} {total:>+14,.0f}"


def _tool_benchmark(args: dict) -> str:
    metric   = args.get("metric", "")
    value    = float(args["value"])
    industry = args.get("industry", "general").lower()

    benchmarks = {
        "gross_margin_pct": {
            "saas": (60, 80), "manufacturing": (25, 45),
            "retail": (20, 40), "services": (40, 70), "general": (30, 60),
        },
        "ebitda_margin_pct": {
            "saas": (15, 35), "manufacturing": (8, 18),
            "retail": (4, 12), "services": (12, 25), "general": (10, 25),
        },
        "operating_margin_pct": {
            "saas": (10, 30), "manufacturing": (5, 15),
            "retail": (2, 8), "services": (10, 20), "general": (8, 20),
        },
        "net_margin_pct": {
            "saas": (8, 25), "manufacturing": (3, 12),
            "retail": (1, 5), "services": (8, 18), "general": (5, 15),
        },
        "current_ratio":  {"general": (1.5, 2.5)},
        "debt_to_equity": {"general": (0.3, 1.5)},
    }

    ranges = benchmarks.get(metric, {})
    norm   = ranges.get(industry) or ranges.get("general")

    if not norm:
        return f"No benchmark data for metric '{metric}' and industry '{industry}'"

    lo, hi = norm
    if value < lo:   assessment = f"BELOW normal range -- potential concern"
    elif value > hi: assessment = f"ABOVE normal range -- investigate driver"
    else:            assessment = f"within normal range"

    return (
        f"Benchmark result\n"
        f"  Metric:     {metric}\n"
        f"  Value:      {value}\n"
        f"  Industry:   {industry.title()}\n"
        f"  Norm range: {lo} - {hi}\n"
        f"  Assessment: {assessment}"
    )


def _tool_working_capital(args: dict) -> str:
    rec  = float(args["receivables"])
    pay  = float(args["payables"])
    rev  = float(args["revenue"])
    inv  = float(args.get("inventory", 0))
    cogs = float(args.get("cogs", rev * 0.6))

    dso = rec / (rev  / 365)
    dpo = pay / (cogs / 365)
    dio = inv / (cogs / 365) if inv else 0
    ccc = dso + dio - dpo

    return (
        f"Working Capital Analysis\n"
        f"  Days Sales Outstanding (DSO):     {dso:>6.0f} days\n"
        f"  Days Inventory Outstanding (DIO): {dio:>6.0f} days\n"
        f"  Days Payable Outstanding (DPO):   {dpo:>6.0f} days\n"
        f"  Cash Conversion Cycle (CCC):      {ccc:>6.0f} days\n"
        f"\n"
        f"  Interpretation:\n"
        f"  DSO benchmark: 30-45 days (B2B). {'Above benchmark -- collections risk' if dso > 45 else 'Within benchmark'}\n"
        f"  DPO benchmark: 30-60 days.        {'Within benchmark' if 30 <= dpo <= 60 else 'Outside benchmark'}\n"
        f"  CCC:           {'Negative -- collecting before paying (excellent)' if ccc < 0 else 'Positive -- cash tied up in working capital'}"
    )


def _tool_rag(args: dict) -> str:
    """Query the RAG knowledge base for relevant financial benchmarks."""
    query     = args.get("query", "")
    n_results = int(args.get("n_results", 5))

    if not query:
        return "Error: query is required"

    try:
        from financial_parser import get_rag
        rag     = get_rag()
        context = rag.context_block(query, n_results=n_results)
        if not context:
            return f"No relevant knowledge found for query: '{query}'"
        return context
    except ImportError:
        return "Error: financial_parser.py not found. Ensure it is in the same directory."
    except Exception as e:
        return f"RAG query failed: {e}"


# ---------------------------------------------------------------------------
# Run the MCP server
# ---------------------------------------------------------------------------

async def main():
    """Start the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    print("Corporate Financial Analyst MCP Server starting...", file=sys.stderr)
    print("Waiting for MCP client connection via stdio...", file=sys.stderr)
    print("Tools available: 10 financial analysis tools", file=sys.stderr)
    asyncio.run(main())