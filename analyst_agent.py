"""
analyst_agent.py
----------------
Author : Learner transitioning into Agentic AI Engineering
Purpose: Autonomous corporate financial analyst agent powered by Groq.

Changes from previous version
------------------------------
RAG INTEGRATION
  Before calling the LLM, the agent now queries the RAG pipeline in
  financial_parser.py for relevant benchmarks and frameworks.
  This context is injected into the prompt alongside the financial data.
  Result: the agent can say "your gross margin of 40.7% is within the
  manufacturing benchmark of 25-45%" rather than relying on training
  data alone.

Everything else (agentic loop, tool execution, conversation history)
remains identical to the previous version.

How to obtain a Groq API key (free)
------------------------------------
1. Visit https://console.groq.com/keys
2. Create a free account
3. Generate an API key (starts with gsk_)
4. Set environment variable: export GROQ_API_KEY=gsk_...
"""

import json
import re
from typing import Optional
from groq import Groq
from financial_parser import parse_file, summarise_for_agent, get_rag


GROQ_MODEL     = "llama-3.3-70b-versatile"
MAX_TOKENS     = 4096
TEMPERATURE    = 0.15
MAX_TOOL_LOOPS = 5


SYSTEM_PROMPT = """You are a senior corporate financial analyst with 20 years of experience
in FP&A, investment banking, and corporate strategy.

You have been given two types of context:
  1. The company's actual financial data (P&L, Balance Sheet, Cash Flow, etc.)
  2. Retrieved benchmarks and frameworks from a financial knowledge database

Your job is to combine both sources to produce analysis that is:
  - Grounded in the company's actual numbers
  - Benchmarked against industry norms from the knowledge database
  - Specific, direct, and actionable

Analytical responsibilities
----------------------------
P&L: Revenue growth trend, margin trajectory at all three levels, cost structure,
     earnings quality, operating leverage signals

Balance Sheet: Liquidity ratios, capital structure, working capital metrics (DSO, DPO, DIO, CCC),
               asset quality warnings

Cash Flow: FCF generation, OCF vs net income divergence (accruals risk),
           capex intensity, cash conversion quality

Budget vs Actual: Variance decomposition by line item, EBITDA bridge construction,
                  management forecast reliability assessment

KPIs: Benchmark each metric, identify leading indicators, correlation to financial outcomes

Calculation tools
------------------
Emit a tool call block when you need a precise calculation.
Do not estimate or guess numeric results. Use the tools.

```tool
{"tool": "yoy_growth", "current": 45200000, "prior": 42400000, "label": "Revenue"}
```
```tool
{"tool": "ratio", "numerator": 19500000, "denominator": 14900000, "label": "Current Ratio"}
```
```tool
{"tool": "variance", "actual": 6800000, "budget": 9200000, "label": "EBITDA"}
```
```tool
{"tool": "margin", "numerator": 18400000, "denominator": 45200000, "label": "Gross Margin"}
```
```tool
{"tool": "cagr", "start": 38100000, "end": 45200000, "years": 2, "label": "Revenue CAGR"}
```
```tool
{"tool": "bridge", "title": "EBITDA Bridge: Budget vs Actual", "items": [["Revenue shortfall", -1800000], ["COGS inflation", -2400000]]}
```
```tool
{"tool": "benchmark", "metric": "ebitda_margin_pct", "value": 15.0, "industry": "manufacturing"}
```
```tool
{"tool": "working_capital", "receivables": 9400000, "inventory": 7200000, "payables": 7100000, "revenue": 45200000, "cogs": 26800000}
```

Response structure
------------------
EXECUTIVE SUMMARY
  3-4 sentences. The most important finding first.

FINANCIAL PERFORMANCE
  Work through every available statement with actual numbers.
  Reference retrieved benchmarks where relevant.

KEY RISKS
  Ranked by severity. Specific and quantified. No vague language.

RECOMMENDATIONS
  Numbered. Specific. Actionable. Not "consider reviewing."

WATCH LIST
  3-5 items to monitor next reporting period.

Rules
-----
- Use actual numbers from the data. No vague generalities.
- When a retrieved benchmark is relevant, cite it explicitly.
- No emoji or decorative symbols.
- Name problems directly without softening language."""


# ---------------------------------------------------------------------------
# Calculation tools (identical to previous version)
# ---------------------------------------------------------------------------

def run_tool(tc: dict) -> str:
    tool = tc.get("tool", "")
    try:
        if tool == "yoy_growth":
            cur, pri = float(tc["current"]), float(tc["prior"])
            label    = tc.get("label", "Metric")
            if pri == 0:
                return f"{label}: prior period is zero, growth rate undefined"
            g   = (cur - pri) / abs(pri) * 100
            dir = "up" if g > 0 else "down"
            return f"{label} year-on-year: {pri:,.0f} -> {cur:,.0f} ({dir} {abs(g):.1f}%)"

        elif tool == "ratio":
            n, d  = float(tc["numerator"]), float(tc["denominator"])
            label = tc.get("label", "Ratio")
            if d == 0:
                return f"{label}: denominator is zero"
            return f"{label}: {n / d:.2f}x"

        elif tool == "variance":
            act, bud = float(tc["actual"]), float(tc["budget"])
            label    = tc.get("label", "Item")
            var      = act - bud
            pct      = (var / abs(bud) * 100) if bud != 0 else 0
            status   = "FAVOURABLE" if var > 0 else "ADVERSE"
            return (
                f"{label} -- Actual: {act:,.0f} | Budget: {bud:,.0f} | "
                f"Variance: {var:+,.0f} ({pct:+.1f}%) -- {status}"
            )

        elif tool == "margin":
            n, d  = float(tc["numerator"]), float(tc["denominator"])
            label = tc.get("label", "Margin")
            if d == 0:
                return f"{label}: denominator is zero"
            return f"{label}: {n / d * 100:.1f}%"

        elif tool == "cagr":
            s, e, y = float(tc["start"]), float(tc["end"]), float(tc["years"])
            label   = tc.get("label", "CAGR")
            if s <= 0 or y <= 0:
                return f"{label}: non-positive start value or zero years"
            cagr = ((e / s) ** (1 / y) - 1) * 100
            return f"{label}: {cagr:.1f}% per year over {y:.0f} years ({s:,.0f} -> {e:,.0f})"

        elif tool == "bridge":
            title = tc.get("title", "Bridge")
            items = tc.get("items", [])
            total = sum(v for _, v in items)
            rows  = "\n".join(f"  {lbl:<44} {val:>+14,.0f}" for lbl, val in items)
            sep   = "  " + "-" * 60
            return f"{title}\n{rows}\n{sep}\n  {'Net impact':<44} {total:>+14,.0f}"

        elif tool == "benchmark":
            metric   = tc.get("metric", "")
            value    = float(tc["value"])
            industry = tc.get("industry", "general").lower()
            table = {
                "gross_margin_pct":     {"saas":(60,80),"manufacturing":(25,45),"retail":(20,40),"services":(40,70),"general":(30,60)},
                "ebitda_margin_pct":    {"saas":(15,35),"manufacturing":(8,18),"retail":(4,12),"services":(12,25),"general":(10,25)},
                "operating_margin_pct": {"saas":(10,30),"manufacturing":(5,15),"retail":(2,8),"services":(10,20),"general":(8,20)},
                "net_margin_pct":       {"saas":(8,25),"manufacturing":(3,12),"retail":(1,5),"services":(8,18),"general":(5,15)},
                "current_ratio":        {"general":(1.5,2.5)},
                "debt_to_equity":       {"general":(0.3,1.5)},
            }
            ranges = table.get(metric, {})
            norm   = ranges.get(industry) or ranges.get("general")
            if norm:
                lo, hi = norm
                if value < lo:   assessment = "BELOW normal range"
                elif value > hi: assessment = "ABOVE normal range"
                else:            assessment = "within normal range"
                return (
                    f"Benchmark | {metric}: {value} | "
                    f"{industry.title()} norm: {lo} - {hi} | {assessment}"
                )
            return f"Benchmark | {metric}: {value} | No benchmark data for '{industry}'"

        elif tool == "working_capital":
            rec  = float(tc["receivables"])
            inv  = float(tc.get("inventory", 0))
            pay  = float(tc["payables"])
            rev  = float(tc["revenue"])
            cogs = float(tc.get("cogs", rev * 0.6))
            dso  = rec / (rev  / 365)
            dpo  = pay / (cogs / 365)
            dio  = inv / (cogs / 365) if inv else 0
            ccc  = dso + dio - dpo
            return (
                f"Working Capital Analysis\n"
                f"  Days Sales Outstanding (DSO):     {dso:.0f} days\n"
                f"  Days Inventory Outstanding (DIO): {dio:.0f} days\n"
                f"  Days Payable Outstanding (DPO):   {dpo:.0f} days\n"
                f"  Cash Conversion Cycle (CCC):      {ccc:.0f} days"
            )

        else:
            return f"Unknown tool: '{tool}'"

    except KeyError as e:
        return f"Tool '{tool}' missing parameter: {e}"
    except Exception as e:
        return f"Tool '{tool}' error: {e}"


def _extract_tool_calls(text: str) -> list:
    pattern = r"```tool\s*\n(.*?)\n```"
    calls   = []
    for m in re.findall(pattern, text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CorporateAnalystAgent:
    """
    Autonomous corporate financial analyst.
    Now augmented with RAG: every question retrieves relevant benchmarks
    from the vector store before calling the LLM.

    Architecture summary
    --------------------
    load_file() -> parse financial file, store context
    ask()       -> retrieve RAG context -> build prompt -> agentic loop -> return answer
    full_analysis() -> autonomous complete analysis, no question required

    The agentic loop (ReAct pattern):
      1. LLM reasons over financial data + RAG context
      2. LLM emits tool calls for calculations
      3. Tools run in Python, results returned
      4. LLM continues reasoning with results
      5. Repeat until LLM produces a final answer
    """

    def __init__(self, api_key: str):
        self.client       = Groq(api_key=api_key)
        self.rag          = get_rag()
        self.conversation: list = []
        self.file_context: Optional[str] = None
        self.filename:     Optional[str] = None
        self.parsed_data:  Optional[dict] = None

    def load_file(self, filepath: str) -> dict:
        self.parsed_data  = parse_file(filepath)
        self.file_context = summarise_for_agent(self.parsed_data)
        self.filename     = self.parsed_data["filename"]
        self.conversation = []

        detected = []
        if self.parsed_data.get("pnl"):             detected.append("P&L")
        if self.parsed_data.get("balance_sheet"):   detected.append("Balance Sheet")
        if self.parsed_data.get("cash_flow"):       detected.append("Cash Flow")
        if self.parsed_data.get("budget_variance"): detected.append("Budget vs Actual")
        if self.parsed_data.get("kpis"):            detected.append("KPIs")
        for t in self.parsed_data.get("other_tables", []):
            detected.append(f"Other: {t['sheet']}")

        return {
            "filename": self.filename,
            "sheets":   self.parsed_data.get("sheets_found", []),
            "detected": detected,
            "errors":   self.parsed_data.get("parse_errors", []),
            "ready":    True,
        }

    def _call_groq(self, messages: list) -> str:
        resp = self.client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
        )
        return resp.choices[0].message.content

    def ask(self, question: str) -> dict:
        """
        Ask the agent a question about the loaded financial data.

        RAG step: before calling the LLM, retrieve the most relevant
        benchmarks and frameworks for this specific question.
        These are injected into the prompt as additional context.
        """
        if not self.file_context:
            return {
                "answer":     "No financial file loaded. Please upload an Excel or CSV file first.",
                "tools_used": [],
                "rag_topics": [],
            }

        # Retrieve relevant benchmarks for this specific question
        rag_context  = self.rag.context_block(question, n_results=5)
        rag_chunks   = self.rag.retrieve(question, n_results=5)
        rag_topics   = [c["topic"] for c in rag_chunks]

        if not self.conversation:
            # First turn: inject full file data + RAG context
            opening = (
                f"You have been provided the following corporate financial data.\n\n"
                f"{self.file_context}\n\n"
                f"{rag_context}\n\n"
                f"QUESTION: {question}"
            )
            self.conversation.append({"role": "user", "content": opening})
        else:
            # Subsequent turns: inject RAG context with the new question
            self.conversation.append({
                "role":    "user",
                "content": f"{rag_context}\n\nQUESTION: {question}" if rag_context else question,
            })

        messages   = [{"role": "system", "content": SYSTEM_PROMPT}] + list(self.conversation)
        tools_used = []

        for _ in range(MAX_TOOL_LOOPS):
            response_text = self._call_groq(messages)
            tool_calls    = _extract_tool_calls(response_text)

            if not tool_calls:
                self.conversation.append({"role": "assistant", "content": response_text})
                return {
                    "answer":     response_text,
                    "tools_used": tools_used,
                    "rag_topics": rag_topics,
                }

            results = []
            for call in tool_calls:
                result = run_tool(call)
                results.append(result)
                tools_used.append(call.get("tool", "unknown"))

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role":    "user",
                "content": "CALCULATION RESULTS:\n" + "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(results)),
            })

        # Force final answer if max loops reached
        messages.append({"role": "user", "content": "Provide your final analysis now."})
        final = self._call_groq(messages)
        self.conversation.append({"role": "assistant", "content": final})
        return {"answer": final, "tools_used": tools_used, "rag_topics": rag_topics}

    def full_analysis(self) -> dict:
        return self.ask(
            "Perform a complete corporate financial analysis of all available statements. "
            "Benchmark every metric against industry norms using the retrieved knowledge. "
            "Identify the top findings, risks, and actionable recommendations. "
            "Structure this as a CFO briefing to the board of directors."
        )

    def reset_conversation(self):
        self.conversation = []

    def file_info(self) -> Optional[dict]:
        if not self.parsed_data:
            return None
        return {
            "filename":   self.filename,
            "sheets":     self.parsed_data.get("sheets_found", []),
            "has_pnl":    bool(self.parsed_data.get("pnl")),
            "has_bs":     bool(self.parsed_data.get("balance_sheet")),
            "has_cf":     bool(self.parsed_data.get("cash_flow")),
            "has_budget": bool(self.parsed_data.get("budget_variance")),
            "has_kpis":   bool(self.parsed_data.get("kpis")),
        }

    def rag_stats(self) -> dict:
        return self.rag.stats()