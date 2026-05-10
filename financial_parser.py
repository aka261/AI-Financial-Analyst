"""
financial_parser.py
-------------------
Author : Learner transitioning into Agentic AI Engineering
Purpose: Two responsibilities in one file.

  1. FILE PARSING
     Reads .xlsx, .xls, .csv financial files.
     Detects which sheet is which statement.
     Derives key margins and growth rates automatically.

  2. RAG PIPELINE (Retrieval-Augmented Generation)
     Maintains a ChromaDB vector store of financial benchmarks,
     industry norms, and corporate finance knowledge.
     Before every agent call, the most relevant knowledge chunks
     are retrieved and injected into the prompt as context.

Why RAG matters for financial analysis
---------------------------------------
Without RAG the agent can only use what it knows from training.
With RAG it can compare the company's numbers against a curated,
updatable knowledge base of industry benchmarks and financial norms.
This significantly improves the precision and credibility of analysis.

How the RAG pipeline works
---------------------------
1. On first run, 40+ financial knowledge documents are embedded
   and stored in ChromaDB (a local vector database).
2. When a question arrives, the query is embedded and the most
   semantically similar knowledge chunks are retrieved.
3. Retrieved chunks are prepended to the LLM prompt as context.
4. The agent reasons over both the company data AND the benchmarks.

Learning note
-------------
This is the "Retrieval" step in RAG. The "Augmented Generation"
happens in analyst_agent.py when the retrieved text is injected
into the prompt. Understanding these two steps separately is
essential for any agentic AI engineer.

Dependencies: pandas, openpyxl, chromadb
"""

import re
import json
import math
import hashlib
from pathlib import Path
from typing import Optional
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


# ---------------------------------------------------------------------------
# Lightweight embedding function
# No external model download required.
# Uses TF-IDF style hashing to produce 256-dimensional vectors.
# Sufficient for keyword-rich financial domain matching.
# ---------------------------------------------------------------------------

class FinancialEmbedder(EmbeddingFunction):
    """
    Custom embedding function for the RAG vector store.

    Learning note: In production you would replace this with a proper
    sentence-transformer model (e.g. all-MiniLM-L6-v2) or an API-based
    embedder. This version works offline and requires no downloads,
    making it suitable for a learning environment.
    """

    DIM = 256
    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "of", "to", "and", "or",
        "for", "on", "at", "by", "with", "as", "are", "was", "were",
        "be", "been", "has", "have", "had", "this", "that", "its",
    }

    def _tokenise(self, text: str) -> list:
        tokens = re.findall(r'\b[a-z][a-z0-9]{1,20}\b', text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def _embed(self, text: str) -> list:
        tokens = self._tokenise(text)
        if not tokens:
            return [0.0] * self.DIM
        vec = [0.0] * self.DIM
        freq: dict = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        for token, count in freq.items():
            tf  = count / len(tokens)
            idf = 1.0 + math.log(1 + 1 / (1 + len(token)))
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[slot] += tf * idf
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(doc) for doc in input]


# ---------------------------------------------------------------------------
# Financial knowledge base
# 40+ documents covering benchmarks, ratios, warning signs, frameworks.
# These are the documents the agent retrieves from when answering questions.
# ---------------------------------------------------------------------------

FINANCIAL_KNOWLEDGE = [

    # Profitability benchmarks by industry
    {
        "id": "bm_gm_mfg",
        "text": "Manufacturing gross margin benchmark: healthy range is 25 to 45 percent. Below 25 percent indicates raw material cost pressure or pricing weakness. Above 45 percent suggests premium positioning or significant automation advantage.",
        "category": "benchmark", "topic": "gross margin manufacturing"
    },
    {
        "id": "bm_gm_saas",
        "text": "SaaS and software gross margin benchmark: healthy range is 60 to 80 percent. Below 60 percent may indicate excessive hosting costs or professional services drag. Best-in-class SaaS companies exceed 75 percent gross margin.",
        "category": "benchmark", "topic": "gross margin saas software"
    },
    {
        "id": "bm_gm_retail",
        "text": "Retail gross margin benchmark: healthy range is 20 to 40 percent. Grocery retail typically runs 20 to 25 percent. Specialty retail can reach 40 to 50 percent. Margin below 20 percent in retail indicates intense price competition.",
        "category": "benchmark", "topic": "gross margin retail"
    },
    {
        "id": "bm_gm_services",
        "text": "Professional services gross margin benchmark: healthy range is 40 to 70 percent. Consulting firms typically achieve 45 to 60 percent. IT services run 35 to 55 percent. Margin below 35 percent in services suggests poor utilisation or underpricing.",
        "category": "benchmark", "topic": "gross margin services consulting"
    },
    {
        "id": "bm_ebitda_mfg",
        "text": "Manufacturing EBITDA margin benchmark: healthy range is 8 to 18 percent. Capital-intensive heavy manufacturing typically runs 10 to 15 percent. Light manufacturing or assembly can reach 15 to 20 percent. Below 8 percent is a warning sign.",
        "category": "benchmark", "topic": "ebitda margin manufacturing"
    },
    {
        "id": "bm_ebitda_saas",
        "text": "SaaS EBITDA margin benchmark: early-stage SaaS may run negative EBITDA while investing in growth. Mature SaaS companies target 15 to 35 percent EBITDA margin. The Rule of 40 states that revenue growth rate plus EBITDA margin should exceed 40.",
        "category": "benchmark", "topic": "ebitda margin saas rule of 40"
    },
    {
        "id": "bm_ebitda_retail",
        "text": "Retail EBITDA margin benchmark: 4 to 12 percent is the normal range. Grocery and discount retail typically 4 to 7 percent. Specialty and luxury retail 8 to 15 percent. Below 4 percent leaves insufficient buffer for capex and debt service.",
        "category": "benchmark", "topic": "ebitda margin retail"
    },
    {
        "id": "bm_net_margin",
        "text": "Net profit margin benchmarks by sector: manufacturing 3 to 12 percent, retail 1 to 5 percent, services 8 to 18 percent, SaaS 8 to 25 percent, financial services 15 to 30 percent. Net margin below 3 percent in most industries is considered thin.",
        "category": "benchmark", "topic": "net profit margin benchmark sector"
    },

    # Liquidity and balance sheet ratios
    {
        "id": "bm_current_ratio",
        "text": "Current ratio benchmark: 1.5 to 2.5 is the healthy range for most industries. Below 1.0 means current liabilities exceed current assets, indicating potential short-term solvency risk. Above 3.0 may signal underdeployed cash or poor working capital management.",
        "category": "benchmark", "topic": "current ratio liquidity"
    },
    {
        "id": "bm_quick_ratio",
        "text": "Quick ratio (acid test) benchmark: excludes inventory from current assets. Healthy range is 1.0 to 2.0. Below 0.8 is a warning for businesses with slow-moving inventory. Best measure of immediate liquidity.",
        "category": "benchmark", "topic": "quick ratio acid test liquidity"
    },
    {
        "id": "bm_debt_equity",
        "text": "Debt-to-equity ratio benchmark: 0.3 to 1.5 is normal for most industrial companies. Technology and asset-light businesses often run below 0.5. Capital-intensive industries like utilities and real estate may run 2.0 or higher. Above 2.0 in a non-capital-intensive business is a concern.",
        "category": "benchmark", "topic": "debt equity leverage capital structure"
    },
    {
        "id": "bm_net_debt_ebitda",
        "text": "Net debt to EBITDA benchmark: below 2.0x is considered conservative. 2.0 to 3.5x is moderate leverage. Above 4.0x is aggressive and increases refinancing risk. Banks typically covenant at 3.0 to 4.0x net debt to EBITDA.",
        "category": "benchmark", "topic": "net debt ebitda leverage covenant"
    },
    {
        "id": "bm_interest_cover",
        "text": "Interest coverage ratio benchmark (EBIT divided by interest expense): above 3.0x is comfortable. 1.5 to 3.0x is marginal. Below 1.5x means EBIT does not comfortably cover interest, raising default risk. Lenders typically require minimum 2.0x coverage.",
        "category": "benchmark", "topic": "interest coverage ratio debt service"
    },

    # Working capital metrics
    {
        "id": "wc_dso",
        "text": "Days Sales Outstanding (DSO) benchmark: 30 to 45 days is normal for B2B businesses. Below 30 days indicates strong collections. Above 60 days suggests collections weakness or customer credit risk. Rapidly rising DSO while revenue grows is a red flag for earnings quality.",
        "category": "working_capital", "topic": "DSO days sales outstanding receivables"
    },
    {
        "id": "wc_dpo",
        "text": "Days Payable Outstanding (DPO) benchmark: 30 to 60 days is typical. Higher DPO means the business is using supplier credit effectively. DPO above 90 days may strain supplier relationships. Retailers and large corporates often negotiate longer payment terms.",
        "category": "working_capital", "topic": "DPO days payable outstanding creditors"
    },
    {
        "id": "wc_dio",
        "text": "Days Inventory Outstanding (DIO) benchmark: manufacturing typically 45 to 90 days. Retail 30 to 60 days. Technology hardware 20 to 45 days. Rising DIO without revenue growth indicates inventory build-up and potential obsolescence risk.",
        "category": "working_capital", "topic": "DIO days inventory outstanding stock"
    },
    {
        "id": "wc_ccc",
        "text": "Cash Conversion Cycle (CCC = DSO + DIO - DPO) benchmark: negative CCC means the business collects cash before paying suppliers, which is ideal (Amazon, Walmart achieve this). 0 to 30 days is excellent. 30 to 60 days is acceptable. Above 90 days strains cash flow significantly.",
        "category": "working_capital", "topic": "cash conversion cycle working capital efficiency"
    },

    # Earnings quality and cash flow
    {
        "id": "eq_fcf_conversion",
        "text": "Free cash flow conversion (FCF divided by net profit) benchmark: above 80 percent is strong and indicates profit is backed by real cash. 50 to 80 percent is acceptable. Below 50 percent suggests aggressive accruals accounting or high maintenance capex. A company with consistently strong net income but low FCF conversion warrants scrutiny.",
        "category": "earnings_quality", "topic": "free cash flow conversion earnings quality"
    },
    {
        "id": "eq_ocf_ni",
        "text": "Operating cash flow versus net income: OCF should generally track net income over time. If net income is consistently positive but OCF is negative or significantly lower, this is a red flag for accruals manipulation. Common causes include aggressive revenue recognition, inadequate bad debt provisioning, or capitalising costs that should be expensed.",
        "category": "earnings_quality", "topic": "operating cash flow net income divergence accruals"
    },
    {
        "id": "eq_receivables",
        "text": "Receivables growth versus revenue growth: if accounts receivable grows significantly faster than revenue, it may indicate channel stuffing, relaxed credit terms to boost reported sales, or collection problems. DSO expansion alongside revenue growth is a key earnings quality warning sign.",
        "category": "earnings_quality", "topic": "receivables revenue growth earnings quality warning"
    },

    # Margin analysis frameworks
    {
        "id": "ma_margin_bridge",
        "text": "Margin compression analysis framework: when EBITDA margin declines, decompose into revenue mix effect, price effect, volume effect, and cost effect. Distinguish between structural margin compression (permanent) and cyclical compression (temporary). Structural drivers include input cost inflation, competitive pricing pressure, and product mix deterioration.",
        "category": "analysis_framework", "topic": "margin bridge compression analysis EBITDA"
    },
    {
        "id": "ma_operating_leverage",
        "text": "Operating leverage: a business with high fixed costs and low variable costs has high operating leverage. This means revenue growth disproportionately boosts EBITDA (positive leverage) but revenue decline disproportionately destroys EBITDA (negative leverage). Manufacturing and airlines have high operating leverage. Consulting and staffing have low operating leverage.",
        "category": "analysis_framework", "topic": "operating leverage fixed variable costs"
    },
    {
        "id": "ma_gross_to_ebitda",
        "text": "Gross margin to EBITDA margin gap analysis: the gap between gross margin and EBITDA margin represents operating expenses as a percentage of revenue. A widening gap indicates opex is growing faster than revenue. A narrowing gap indicates opex efficiency or scaling benefits. Target: opex growing slower than revenue.",
        "category": "analysis_framework", "topic": "gross margin ebitda gap opex efficiency"
    },

    # Budget variance analysis
    {
        "id": "bva_framework",
        "text": "Budget variance analysis framework: decompose total revenue variance into volume variance (units sold vs budgeted units) and price variance (actual price vs budgeted price). For costs, separate volume-driven variances from efficiency variances. Persistent adverse variances in the same line items indicate a forecasting methodology problem, not a one-time event.",
        "category": "variance_analysis", "topic": "budget variance analysis volume price decomposition"
    },
    {
        "id": "bva_ebitda_bridge",
        "text": "EBITDA bridge construction: start from budgeted EBITDA. Add or subtract revenue variance impact on gross profit using actual gross margin rate. Add or subtract cost variances line by line. Each bridge item should be labelled as favourable or adverse, one-time or recurring, and within or outside management control.",
        "category": "variance_analysis", "topic": "EBITDA bridge budget actual variance"
    },
    {
        "id": "bva_forecast_reliability",
        "text": "Management forecast reliability assessment: if actual results are consistently below budget across multiple periods, management is systematically over-optimistic. Calculate the average revenue variance percentage over the last three periods. A consistent negative variance above 5 percent signals a forecasting credibility problem that affects planning and cash management.",
        "category": "variance_analysis", "topic": "forecast accuracy management credibility budget"
    },

    # Red flags and warning signs
    {
        "id": "rf_cash_burn",
        "text": "Cash burn warning signs: declining cash balance alongside negative free cash flow without a clear investment rationale is a serious concern. Calculate months of cash runway (cash balance divided by monthly cash burn). Below 12 months of runway requires immediate management action or refinancing.",
        "category": "red_flags", "topic": "cash burn runway solvency risk"
    },
    {
        "id": "rf_margin_deterioration",
        "text": "Margin deterioration warning signs: three consecutive periods of declining gross margin is a structural signal, not noise. Key causes include raw material inflation, customer mix shifting to lower-margin products, competitive price pressure, or loss of pricing power. Each requires a different management response.",
        "category": "red_flags", "topic": "margin deterioration gross ebitda warning"
    },
    {
        "id": "rf_working_capital_trap",
        "text": "Working capital trap: when a growing business requires disproportionate working capital investment to fund growth, free cash flow can be negative despite profitability. Warning signs: DSO and DIO rising while DPO is flat or declining. This creates a cash funding gap that worsens with revenue growth.",
        "category": "red_flags", "topic": "working capital trap growth cash flow"
    },
    {
        "id": "rf_debt_service",
        "text": "Debt service risk indicators: interest coverage below 2.0x, net debt above 4.0x EBITDA, or significant debt maturing within 12 months without refinancing in place are high-priority risks. Covenant breach risk should be assessed against lender-defined thresholds, which are typically set 15 to 20 percent above the reported metric.",
        "category": "red_flags", "topic": "debt service covenant breach interest coverage"
    },
    {
        "id": "rf_revenue_concentration",
        "text": "Revenue concentration risk: if one customer represents more than 20 percent of total revenue, loss of that customer would materially impact the business. If one product or geography represents more than 50 percent of revenue, the business is exposed to concentration risk. Both situations warrant disclosure and a diversification strategy.",
        "category": "red_flags", "topic": "revenue concentration customer product risk"
    },

    # KPI analysis
    {
        "id": "kpi_return_metrics",
        "text": "Return on equity (ROE) benchmark: above 15 percent is strong. 10 to 15 percent is acceptable. Below 10 percent suggests the business is not generating adequate returns on shareholder capital. Return on assets (ROA) above 5 percent is healthy for asset-intensive businesses.",
        "category": "kpi", "topic": "ROE ROA return on equity assets"
    },
    {
        "id": "kpi_asset_turnover",
        "text": "Asset turnover ratio (revenue divided by total assets) benchmark: above 1.0x means the business generates more than one dollar of revenue per dollar of assets. Retail typically runs 1.5 to 2.5x. Manufacturing 0.6 to 1.2x. Declining asset turnover without revenue decline signals underutilised assets.",
        "category": "kpi", "topic": "asset turnover ratio efficiency"
    },
    {
        "id": "kpi_capex_intensity",
        "text": "Capital expenditure intensity (capex as percentage of revenue) benchmark: manufacturing 5 to 10 percent, retail 1 to 3 percent, SaaS 1 to 5 percent, telecoms 15 to 25 percent. Maintenance capex keeps the business running. Growth capex expands capacity. Understanding the split between maintenance and growth capex is essential for FCF analysis.",
        "category": "kpi", "topic": "capex intensity capital expenditure maintenance growth"
    },

    # Corporate finance frameworks
    {
        "id": "cf_dupont",
        "text": "DuPont decomposition of ROE: ROE equals net margin multiplied by asset turnover multiplied by financial leverage (assets divided by equity). This framework identifies whether ROE is driven by profitability (margin), efficiency (turnover), or leverage. High ROE driven purely by leverage is riskier than ROE driven by margin and efficiency.",
        "category": "framework", "topic": "DuPont ROE decomposition profitability efficiency leverage"
    },
    {
        "id": "cf_valuation_multiples",
        "text": "EV/EBITDA valuation multiples by sector: manufacturing 6 to 10x, retail 5 to 8x, professional services 6 to 10x, SaaS 10 to 25x, technology hardware 8 to 14x. These multiples compress during rising interest rate environments. A company trading below its sector average may be undervalued or facing structural headwinds.",
        "category": "framework", "topic": "EV EBITDA valuation multiples sector"
    },
    {
        "id": "cf_cost_structure",
        "text": "Cost structure analysis: fixed costs do not vary with revenue (rent, salaries, depreciation). Variable costs scale with revenue (raw materials, commissions, delivery). Semi-variable costs have both components (utilities, overtime). A business with high fixed cost proportion has more operating leverage and greater downside risk in a revenue decline scenario.",
        "category": "framework", "topic": "cost structure fixed variable semi-variable"
    },
    {
        "id": "cf_working_capital_management",
        "text": "Working capital optimisation levers: to reduce DSO, tighten credit terms and improve collections processes. To reduce DIO, implement just-in-time inventory or improve demand forecasting. To increase DPO, negotiate longer supplier payment terms (without damaging relationships). Each lever has trade-offs that management must weigh.",
        "category": "framework", "topic": "working capital optimisation DSO DIO DPO management"
    },
    {
        "id": "cf_board_reporting",
        "text": "Board-level financial reporting standards: a board pack should lead with an executive summary of no more than one page covering the period's performance against budget, year-on-year comparison, and the top three risks. Supporting detail follows. The board needs conclusions and recommendations, not just data. Every number presented should have context and a trend.",
        "category": "framework", "topic": "board reporting financial pack executive summary"
    },
    {
        "id": "cf_restructuring_triggers",
        "text": "Financial restructuring trigger points: EBITDA margin below 5 percent for two consecutive periods, interest coverage below 1.5x, net debt above 5x EBITDA, or a covenant breach are typical triggers for a formal financial review. Management actions at this stage include cost reduction programmes, asset disposals, working capital releases, and lender renegotiation.",
        "category": "framework", "topic": "restructuring distress triggers financial review"
    },
    {
        "id": "cf_revenue_quality",
        "text": "Revenue quality assessment criteria: recurring revenue is higher quality than one-time revenue. Contracted revenue is higher quality than order-driven revenue. Revenue from a diversified customer base is higher quality than concentrated revenue. Organic growth is higher quality than acquisition-driven growth. High-quality revenue commands a valuation premium.",
        "category": "framework", "topic": "revenue quality recurring contracted organic growth"
    },
]


# ---------------------------------------------------------------------------
# RAG Pipeline class
# ---------------------------------------------------------------------------

RAG_DB_PATH        = "./rag_db"
RAG_COLLECTION     = "financial_knowledge"


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for financial knowledge.

    On initialisation, seeds a ChromaDB collection with financial benchmarks
    and frameworks. At query time, retrieves the most relevant chunks to
    provide as context to the LLM.

    This is the core of Option 3 in the system architecture.
    """

    def __init__(self):
        self.client     = chromadb.PersistentClient(path=RAG_DB_PATH)
        self.embedder   = FinancialEmbedder()
        self.collection = self.client.get_or_create_collection(
            name             = RAG_COLLECTION,
            embedding_function = self.embedder,
            metadata         = {"description": "Corporate financial benchmarks and frameworks"},
        )
        self._seed()

    def _seed(self):
        """Load knowledge documents into the vector store on first run."""
        existing = set(self.collection.get()["ids"])
        new_docs = [d for d in FINANCIAL_KNOWLEDGE if d["id"] not in existing]

        if new_docs:
            self.collection.add(
                ids       = [d["id"]   for d in new_docs],
                documents = [d["text"] for d in new_docs],
                metadatas = [{"category": d["category"], "topic": d["topic"]} for d in new_docs],
            )

    def retrieve(self, query: str, n_results: int = 5) -> list:
        """
        Find the n most relevant knowledge documents for the given query.

        Returns a list of dicts with keys: text, category, topic, relevance.
        """
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_texts = [query],
            n_results   = min(n_results, count),
            include     = ["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                relevance = max(0.0, 1.0 - dist / 2.0)
                retrieved.append({
                    "text":      doc,
                    "category":  meta.get("category", ""),
                    "topic":     meta.get("topic", ""),
                    "relevance": round(relevance, 3),
                })

        return retrieved

    def context_block(self, query: str, n_results: int = 5) -> str:
        """
        Return a formatted text block of retrieved knowledge
        ready to inject into the LLM prompt.
        """
        chunks = self.retrieve(query, n_results)
        if not chunks:
            return ""

        lines = [
            "RETRIEVED KNOWLEDGE FROM BENCHMARK DATABASE",
            "-" * 50,
            "(Use the following benchmarks and frameworks to contextualise your analysis.)",
            "",
        ]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"[{i}] Topic: {chunk['topic']}")
            lines.append(f"    {chunk['text']}")
            lines.append("")

        lines.append("-" * 50)
        return "\n".join(lines)

    def add_document(self, text: str, category: str = "custom", topic: str = "general") -> str:
        """Add a custom document to the knowledge base at runtime."""
        import hashlib
        doc_id = "custom_" + hashlib.md5(text.encode()).hexdigest()[:10]
        existing = self.collection.get(ids=[doc_id])["ids"]
        if not existing:
            self.collection.add(
                ids       = [doc_id],
                documents = [text],
                metadatas = [{"category": category, "topic": topic}],
            )
            return f"Document added with ID: {doc_id}"
        return f"Document already exists: {doc_id}"

    def stats(self) -> dict:
        docs = self.collection.get(include=["metadatas"])
        categories: dict = {}
        for meta in docs["metadatas"]:
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_documents": self.collection.count(),
            "categories":      categories,
            "db_path":         RAG_DB_PATH,
        }


# ---------------------------------------------------------------------------
# Keep a module-level RAG instance so it is only initialised once
# ---------------------------------------------------------------------------
_rag_instance: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


# ---------------------------------------------------------------------------
# File parsing logic (same as previous version, unchanged)
# ---------------------------------------------------------------------------

PNL_KEYWORDS = {
    "revenue", "sales", "turnover", "income", "gross profit", "ebitda",
    "operating profit", "ebit", "net profit", "net income", "cost of goods",
    "cogs", "gross margin", "operating expenses", "opex", "depreciation",
    "amortisation", "amortization", "interest expense", "tax", "eps",
}
BALANCE_SHEET_KEYWORDS = {
    "assets", "liabilities", "equity", "cash and cash equivalents",
    "accounts receivable", "debtors", "inventory", "stock",
    "property plant", "goodwill", "intangibles", "accounts payable",
    "creditors", "long term debt", "retained earnings", "share capital",
    "total assets", "total liabilities", "shareholders equity",
}
CASH_FLOW_KEYWORDS = {
    "operating activities", "investing activities", "financing activities",
    "capital expenditure", "capex", "dividends paid", "net cash",
    "free cash flow", "cash from operations", "depreciation and amortisation",
}
BUDGET_KEYWORDS = {
    "budget", "actual", "forecast", "variance", "plan", "target",
    "ytd", "mtd", "prior year",
}
KPI_KEYWORDS = {
    "kpi", "metric", "ratio", "margin", "return", "growth", "headcount",
    "customers", "units", "volume", "utilisation", "utilization", "churn",
}


def _clean_value(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    s = re.sub(r"[£$€¥%]", "", s)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _score_sheet(df: pd.DataFrame, keywords: set) -> int:
    text = " ".join(str(c).lower() for c in df.columns)
    text += " " + " ".join(str(v).lower() for v in df.iloc[:, 0] if pd.notna(v))
    return sum(1 for kw in keywords if kw in text)


def _df_to_records(df: pd.DataFrame) -> list:
    records = []
    if df.shape[1] < 2:
        return records
    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
        if not label or label.lower() in ("nan", "none", ""):
            continue
        values = {}
        for col in df.columns[1:]:
            v = _clean_value(row[col])
            if v is not None:
                values[str(col).strip()] = v
        if values:
            records.append({"line_item": label, "values": values})
    return records


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.empty:
        return df
    best_row, best_score = 0, 0
    for i, row in df.head(15).iterrows():
        score = sum(1 for v in row if isinstance(v, str) and len(v.strip()) > 1)
        if score > best_score:
            best_score, best_row = score, i
    if best_row > 0:
        df.columns = [str(v).strip() if pd.notna(v) else f"col_{j}" for j, v in enumerate(df.iloc[best_row])]
        df = df.iloc[best_row + 1:].reset_index(drop=True)
    else:
        df.columns = [str(v).strip() if pd.notna(v) else f"col_{j}" for j, v in enumerate(df.columns)]
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    return df


def parse_file(filepath: str) -> dict:
    path   = Path(filepath)
    suffix = path.suffix.lower()
    result = {
        "filename": path.name, "sheets_found": [],
        "pnl": None, "balance_sheet": None, "cash_flow": None,
        "budget_variance": None, "kpis": None,
        "other_tables": [], "derived_metrics": {},
        "raw_summary": {}, "parse_errors": [],
    }
    sheets = {}
    try:
        if suffix in (".xlsx", ".xls"):
            xf = pd.ExcelFile(filepath)
            result["sheets_found"] = xf.sheet_names
            for name in xf.sheet_names:
                try:
                    sheets[name] = pd.read_excel(filepath, sheet_name=name, header=None)
                except Exception as e:
                    result["parse_errors"].append(f"Sheet '{name}': {e}")
        elif suffix == ".csv":
            sheets["Sheet1"] = pd.read_csv(filepath, header=None)
            result["sheets_found"] = ["Sheet1"]
        else:
            result["parse_errors"].append(f"Unsupported type: {suffix}")
            return result
    except Exception as e:
        result["parse_errors"].append(f"Cannot open file: {e}")
        return result

    scored = []
    for name, raw in sheets.items():
        df = _normalise(raw.copy())
        if df.empty or df.shape[0] < 2:
            continue
        scored.append({
            "name": name, "df": df,
            "pnl":    _score_sheet(df, PNL_KEYWORDS),
            "bs":     _score_sheet(df, BALANCE_SHEET_KEYWORDS),
            "cf":     _score_sheet(df, CASH_FLOW_KEYWORDS),
            "budget": _score_sheet(df, BUDGET_KEYWORDS),
            "kpi":    _score_sheet(df, KPI_KEYWORDS),
        })

    def best(key, min_s=1):
        cands = sorted(scored, key=lambda x: x[key], reverse=True)
        return cands[0] if cands and cands[0][key] >= min_s else None

    assigned = set()
    for sk, rk, mn in [("pnl","pnl",2),("bs","balance_sheet",2),("cf","cash_flow",1),("budget","budget_variance",2),("kpi","kpis",1)]:
        m = best(sk, mn)
        if m and m["name"] not in assigned:
            assigned.add(m["name"])
            result[rk] = _df_to_records(m["df"])
            result["raw_summary"][rk] = m["name"]

    for s in scored:
        if s["name"] not in assigned:
            recs = _df_to_records(s["df"])
            if recs:
                result["other_tables"].append({"sheet": s["name"], "data": recs})

    if result["pnl"]:
        result["derived_metrics"] = _compute_metrics(result["pnl"])

    return result


def _compute_metrics(pnl: list) -> dict:
    def find(rows, *terms):
        for t in terms:
            for r in rows:
                if t.lower() in r["line_item"].lower():
                    return r
        return None

    rev_r  = find(pnl, "revenue", "total revenue", "net revenue", "sales", "turnover")
    gp_r   = find(pnl, "gross profit")
    ebd_r  = find(pnl, "ebitda")
    ebt_r  = find(pnl, "operating profit", "ebit")
    np_r   = find(pnl, "net profit", "net income", "profit after tax")

    periods = list(rev_r["values"].keys()) if rev_r else (list(gp_r["values"].keys()) if gp_r else [])
    metrics: dict = {}

    for p in periods:
        m: dict = {}
        rev   = rev_r["values"].get(p)  if rev_r  else None
        gp    = gp_r["values"].get(p)   if gp_r   else None
        ebd   = ebd_r["values"].get(p)  if ebd_r  else None
        ebt   = ebt_r["values"].get(p)  if ebt_r  else None
        net   = np_r["values"].get(p)   if np_r   else None

        if rev and rev != 0:
            if gp  is not None: m["gross_margin_pct"]     = round(gp  / rev * 100, 1)
            if ebd is not None: m["ebitda_margin_pct"]    = round(ebd / rev * 100, 1)
            if ebt is not None: m["operating_margin_pct"] = round(ebt / rev * 100, 1)
            if net is not None: m["net_margin_pct"]       = round(net / rev * 100, 1)

        for k, v in [("revenue",rev),("gross_profit",gp),("ebitda",ebd),("net_profit",net)]:
            if v is not None:
                m[k] = v
        if m:
            metrics[p] = m

    ks = list(metrics.keys())
    if len(ks) >= 2:
        latest, prior = ks[-1], ks[-2]
        for k in ("revenue","gross_profit","ebitda","net_profit"):
            lv, pv = metrics[latest].get(k), metrics[prior].get(k)
            if lv is not None and pv and pv != 0:
                metrics[latest][f"{k}_yoy_pct"] = round((lv - pv) / abs(pv) * 100, 1)

    return metrics


def summarise_for_agent(parsed: dict) -> str:
    lines = [f"FINANCIAL DATA: {parsed['filename']}", "=" * 60, ""]

    if parsed.get("derived_metrics"):
        lines.append("AUTOMATICALLY DERIVED METRICS")
        lines.append("-" * 40)
        for period, m in parsed["derived_metrics"].items():
            lines.append(f"  Period: {period}")
            for k, v in m.items():
                fmt = f"{v:>14,.1f}" if isinstance(v, float) else f"{v:>14}"
                lines.append(f"    {k:<35} {fmt}")
        lines.append("")

    def section(title, records, max_rows=30):
        if not records:
            return
        lines.append(title)
        lines.append("-" * 40)
        for r in records[:max_rows]:
            vals = "  |  ".join(f"{p}: {v:,.0f}" for p, v in r["values"].items())
            lines.append(f"  {r['line_item']:<38} {vals}")
        if len(records) > max_rows:
            lines.append(f"  ... {len(records)-max_rows} more rows")
        lines.append("")

    section("P&L / INCOME STATEMENT",    parsed.get("pnl"))
    section("BALANCE SHEET",             parsed.get("balance_sheet"))
    section("CASH FLOW STATEMENT",       parsed.get("cash_flow"))
    section("BUDGET VS ACTUAL",          parsed.get("budget_variance"))
    section("KPIs / METRICS",            parsed.get("kpis"))
    for other in parsed.get("other_tables", []):
        section(f"TABLE: {other['sheet']}", other["data"], max_rows=15)

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    rag = get_rag()
    print("RAG stats:", json.dumps(rag.stats(), indent=2))
    print()
    print("Sample retrieval for 'EBITDA margin declining':")
    print(rag.context_block("EBITDA margin declining manufacturing"))
    if len(sys.argv) > 1:
        parsed = parse_file(sys.argv[1])
        print(summarise_for_agent(parsed))