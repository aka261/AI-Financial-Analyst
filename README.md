# AI Financial Analyst — Agentic AI System

A corporate financial analyst agent built with Groq, ChromaDB, Flask, and MCP.
Upload any company Excel or CSV financials and the agent autonomously analyses
P&L, Balance Sheet, Cash Flow, Budget vs Actual, and KPIs — benchmarking every
metric against industry norms and answering questions like a financial Analyst.

---

## About This Project

This project was built as part of my transition from a non-technical background
into Agentic AI Engineering. Every design decision was made to demonstrate core
agentic concepts from scratch — without hiding complexity behind frameworks.

The system demonstrates four key agentic AI concepts:

- **Agentic Loop** — The agent reasons, calls tools, observes results, and iterates
  until it has enough information to produce a final answer (ReAct pattern)
- **RAG Pipeline** — Before every response, the agent retrieves relevant financial
  benchmarks from a ChromaDB vector store and injects them into the prompt
- **Autonomous Triggering** — A folder watcher detects new files and triggers
  analysis automatically with no human intervention
- **MCP Integration** — All financial tools are exposed via Model Context Protocol
  so any MCP-compatible client can use them

---

## Features

- Upload Excel or CSV financial files via browser drag and drop
- Auto-detects P&L, Balance Sheet, Cash Flow, Budget vs Actual, and KPI sheets
- Derives margins and growth rates automatically from raw data
- RAG knowledge base with 41 financial benchmarks and frameworks
- Autonomous folder watcher that analyses files without any user action
- Saves every analysis as a timestamped report in the reports folder
- MCP server exposing 10 financial tools for use with Claude Desktop
- Multi-turn conversation with full context memory across questions

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core language |
| Groq API | LLM inference — LLaMA 3.3 70B |
| Flask | Web server and browser UI |
| ChromaDB | Vector database for RAG pipeline |
| Pandas | Excel and CSV file parsing |
| Watchdog | Autonomous folder monitoring |
| MCP | Model Context Protocol tool server |

---

## Project Structure

```
AI-Financial-Analyst/
    financial_parser.py     # File parser + RAG pipeline (ChromaDB)
    analyst_agent.py        # Autonomous agent with agentic loop
    server.py               # Flask web server + folder watcher
    mcp_server.py           # MCP server with 10 financial tools
    requirements.txt        # All dependencies
    static/
        index.html          # Browser UI
    uploads/                # Files uploaded via browser (auto-created)
    watched/                # Drop files here for automatic analysis
    reports/                # Auto-generated report files
    rag_db/                 # ChromaDB vector store (auto-created)
```

---

## How It Works

### 1. File Parsing
The parser reads Excel and CSV files using pandas. It scores each sheet against
keyword sets to detect which sheet is a P&L, Balance Sheet, Cash Flow, Budget
vs Actual, or KPI table. It then derives gross margin, EBITDA margin, net margin,
and year-on-year growth rates automatically from the raw numbers.

### 2. RAG Pipeline
On first run, 41 financial knowledge documents are embedded and stored in
ChromaDB. The documents cover industry benchmarks, working capital norms,
earnings quality signals, red flags, and corporate finance frameworks. Before
every agent call, the most relevant documents are retrieved and injected into
the LLM prompt as context. This allows the agent to say "your EBITDA margin of
15% is within the manufacturing benchmark of 8 to 18 percent" rather than
relying on training data alone.

### 3. Agentic Loop
The agent follows the ReAct pattern — Reason, Act, Observe. The LLM reads the
financial data and RAG context, then decides whether it needs to run a
calculation. If it does, it emits a structured tool call. Python executes the
calculation and feeds the result back. The loop repeats up to five times per
question until the agent has a complete answer.

### 4. Folder Watcher
A watchdog observer monitors the watched/ directory in a background thread.
When a new financial file appears, the agent detects it automatically, analyses
it, prints the report to the terminal with a timestamp, and saves it to the
reports/ folder. No browser interaction required.

### 5. MCP Server
Ten financial tools are exposed as MCP tools — parse a file, calculate variance,
build an EBITDA bridge, benchmark a metric, query the RAG database, and more.
Any MCP-compatible client including Claude Desktop can use these tools directly.

---

## Setup Instructions

### Step 1 — Clone the repository
```
git clone https://github.com/aka261/AI-Financial-Analyst.git
cd AI-Financial-Analyst
```

### Step 2 — Create a virtual environment with Python 3.11
```
py -3.11 -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```
pip install -r requirements.txt
```

### Step 4 — Get a free Groq API key
Visit https://console.groq.com/keys and create a free account.
Generate an API key.

### Step 5 — Set the API key
```
# Windows
$env:GROQ_API_KEY = "gsk_..."

# Or create a .env file
echo GROQ_API_KEY=gsk_... > .env
```

### Step 6 — Run the server
```
python server.py
```

### Step 7 — Open the browser
```
http://localhost:5000
```

---

## Usage

### Upload and analyse a file
1. Drag and drop any Excel or CSV financial file into the upload area
2. The agent detects which statements are present
3. Click **Run Full Analysis** for a complete board-level report
4. Or type any specific question in the chat box

### Autonomous folder watcher
Drop any financial file into the `watched/` folder while the server is running.
The agent will detect it within 2 seconds and produce a full report automatically.
The report is printed to the terminal and saved to the `reports/` folder.

### Connect Claude Desktop via MCP
Add this to your Claude Desktop config file:
```json
{
  "mcpServers": {
    "financial-analyst": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "C:\\path\\to\\AI-Financial-Analyst"
    }
  }
}
```

---

## MCP Tools Available

| Tool | Description |
|------|-------------|
| parse_financial_file | Parse an Excel or CSV file |
| calculate_yoy_growth | Year-on-year growth rate |
| calculate_ratio | Financial ratio calculation |
| calculate_variance | Budget vs actual variance |
| calculate_margin | Margin percentage |
| calculate_cagr | Compound annual growth rate |
| build_bridge | EBITDA or variance bridge |
| benchmark_metric | Compare against industry norm |
| analyse_working_capital | DSO, DPO, DIO, CCC |
| retrieve_rag_context | Query the RAG knowledge base |

---

## Sample Questions to Ask

```
What is the revenue growth trend and is it accelerating or decelerating?
```
```
Analyse gross, EBITDA, and net margins and benchmark them against industry norms
```
```
Build an EBITDA bridge from budget to actual
```
```
Is the cash position a concern? How many months of runway does the business have?
```
```
Calculate DSO, DPO, DIO, and the cash conversion cycle
```
```
What are the top three risks the board should know about?
```
```
What should management prioritise in the next 90 days?
```

---

## Requirements

```
groq
flask
pandas
openpyxl
chromadb
watchdog
mcp
anyio
httpx
pydantic
typing-extensions
python-dotenv
```

---

## What I Learned Building This

This project taught me the difference between using AI and building with AI.

Before this project I did not know what a vector database was, what RAG stood
for, or how an agentic loop worked. I built this to learn these concepts by
implementing them rather than just reading about them.

The most important insight was understanding why the agentic loop matters. A
chatbot answers a question in one shot. An agent can pause, calculate something
precise, observe the result, and continue reasoning. That difference is what
makes an agent genuinely useful for financial analysis where numbers must be
exact.

The second important insight was RAG. Without it the model guesses at
benchmarks. With it the model can say your gross margin of 40.7% is within the
manufacturing benchmark of 25 to 45 percent with a specific source. That
changes the quality of the output entirely.

---

## Author

Transitioning from non-technical background into Agentic AI Engineering.
Built to demonstrate practical understanding of RAG, agentic loops, MCP,
autonomous triggering, and corporate financial analysis.

---

