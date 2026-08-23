# ParcelPilot Copilot

An AI-powered support and operations copilot that searches authoritative documents, queries operational data, resolves conflicting sources by authority ranking, enforces account-level permissions in the tool layer, detects issues proactively, and only executes state-changing actions after human confirmation.

 It is a LangGraph tool-using agent with a deterministic planner, multiple tools, a conflict resolver, tool-layer ACL, policy parameter extraction from documents, and HITL confirmation.

---

## Architecture

```
User (customer | internal)
         │
         ▼
   ┌─────────────┐
   │  Chat UI    │  
   └─────┬───────┘
         │ SSE stream
         ▼
   ┌─────────────┐
   │  FastAPI    │  Auth context injected on every call
   └─────┬───────┘
         │
         ▼
   ┌───────────────────────────────────────────────────────────┐
   │                    LangGraph Agent                         │
   │                                                           │
   │  ┌─────────┐    ┌────────────────────────────────────┐   │
   │  │CLASSIFY │───▶│ Intent = GENERAL?                   │   │
   │  │(no LLM) │    │   YES → ReAct loop (LLM + tools)   │   │
   │  └─────────┘    │   NO  → Recipe path (deterministic) │   │
   │                  └────────────────────────────────────┘   │
   │                                                           │
   │  RECIPE PATH:                                             │
   │  gather_entities → retrieve_docs → extract_capabilities   │
   │  → compute (multi-entity) → resolve → evidence_gate       │
   │  → [prepare_action → HITL interrupt] → compose            │
   │                                                           │
   │  REACT PATH:                                              │
   │  agent ←→ tools (loop) → resolve → evidence_gate          │
   │  → compose (uses streamed tokens, no overwrite)           │
   │                                                           │
   │  SHARED TAIL:                                             │
   │  compose_answer → Pydantic StructuredAnswer → END         │
   └───────────────────────────────────────────────────────────┘
```

Open `architecture.drawio` in [draw.io]

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | LangGraph (Python) with `interrupt()` for HITL |
| LLM | NVIDIA NIM (configurable model via env) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| API | FastAPI with SSE streaming |
| Frontend | Vanilla JS + HTML/CSS (dark glass UI) |
| Documents | pdfplumber + recursive chunker → PGVector (Supabase) |
| Structured Data | Excel → PostgreSQL via SQLModel |
| Auth | Mock session (4 demo users, token-based) |

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL (Supabase or local) with pgvector extension
- NVIDIA NIM API key

### Environment Variables

Create a `.env` file in the project root:

```env
supabase="postgresql://user:pass@host:port/db"
NVIDIA_NIM_API_KEY=nvapi-xxxxx
NEMOTRON_MODEL=meta/llama-3.1-8b-instruct
NEMOTRON_BASE_URL=https://integrate.api.nvidia.com/v1
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest structured data (Excel → PostgreSQL)
cd backend
python -m app.db.ingest_excel

# Ingest PDFs into vector store
python -m app.rag.ingest_pdfs

# Verify everything is working
python -m scripts.check_pipeline
```

### Running

```bash
# Start the API server
cd backend
uvicorn app.main:app --reload --port 8000

# Open browser
# http://localhost:8000
```

---

## Demo Users

| User | Role | Account | Can See |
|------|------|---------|---------|
| `priya.northstar` | Customer | ACCT-001 (Northstar) | Own orders, tickets, contract + global docs |
| `arjun.lumenworks` | Customer | ACCT-002 (LumenWorks) | Own orders, tickets, contract + global docs |
| `rohit.ops` | Internal Agent | All | All accounts, all data, can create escalations |
| `admin.ops` | Internal Admin | All | Same as rohit.ops + proactive scan |

---

## Key Design Decisions

1. **Tool-layer ACL, not prompt-layer.** Permissions are enforced inside each tool function. The model cannot bypass them even if instructed to.

2. **Deterministic recipe for known intents.** Cancellation, service credit, SLA, product issues, and security questions follow a fixed pipeline. The LLM does not decide whether to retrieve or calculate — the code guarantees it.

3. **Policy parameters extracted from documents.** Fee amounts (INR 250, INR 500), time windows (30 minutes), credit thresholds (2 hours) come from the PDF text, not compiled constants. Change the PDFs → re-ingest → the system updates.

4. **Evidence gate on both paths.** Neither the recipe nor the ReAct path can state a fee, credit or SLA figure without at least one authoritative citation.

5. **Conflict resolver with topic awareness.** Customer contracts override global policy for commercial terms. Product Operations Guide governs product defects. Historical ticket resolutions are always non-binding.

6. **HITL for state changes.** `prepare_action` creates a draft. The graph interrupts. Only after user confirmation does `execute_action` run.

7. **Pydantic-validated answers.** The compose node produces a `StructuredAnswer` model. The frontend receives pre-parsed JSON, never a raw string.

---

## Test Questions

### Login as `arjun.lumenworks` (Customer, LumenWorks)

| # | Question | Expected Answer |
|---|----------|-----------------|
|  | Is LumenWorks eligible for a service credit on order ORD-2002? If so, what is the credit amount? | **INR 300.** Contract: fixed INR 300 if delay > 4h. Standard would be INR 240. |
|  | LumenWorks is getting errors uploading a 4,200-row CSV (TKT-502). Does the Growth plan support this, and what is the workaround? | **5,000 rows supported.** Bug KI-208. Workaround: split below 3,000 rows. TKT-451 was wrong. |

### Login as `rohit.ops` (Internal Agent)

| # | Question | Expected Answer |
|---|----------|-----------------|
| | What should we do about Axis Labs ticket TKT-505 regarding API key exposure? | **Critical security.** Immediate escalation. Confirm card appears. |
|  | What changed between the old support policy and the current one? | Compare deprecated Support Policy v2 and current Support Policy v3 (e.g. cancellation free window changes). |
|  | Does Northstar have premium support? Who is their CSM? | **Yes. CSM: Priya Mehta.** |



---

## Project Structure

```
parcelip/
├── .env
├── README.md
├── architecture.drawio         # draw.io graph diagram
├── requirements.txt
├── data/
│   └── raw/                    # 6 PDFs + Excel (assessment data)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── backend/
    ├── app/
    │   ├── main.py             # FastAPI + SSE endpoints
    │   ├── config.py           # SNAPSHOT_TS, paths, env
    │   ├── auth.py             # 4 demo users
    │   ├── agent/
    │   │   ├── graph.py        # LangGraph state machine
    │   │   ├── planner.py      # Intent classification + recipes
    │   │   ├── resolver.py     # Source conflict resolution
    │   │   ├── prompts.py      # System prompts (compose, react, denial)
    │   │   ├── policy_params.py # Extract fee/SLA numbers from docs
    │   │   └── answer_schema.py # Pydantic StructuredAnswer
    │   ├── db/
    │   │   ├── models.py       # SQLModel: Account, Order, Ticket, etc.
    │   │   ├── engine.py       # PostgreSQL connection
    │   │   └── ingest_excel.py # Excel → DB (lossless)
    │   ├── rag/
    │   │   ├── ingest_pdfs.py  # PDF → PGVector chunks
    │   │   └── registry.yaml   # Document metadata
    │   ├── tools/
    │   │   ├── acl.py          # Account-level access control
    │   │   ├── lookup.py       # get_order, get_ticket, query_*
    │   │   ├── calculator.py   # Deterministic fee/credit/SLA math
    │   │   ├── documents.py    # Vector search with scope filter
    │   │   └── actions.py      # prepare_action, execute_action
    │   └── ops/
    │       └── detectors.py    # Proactive issue detection
    └── scripts/
        └── check_pipeline.py   # Pre-flight diagnostic
```

---



## Confirm/Approve Flow

When the system detects a required escalation (security incidents, SLA breaches), it:

1. Calls `prepare_action` → creates a draft row in the DB (`status=draft`)
2. The graph hits `interrupt()` → SSE sends `pending_action` event
3. Frontend shows a **Confirm / Cancel** card
4. User clicks **Confirm** → `POST /chat/confirm` → graph resumes → `execute_action` sets `status=executed`
5. User clicks **Cancel** → `cancel_action` sets `status=cancelled`

No state change happens without explicit human confirmation.
