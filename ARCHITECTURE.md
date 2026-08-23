# Architecture Note

## Agent Design
The Copilot uses a two-path system (built with LangGraph):
1. **The Recipe Path (Strict)**: When a user asks about something we have strict rules for—like cancellations, service credits, or SLA targets—the system follows a hardcoded path. It guarantees the document is read, the math is done correctly, and the sources are checked. The AI just narrates the final result.
2. **The ReAct Path (Flexible)**: For general questions or summarizing data, the system falls back to a standard AI agent that can pick and choose which tools to use.

I chose this design because relying purely on an AI agent caused it to skip retrieving documents and hallucinate answers. Forcing it down a strict path for critical tasks fixed this entirely.

## Tool Design
The system has a specific set of tools that allow it to read data, calculate answers, and safely take actions:

**Search Tools**
- **`search_documents`**: Scans the PDF database (PGVector) for relevant policy documents. Crucially, it filters out anything the user isn't allowed to see, so customers can never accidentally search another company's private contract.

**Lookup Tools**
- **`get_account`, `get_order`, `get_ticket`**: Fetches a single, specific record using its ID. If a customer tries to fetch an ID belonging to another company, the tool simply returns "not found" to protect data privacy.
- **`query_orders`, `query_tickets`**: Allows the AI to search lists (like "show me all open tickets"). For customers, the system automatically forces a filter so they only get their own account's records back.

**Calculation Tools**
- **`cancellation_fee`**: A math function that calculates how much a customer owes for cancelling an order, based on the hours passed and whether they have a waiver.
- **`service_credit`**: Calculates how much refund credit a customer gets if a delivery was delayed, factoring in carrier faults and specific contract terms.
- **`sla_remaining_minutes`**: Checks exactly how much time is left to resolve a ticket before it breaks the SLA target.

**Action & Ops Tools**
- **`prepare_action`**: Used to draft an escalation or follow-up task. It *never* actually executes the action itself; it just saves a draft to the database.
- **`execute_action`**: Finalizes the draft created by `prepare_action`. It only works if a human has explicitly clicked "Confirm" on the frontend UI.
- **`detect_issues`**: A special tool strictly for internal operations staff. It runs the proactive scanners to find P0 outages, recurring bugs, or SLA risks across the whole system.

All tools use a frozen snapshot time (`August 16, 2026`) so calculations are always consistent and testable.

## Handling Documents and Data
- **Documents**: The 6 PDFs are chopped into chunks and stored in a PGVector database. 
- **Structured Data**: The Excel data was safely moved into a proper PostgreSQL database.
- **Smart Policies**: Instead of hardcoding numbers (like a "30-minute SLA" or "250 INR fee") into the code, the system actually reads the PDFs, extracts the numbers, and uses them in its calculators. If you update the PDF, the system's logic automatically updates!

## Trusting the Right Sources
When the system finds multiple documents, it ranks them by authority:
- **100**: Customer-specific contracts (these override everything else).
- **90**: Current global policies (like Support Policy v3).
- **75**: Product Guides (these win if the issue is a software bug).
- **15**: Old, closed tickets (these are strictly used for context and marked as "non-binding" so the AI doesn't treat them as a rule).

## Major Technical Trade-offs
- **Keyword Routing vs. AI Routing**: I used keyword matching to decide if a query goes down the "Strict" path. It's fast and reliable, but it struggles if a user asks two completely different questions in one sentence.
- **One Big Model**: I used a single LLM for everything. It's simpler to host, but sometimes a smaller specialized model could be faster for basic tasks.

