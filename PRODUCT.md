# Product Note

## Which Additional Problem I Addressed
I decided to tackle both of the extra problems: **Proactive Issue Detection** and **Trust & Reliability**. 

For **Proactive Issue Detection**, I built an internal `/ops` dashboard that automatically scans open tickets and orders when it loads. It looks for things like P0 outages, recurring bugs (like CSV upload failures), webhook delays, security incidents, and SLA risks. It flags these directly to the operations team with a one-click "Take Action" button. The agent can also use this system to answer questions like "which accounts are near SLA breach?".

For **Trust & Reliability**, I focused on building hard rules into the code rather than just asking the AI to behave. The system physically cannot state a fee or SLA figure without citing an authoritative document. It pulls exact numbers directly from the PDFs so it never drifts from the actual policy. I also made sure customer contracts override standard policies, and historical tickets are explicitly marked as "non-binding". Finally, security and data access rules are enforced strictly in the Python code, so even if the AI gets confused, it can never access another customer's data.

## What I Would Build Next
1. **Smarter Request Planning**: Right now, the system uses keywords to figure out what the user wants. I'd love to replace this with a structured LLM step that plans out complex, multi-part requests (e.g., "cancel this AND check credit on that").
2. **Better Search (Hybrid Retrieval)**: Our vector search is great for concepts but can struggle with exact IDs (like "KI-211"). Combining it with keyword search (BM25) would make it bulletproof.


## How I Would Measure Success
I would track the **Ops-Accepted Citation Rate**. 
This is the percentage of answers where the operations team accepts the agent's verdict and cited source without needing to edit it. If this rate is high (target 70%+), it means the ops team trusts the system. If it's low, it means the agent is guessing too much.
