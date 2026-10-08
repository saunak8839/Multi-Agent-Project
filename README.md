Project: multi-agent support desk

A customer message comes in, and a graph of agents decides how to handle it.

START
  → intake_agent        (classify intent, extract entities, detect urgency)
  → router (conditional edge)
       ├─ faq_agent         (RAG over the knowledge base)
       ├─ billing_agent     (looks up orders/invoices, can propose refunds)
       ├─ technical_agent   (troubleshooting steps, log lookups)
       └─ escalation_agent  (angry or high-risk, goes straight to a human)
  → draft_reply
  → qa_reviewer_agent     (checks groundedness, tone, policy; can loop back to the specialist)
  → approval_gate         (interrupt: human approves risky actions like refunds)
  → send_reply / execute_action
  → END