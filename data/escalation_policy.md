# Escalation policy (training simulation)

This is fictional policy for the learning project, not a real corporate policy.

Escalate to a human support representative when:
1. The employee explicitly requests a human.
2. The knowledge base says the issue requires IT Support intervention (for example, a locked account).
3. The issue remains unresolved after documented basic troubleshooting.
4. The employee reports a business-critical outage or inability to work.
5. A request requires approval or privileged access; the assistant must not bypass approval controls.

For this exercise, escalation is simulated:
- The graph sets needs_human_review = true.
- The graph records an escalation_reason.
- A human-review node marks the workflow for review.
- No real ticket is created and no external system is contacted.

The assistant must never ask for or expose passwords, MFA codes, recovery codes, private keys or other secrets.
