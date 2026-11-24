# Data contract

All data is synthetic training material.

## Knowledge articles
Markdown files contain metadata: document_id, title, category, keywords, last_updated and version.

## Tickets
The JSON object is keyed by ticket ID. Each record contains status, assigned_team, priority, summary and last_update.

## Scenario cases
Each case includes case_id, query, expected_intent, expected_route, optional expected_kb/expected_ticket and expected_outcome.

## Safety constraints
Never ask for secrets, invent missing ticket records, bypass approval or claim a simulated escalation created a real ticket.
