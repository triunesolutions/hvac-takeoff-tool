---
name: Focus on Small PDFs First
description: For HVAC Takeoff Tool, restrict dev/test to PDFs ≤15–20 MB until accuracy is solid; bigger files come later
type: project
originSessionId: 59858cdf-981f-44ef-a6f9-684ba643320f
---
For the HVAC Takeoff Tool: only use PDFs roughly **≤15–20 MB** for development and testing. Get the pipeline accurate on these first, then graduate to bigger files.

**Why:** Smaller files iterate fast, avoid pdfplumber memory issues (St Elizabeth at 4+ GB already crashes), and the team is actively sourcing more small PDFs to expand the small-file test set. Validating on this tier first means we don't burn cycles debugging large-file crashes before the core extraction is right.

**How to apply:**
- When picking test/benchmark PDFs, default to files in the 04_Flex_230-class size band (a few MB to ~20 MB).
- When the user says "test it on the project", assume small-file projects unless they name a big one.
- Do NOT silently skip large files — if a user-named PDF is >20 MB, flag it and ask whether to proceed.
- Bigger files (St Elizabeth scale) are explicitly Phase-2 work, not blockers right now.
