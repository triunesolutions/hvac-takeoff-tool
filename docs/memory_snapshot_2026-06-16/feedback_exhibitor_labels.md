---
name: Exhibitor list indicator columns are unreliable
description: For Jhaveri Flexo project, ignore YES/DEFINITELY YES labels in exhibitor files — use all rows as raw seed
type: feedback
originSessionId: c2dfc6ac-b37f-412a-a075-1fa1f33dc9d1
---
For the Jhaveri Flexo lead pipeline, the exhibitor files at `C:\Users\JFL\Downloads\` (Alimentaria, Interzoo, ISM Cologne, and future shows) may contain an `indicator` column with values like "DEFINITELY YES" / "YES" / "MAYBE" / "NO" / "NOT SURE". **Do not filter on these labels.**

**Why:** The user told me on 2026-04-16 that these labels are not to be taken seriously — they were prior classification attempts of unknown quality. Filtering on them risks dropping real leads. Jhaveri's own pipeline (scrape + Groq + Gemini text + Gemini vision) should make the fit decision.

**How to apply:**
- When building `seed_universe.csv` from these files, include ALL rows regardless of indicator value
- Keep the `indicator` column as a pass-through metadata field (useful for later comparison: our verdict vs the file's label — interesting but not authoritative)
- Don't repeat the "pre-filter by YES" shortcut in any future ingestion
