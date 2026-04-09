# HVAC AI Takeoff Tool — A Plain-English Guide

*For non-technical readers. Last updated: April 8, 2026.*

---

## What We're Building

A tool that reads HVAC blueprint PDFs and produces a takeoff (a list of all equipment with counts) — the same thing your team currently does manually in Bluebeam, but in seconds instead of hours.

**Goal:** Internal tool first. Public SaaS product later.

---

## How It Works (Simple Version)

Imagine teaching a child to recognize dogs. You show them 1,000 pictures of dogs and say "this is a dog". After enough examples, the child can spot a dog they've never seen before.

That's exactly what we're doing — except instead of dogs, we're teaching the computer to recognize HVAC equipment symbols (diffusers, grilles, dampers, fans, etc.).

**The 3 steps:**

1. **Show examples** — Your team's old projects (where they marked equipment in Bluebeam) become "answer keys" the computer learns from.
2. **Train the model** — The computer studies these examples on a powerful GPU computer (we use Google Colab — free) until it can recognize patterns.
3. **Use the model** — Upload a new blueprint, the computer marks all the equipment it sees, you review and correct.

Every correction your team makes becomes a new training example, so the system gets smarter over time.

---

## The Words People Throw Around

| Term | What it means in plain English |
|---|---|
| **Model** | The "brain" of the system. It's just a file (~22MB) that knows how to spot HVAC equipment. We're on version 7 (v7). |
| **Training** | Teaching the brain. Takes ~45 minutes on a GPU. |
| **Inference** | Asking the brain to do a job. Takes ~5 seconds per page. |
| **Annotation** | When your team draws a box around a diffuser in Bluebeam, that's an annotation. |
| **Ground truth** | What the right answer SHOULD be (your team's annotations). |
| **Class** | A type of equipment (e.g., "T-bar supply diffuser" or "exhaust fan"). |
| **GPU vs CPU** | GPU is the fast computer (Google Colab). CPU is your laptop — too slow for training. |
| **Dataset** | The collection of all training examples. Currently ~5,500 equipment instances from 23 projects. |
| **Confidence** | How sure the computer is about a detection (0-100%). We typically accept anything above 40%. |

### The Two Numbers That Matter Most

When we test the model, we measure two things:

**Recall = "Did we find everything?"**
- If a drawing has 100 diffusers and the model finds 81, that's **81% recall**.
- High recall = nothing is missed.
- *Currently: 81% — we find 4 out of every 5 pieces of equipment.*

**Precision = "Are our answers correct?"**
- If the model says it found 100 diffusers and 75 of them are actually diffusers, that's **75% precision**.
- High precision = no false alarms.
- *Currently: 62% — about 2 out of every 3 answers are correct.*

The dream is **100% recall AND 100% precision**. In practice, you trade them off.

We also track a stricter version called **"full match recall"** — we found the equipment AND labeled it with the right type. *Currently: 51%.*

---

## Where We Are Right Now (April 8, 2026)

| What | Status |
|---|---|
| Model finds equipment positions | ✅ **81%** of the time — works |
| Model labels equipment correctly | 🟡 **51%** of the time — needs work |
| Outputs annotated PDF | ✅ Works |
| Outputs Excel takeoff with counts | ⏳ Not yet built |
| Outputs CFM/dimensions | ⏳ Not yet built |
| Has a user interface | ⏳ Phase 2 — not yet |
| Has test data from 36 real projects | ✅ Done |

### The Honest Score

For a typical project, the tool will:
- Find about **4 out of 5** pieces of equipment correctly
- Get about **half** of them labeled with the exact right product type
- Save your team **most of the manual counting time**, but they still need to review and correct

We are **not yet at production quality**. We're at "useful internal helper" quality.

---

## Why Some Projects Work Better Than Others

Different engineering firms draw HVAC symbols differently. We have 3 main "drawing styles" in our training data:

| Style | Example Projects | Accuracy |
|---|---|---|
| **Flex/Plum** (Gensler tenant fit-outs) | Flex 200/210/220/230 | 86% — excellent |
| **Haldeman** (Aritzia, Yucaipa, Mission Bay) | Aritzia, Shamrock, ARE Campus | 87% — excellent |
| **Larson/Micah** (St Elizabeth, Aaron Packaging) | St Elizabeth, Mygrant, Larchmont | 35-85% — varies |

The model is great at styles it has seen many examples of. New styles need more training data.

---

## What's Next

### Immediate (this week)
**Fix the class confusion problem.** The model is good at finding equipment but sometimes mislabels it. We need to look at WHICH labels it confuses with WHICH (called a "confusion matrix") and fix the training data accordingly.

### Short term (next 2-3 weeks)
- Add value extraction (read CFM ratings, neck sizes, dimensions)
- Build the Excel BOM output in your team's format
- Test on 10+ unseen projects

### Medium term (1-2 months)
- Build a simple web tool the team can use daily (Phase 2)
- Add human-in-the-loop correction (every fix becomes training data)
- Get to 90%+ accuracy on all common drawing styles

### Long term (3-6 months)
- Public SaaS launch
- Expand to plumbing and electrical takeoffs

---

## How To Talk About This in One Sentence

> "We're building an AI that reads HVAC blueprints and produces equipment takeoffs automatically. It currently finds 81% of equipment correctly. We're tuning it to be production-ready over the next few weeks."

---

## How We Compare to Rebar (the competition)

Rebar (withrebar.ai) is the AI HVAC takeoff company that raised $14M and uses our team's data to bootstrap their model. Their approach is the same as ours — visual pattern recognition trained on labeled blueprints. They just have:
- More training data (millions of files vs our 5,500 instances)
- More compute (production GPUs vs free Colab)
- A polished web app

We're early. We have the same fundamental tech and a real edge: domain expertise and a takeoff team that generates training data daily. With more annotated projects, we can match their accuracy.

---

## Glossary of Files in the Repo

| File | What it does |
|---|---|
| `train_yolo.py` | Teaches a new model from your team's labeled projects |
| `benchmark.py` | Tests how accurate the model is on real projects |
| `class_aliases.py` | Fixes typos and merges duplicate equipment names |
| `colab_train.ipynb` | The notebook we run on Google Colab to train (free GPU) |
| `models/hvac_yolov8s_v7.pt` | The current best model (the "brain") |
| `data to train/projects/` | All 36 organized projects with labeled examples |
| `PRD.md` | The full product roadmap |
| `CLAUDE.md` | Technical context for engineers |
| `WHAT_WE_ARE_BUILDING.md` | This document |
