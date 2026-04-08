---
name: Waiting for Bluebeam Export
description: Team is preparing Bluebeam markup CSV export for ground truth data to calibrate the HVAC takeoff tool
type: project
---

User has asked their takeoff team to export Bluebeam markup summaries (CSV) from completed projects.

**Why:** We need ground truth data to measure detection accuracy (precision/recall), tune template matching thresholds scientifically, and eventually train ML models. Currently detecting symbols but can't verify accuracy without knowing the real counts.

**How to apply:** When the Bluebeam CSV arrives, build a parser to extract symbol names, counts, and positions. Then compare against our detection output to compute accuracy metrics. Use the delta to tune thresholds per-symbol-type.

**Current state of the tool (as of 2026-04-08, end of day):**
- v7 model trained on Colab GPU and benchmarked: 81.2% pos / 51.5% full recall (vs v6: 65.9/48)
- 19 consolidated classes via class_aliases.py
- Position recall improved +15 points across the board
- Full recall improved +3.5 points but with surprises:
  - WINS: Shamrock 48->87 (+39!), ARE Campus 65->88, Larchmont 28->43, Flex 230 76->86
  - LOSSES: Columbia Bank 85->49 (-36), Aaron Packaging 58->38 (-20), Mygrant 43->35 (-8)
- Root cause of losses: AD-GRD became dominant (1285 examples), model now over-predicts it instead of AD-T-BAR SUPPLY for Larson-style ceiling diffusers
- Class consolidation went too aggressive (e.g., AD-MISC/LINEAR → AD-LINEAR PLENUM merge hurt Aaron)

**TOMORROW'S NEXT MOVE:**
- Build confusion matrix tool: see exactly which classes are being confused
- Tune class_aliases.py surgically to roll back bad merges
- Train v8 with corrected aliases
- v6 benchmark on 12 projects via Colab GPU: 65.9% pos recall / 48% full recall
  - Excellent: St Elizabeth (94/83), Columbia Bank (96/85)
  - Good: Flex 230 (78/76), ARE Campus Point (65/65)
  - Mediocre: Shamrock (49/48), Mygrant (94/43), Aaron (86/58)
  - Poor: Capitol (8/8), LUS (71/13), Larchmont (38/28), iThink (15/8)
  - Failed: Fort Totten (rendering bug — split raw PDFs)
- Key insight: model finds equipment positions well (~83%) but mislabels classes (class confusion)
- Class consolidation should fix Mygrant/Aaron type failures (high pos / low full)

**Previous state (2026-04-07):**
- v6 model: yolov8s trained on Colab GPU, 23 projects, 75 classes
- Production model: models/hvac_yolov8s_v6.pt (22 MB)
- Fallback: models/hvac_yolov8s_v4.pt (kept for Flex/Haldeman style)

**Training data: 36 organized projects in `data to train/projects/`**
- 01-04 Flex (Plum/Gensler tenant fit-outs, simple)
- 05-13 MMS + SOUTHVAC (mostly large/skipped)
- 14-19 SAMPLE no-raw
- 20-25 Haldeman (6 projects, BACKDRAFT/JET VENT pattern)
- 26-36 Micah/GA Larson (11 projects, AD-GRD/MANUAL VOLUME DAMPER pattern)

**v6 benchmark results (5 projects, conf=0.4):**
- Flex 230: 78% pos / 76% full
- Mission Bay (Haldeman): 92% pos / 78% full
- St Elizabeth (Larson): 94% pos / 83% full ← EXCELLENT
- Aaron Packaging (held-out, Larson): 86% pos / 58% full
- Larchmont (outlier): 38% pos / 28% full ← FAILED
- Overall: 83% pos recall, 68% full recall

**Key insights:**
- Visual position detection generalizes well (~83% across styles)
- Class disambiguation is the bottleneck — model finds equipment but mislabels it
- Larchmont's AD-MISC/LINEAR and FANS classes need more training examples
- YOLO early stopping triggered at epoch 29 due to tiny val set (57 images)
- 75-class vocabulary too large for some rare classes (3-9 examples each)

**Files:**
- train_yolo.py — dataset prep + training (auto-discovers classes, tiles 640x640)
- benchmark.py — runs v6 on any project, computes pos/full recall, saves viz
- colab_train.ipynb — GPU training on Colab (yolov8s, 120 epochs)
- PRD.md — product roadmap
- CLAUDE.md — full project context for new sessions
- models/hvac_yolov8s_v6.pt — production
- models/hvac_yolov8s_v4.pt — fallback
