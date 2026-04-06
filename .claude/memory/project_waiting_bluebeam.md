---
name: Waiting for Bluebeam Export
description: Team is preparing Bluebeam markup CSV export for ground truth data to calibrate the HVAC takeoff tool
type: project
---

User has asked their takeoff team to export Bluebeam markup summaries (CSV) from completed projects.

**Why:** We need ground truth data to measure detection accuracy (precision/recall), tune template matching thresholds scientifically, and eventually train ML models. Currently detecting symbols but can't verify accuracy without knowing the real counts.

**How to apply:** When the Bluebeam CSV arrives, build a parser to extract symbol names, counts, and positions. Then compare against our detection output to compute accuracy metrics. Use the delta to tune thresholds per-symbol-type.

**Current state of the tool (as of 2026-04-02):**
- YOLOv8n model trained: 88% precision, 51% recall, 62.5% mAP50
- Model weights: models/hvac_yolov8n_v1.pt (6MB)
- Trained on 4 projects (Flex 200/210/220/230) with 248 annotations, 6 classes
- Inference pipeline working: tiles page → runs YOLO → NMS → annotated output
- GitHub repo: triunesolutions/hvac-takeoff-tool (private)
- Code: train_yolo.py, visual_detect.py, ocr_detect.py, detect.py
- Also reviewed pdf-detection-main codebase — will integrate schedule extraction later
- Bluebeam CSV exports still pending from team
