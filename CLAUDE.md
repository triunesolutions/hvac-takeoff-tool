# HVAC AI Takeoff Tool

## What This Project Is

An AI-powered tool that reads HVAC blueprint PDFs, detects equipment (diffusers, grilles, VAV boxes, dampers, etc.), and produces a structured Bill of Materials (takeoff). Built by Triune Solutions — an HVAC industry company with an in-house takeoff team.

**Goal:** Internal tool first (assist human estimators), then public SaaS product.

## Setup Instructions

### Prerequisites
- Python 3.12+ (developed on 3.14)
- Git
- Claude Code CLI

### Quick Start
```bash
git clone https://github.com/triunesolutions/hvac-takeoff-tool.git
cd hvac-takeoff-tool
pip install PyMuPDF Pillow opencv-python-headless pandas openpyxl easyocr ultralytics
```

### Training Data
Training data is NOT in this repo (large PDFs + sensitive project data). It lives locally at:
```
C:\Users\JFL\Downloads\Triune\data to train\Plans_Specs\
```
Contains 4 projects (Flex 200/210/220/230) with:
- Unlabeled PDFs (model input)
- Labeled "Final" PDFs (human annotations = ground truth)
- Excel takeoffs (expected output)

Ask JFL for access to training data if you need it.

### Run Inference (detect equipment on a PDF)
```bash
# Uses the trained YOLOv8 model (models/hvac_yolov8n_v1.pt)
python -c "
from ultralytics import YOLO
model = YOLO('models/hvac_yolov8n_v1.pt')
results = model.predict('your_blueprint.pdf', conf=0.25)
"
```
See the full inference pipeline in the last section of `visual_detect.py` or run `train_yolo.py` to retrain.

### Retrain the Model
```bash
# 1. Prepare dataset from annotated PDFs
# 2. Train YOLOv8
python train_yolo.py
```

### Resume Training (if interrupted)
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/runs/hvac_detect/weights/last.pt')
model.train(resume=True)
"
```

## Communication Style

Be blunt and honest. Push back when ideas are wrong. Suggest better approaches proactively. No sugarcoating. This is a serious long-term product — give real engineering counsel.

## Project Context

- Triune provided the original 50 blueprint files that Rebar (withrebar.ai) used to bootstrap their product. We're building our own competing solution.
- We have a human takeoff team that uses Bluebeam Revu + Excel. They are the domain experts AND the source of training data.
- The team is providing labeled PDFs (annotations embedded as Polygon annotations with `subject` = product type, `content` = tag).
- Blueprints include both English (US, San Diego — Gensler/Plum Engineering) and French (Quebec) projects.
- A separate codebase exists at `pdf-detection-main` — regex+text based mark detector. Useful for schedule extraction (Phase 2), but cannot detect CAD vector symbols.

## Architecture Decisions Made

1. **YOLO object detection is the primary method.** OCR and template matching were tried and found insufficient — most equipment tags are drawn as CAD vector graphics, not searchable text. YOLOv8n trained on annotated PDFs achieves 88% precision / 51% recall.
2. **Template matching was tried and found insufficient.** Legend symbols don't match floor plan symbols well — too much surrounding clutter. See `detect.py`.
3. **OCR (PyMuPDF + EasyOCR) was tried and found insufficient.** Only 12% recall — most tags are vector graphics, not text objects. See `ocr_detect.py`.
4. **Data flywheel from day 1.** Every project processed generates training data. Human corrections feed back into model improvement.
5. **Tiled inference.** Blueprint pages are large (8400x6000+ px). We tile into 640x640 patches with 160px overlap, run YOLO on each tile, then NMS across tiles.

## Current State (as of April 2, 2026)

### What Exists & Works
- **`models/hvac_yolov8n_v1.pt`** — Trained YOLOv8n model (6MB). 88% precision, 51% recall, 62.5% mAP50. Trained on 4 projects, 248 annotations, 6 classes, 30 epochs on CPU.
- **`train_yolo.py`** — End-to-end: extracts annotations from labeled PDFs → tiles images → trains YOLO.
- **`ocr_detect.py`** — OCR-based detection + accuracy scoring against ground truth. Useful as baseline.
- **`visual_detect.py`** — Hough circle detection + EasyOCR (superseded by YOLO but kept for reference).
- **`detect.py`** / **`takeoff.py`** — Early template matching prototypes (kept for reference).
- **`PRD.md`** — Full product requirements document with 5-phase roadmap.
- **`templates/`** — 37 named HVAC symbol templates from Beaconsfield legend.

### Detection Accuracy Progression
| Method | Precision | Recall | F1 |
|---|---|---|---|
| PyMuPDF text extraction | 70% | 12% | 20% |
| Hough circles + EasyOCR | 34% | 15% | 21% |
| **YOLOv8n (current)** | **88%** | **51%** | **65%** |

### 6 Equipment Classes
```
0: AD-T-BAR SUPPLY      — Square diffuser with circle-A tag
1: AD-T-BAR RETURN       — Square grille with circle-B tag
2: AD-SURF SUPPLY        — Round diffuser with circle-C tag
3: AD-SURF RETURN        — Square return with circle-D tag (TITUS 350R)
4: AD-LINEAR SLOT DIFFUSER — Linear slot diffuser
5: AD-LINEAR PLENUM      — Linear plenum box
```

### What Failed (so you don't repeat it)
- Template matching from legend crops → too many false positives from ductwork
- Hough circle detection → 783 circles per page (ceiling grid, duct connections, etc.)
- Brightness filtering on circles → too aggressive, killed real detections
- Blind threshold tuning without ground truth → guesswork

### Key Technical Details
- Pages are often rotated 270° — annotation coords are in mediabox space, not display space. Use `annot_to_display()` transform.
- Training images are tiled 640x640 with 160px overlap from full-page renders at 200 DPI.
- Annotation bounding boxes in labeled PDFs are ~22.5x27 pts (just the tag bubble). Training uses expanded 45pt radius boxes to capture the full symbol + tag.
- NMS across tiles uses 40px center distance threshold.

## Roadmap (from PRD.md)

- **Phase 1 (DONE):** YOLOv8 detection model trained + accuracy scoring pipeline
- **Phase 2 (NEXT):** More training data + schedule parsing + human-in-the-loop correction UI
- **Phase 3:** Improve model (more data, augmentation, larger YOLO variant)
- **Phase 4:** Public SaaS product
- **Phase 5:** Expand to plumbing and electrical

## File Organization

```
hvac-takeoff-tool/
├── CLAUDE.md              ← You are here
├── PRD.md                 ← Full product requirements document
├── train_yolo.py          ← Dataset prep + YOLO training pipeline
├── ocr_detect.py          ← OCR detection + accuracy scoring
├── visual_detect.py       ← Circle detection + EasyOCR (reference)
├── detect.py              ← Template matching prototype (reference)
├── takeoff.py             ← Initial pipeline prototype (reference)
├── models/
│   └── hvac_yolov8n_v1.pt ← Trained YOLOv8n weights (6MB)
├── templates/             ← 37 HVAC symbol templates from legend
│   └── symbol_map.json
├── output/                ← Detection results (gitignored)
├── runs/                  ← YOLO training runs (gitignored)
└── yolo_dataset/          ← Training tiles (gitignored)
```

## Next Steps (Priority Order)

1. **Add more training data** — more labeled projects = better recall. Need both unlabeled + labeled PDFs.
2. **Retrain with augmentation** — rotation, scaling, brightness jitter to improve generalization.
3. **Build inference script** — upload PDF → run YOLO → output annotated PDF + Excel BOM.
4. **Port schedule extraction** from `pdf-detection-main` into our pipeline.
5. **Test on French blueprints** (Beaconsfield) to check cross-style generalization.
6. **Build Streamlit UI** for team to review detections and correct errors.
7. **Integrate with Triune's manufacturer cross-reference databases** (already scraped).

## Don't Do

- Don't over-engineer infrastructure before core detection is >80% recall
- Don't build a web UI until the model is good enough for the team to use daily
- Don't try to generalize across all drawing styles yet — nail the current training projects first, then expand
- Don't use Claude Vision API as primary detection (cost/latency at scale) — save it for spec book parsing
- Don't commit training data (PDFs) or large artifacts (yolo_dataset/, runs/) to the repo
