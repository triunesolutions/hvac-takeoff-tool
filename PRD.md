# Product Requirements Document: HVAC AI Takeoff Tool

**Version:** 1.0  
**Date:** April 1, 2026  
**Author:** Triune Solutions + Claude Code  
**Status:** Draft

---

## 1. Problem Statement

Commercial HVAC suppliers and contractors receive construction blueprint sets (PDFs, typically 10-50+ pages per project) and must produce a **takeoff** — a complete Bill of Materials listing every piece of HVAC equipment, its type, quantity, airflow rating, dimensions, and location.

**Current process:**
- A trained estimator opens the PDF in Bluebeam Revu
- Manually identifies which pages are mechanical/HVAC sheets
- Reads the legend to understand the drawing's symbol conventions
- Scans every mechanical page, counting and categorizing each symbol
- Cross-references equipment schedules and spec books
- Enters everything into Excel or quoting software
- **Time per project:** 2-8 hours depending on project size
- **Cost:** $50-150/hour estimator time
- **Error rate:** Manual counting errors are common, leading to missed equipment or over-ordering

**Why this matters:**
- Speed determines how many bids a supplier can respond to
- Faster quoting = more bids = more wins
- Accuracy directly impacts profitability (under-count = change orders, over-count = margin loss)

---

## 2. Product Vision

Build an AI-powered tool that reads HVAC blueprint PDFs and produces a structured takeoff — first as an internal tool for the Triune team, then as a public SaaS product.

**Internal tool (Phase 1-3):** Assist human estimators by auto-detecting 70-90% of equipment and presenting it for review/correction. Humans verify and fix. Every correction improves the system.

**Public product (Phase 4+):** Standalone web application where HVAC suppliers upload blueprints and receive takeoffs, with human-in-the-loop review workflow.

---

## 3. Users

### 3.1 Internal (Phase 1-3)
- **Triune takeoff estimators** — experienced HVAC professionals who currently work in Bluebeam + Excel
- Use the tool to speed up their existing workflow, not replace it
- Trust the tool to do first-pass work, then correct errors

### 3.2 External (Phase 4+)
- **Inside sales staff at HVAC distributors** — need to quote jobs quickly
- **Mechanical contractors** — need equipment lists for bidding
- **Estimating firms** — third-party takeoff services

---

## 4. What a Takeoff Actually Contains

Based on analysis of real blueprint (Beaconsfield Recreation Center, 15-page ventilation set):

### 4.1 Equipment Categories Found on Typical HVAC Drawings

| Category | Tag Prefix | Example Tags | What to Capture |
|---|---|---|---|
| Diffusers (supply) | D- | D-1, D-2, D-4 | Type, airflow (L/s or CFM), neck size |
| Grilles (return/exhaust) | GR- | GR-1, GR-2, GR-3 | Type, size (WxH), airflow |
| Grilles (exhaust) | GE- | GE-1, GE-4 | Type, size, airflow |
| Grilles (supply) | GA- | GA-1, GA-2 | Type, size, airflow |
| VAV Boxes | VAV- | VAV-01 | Model, min/max airflow, inlet size |
| Motorized dampers | VM- | VM-01 through VM-22 | Size, type (opposed/parallel) |
| Transfer air | TA- | TA-1, TA-2 | Size, acoustic lining |
| Fire dampers | BV | BV | Size, rating |
| Serpentins (coils) | SE, SC- | SE, SC-08 | Type (electric/glycol/HW), capacity |
| Humidifiers | HUM- | HUM-01 | Type, capacity |
| Evacuators | EVAC | EVAC | Type, airflow |

### 4.2 Data Points Per Equipment Instance

Each piece of equipment on a drawing has:
1. **Tag ID** — e.g., "D-1" (identifies the type within the project)
2. **Airflow** — e.g., "189 L/s" or "400 CFM"
3. **Size/Dimensions** — e.g., "600x100" (duct connection size) or "ø250" (round)
4. **Quantity at location** — sometimes multiple units at one tag callout
5. **Room/Location** — derived from room label nearby
6. **Page number** — which drawing sheet

### 4.3 Real Data from Sample Blueprint

From the Beaconsfield project (pages 6-7 only):
- **203 total equipment tag instances**
- **52 unique equipment tags**
- **135 airflow values** (L/s)
- **226 dimension values** (WxH or ø)
- Equipment types: 11 diffuser types, 6 grille types, 22 motorized dampers, transfer air, fire dampers, serpentins, humidifier

### 4.4 Expected Output Format

**Summary Table (what the estimator needs):**

| Tag | Description | Quantity | Airflow (L/s) | Size | Notes |
|---|---|---|---|---|---|
| D-1 | Diffuseur carré 4 voies | 8 | 189 | ø250 | |
| D-2 | Diffuseur carré 2 voies | 8 | 125 | ø200 | |
| GR-2 | Grille murale retour | 13 | Various | 600x100 | |
| GE-4 | Grille évacuation | 29 | Various | Various | |
| VM-01 | Volet motorisé | 1 | — | Various | |
| ... | ... | ... | ... | ... | |

---

## 5. Technical Architecture

### 5.1 Core Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  PDF Upload  │───>│ Page Classify │───>│ Tag Detection  │───>│  Data Parse   │
│              │    │ (HVAC pages)  │    │ (OCR + Vision) │    │ (Tag+Value)   │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
                                                                     │
                   ┌──────────────┐    ┌───────────────┐             │
                   │ Annotated PDF│<───│ Cross-Reference│<────────────┘
                   │  + Excel BOM │    │ (Sched+Legend) │
                   └──────────────┘    └───────────────┘
```

### 5.2 Module Breakdown

**Module 1: PDF Ingestion & Page Classification**
- Input: Multi-page PDF blueprint set
- Convert pages to images (PyMuPDF)
- Classify pages by type: cover, legend, floor plan, section, detail, schedule
- Method: Text-based heuristics first (look for "V0xx" sheet numbers, "LÉGENDE", "TABLEAU"), fall back to image classifier
- Output: Ordered list of pages with types

**Module 2: Legend Parser**
- Input: Legend page(s)
- Extract symbol definitions: graphic + name + abbreviation
- Build project-specific symbol dictionary
- Method: Structured text extraction from legend page (symbol descriptions are consistently formatted)
- Output: Dictionary mapping tag prefixes to equipment descriptions

**Module 3: Equipment Tag Detector (PRIMARY DETECTION METHOD)**
- Input: Floor plan page images
- Detect equipment tag labels using OCR (e.g., "D-1", "GR-2", "VM-05")
- Method: OCR the full page, then pattern-match for known tag formats
- This is the primary detection method because:
  - Tags are TEXT — OCR is more reliable than symbol template matching
  - Tags follow predictable patterns (prefix + number)
  - Tags are always present (every equipment callout has a tag)
  - False positive rate is much lower than visual template matching
- Output: List of (tag, x, y, page) for each detected tag

**Module 4: Value Extractor**
- Input: Detected tag positions + full page text
- For each detected tag, find nearby text containing:
  - Airflow values (number followed by "L/s" or "CFM")
  - Dimensions (WxH format or ø + number)
  - Quantity indicators
- Method: Spatial proximity search — look within a radius of each tag position
- Output: Enriched tag records with airflow, dimensions, etc.

**Module 5: Schedule Parser**
- Input: Equipment schedule pages (tables)
- Extract tabular data from schedule sheets
- Method: OCR + table structure detection, or PyMuPDF text extraction with position-based column parsing
- Output: Structured schedule data mapping tag IDs to full specifications

**Module 6: Cross-Reference & BOM Assembly**
- Input: Detected tags + extracted values + schedule data + legend dictionary
- Reconcile floor plan counts with schedule specs
- Flag discrepancies (tag on floor plan not in schedule, or vice versa)
- Output: Complete Bill of Materials

**Module 7: Output Generation**
- Annotated PDF with colored highlights on detected equipment
- Excel/CSV takeoff report
- Accuracy report (when ground truth is available)

### 5.3 Symbol Detection (SECONDARY METHOD — Phase 3+)
- Template matching / object detection as a backup
- Used for equipment that doesn't have a text tag (rare)
- Used to validate OCR detections
- Will improve over time as training data accumulates from human corrections

---

## 6. Phased Roadmap

### Phase 1: Data Foundation & OCR Takeoff (Weeks 1-4)

**Goal:** Working tool that detects equipment tags via OCR and produces a count table.

| Task | Description | Priority |
|---|---|---|
| 1.1 | Bluebeam export parser — ingest team's historical markups as ground truth | P0 |
| 1.2 | OCR tag detector — find all D-x, GR-x, GE-x, GA-x, VAV-x, VM-x, etc. on floor plans | P0 |
| 1.3 | Value extractor — read airflow (L/s) and dimensions near each tag | P0 |
| 1.4 | Basic page classifier — identify mechanical pages by sheet numbering | P1 |
| 1.5 | Excel/CSV output — tag, count, airflow, dimensions per page | P0 |
| 1.6 | Annotated PDF — highlight detected tags with colored boxes | P1 |
| 1.7 | Accuracy scoring — compare tool output vs. Bluebeam ground truth | P0 |

**Success metric:** >80% of equipment tags detected with <10% false positive rate.

**Deliverable:** CLI tool: `python takeoff.py blueprint.pdf` → outputs Excel + annotated PDF.

### Phase 2: Schedule Parsing & Human-in-the-Loop (Weeks 5-8)

**Goal:** Parse equipment schedules and build correction workflow.

| Task | Description | Priority |
|---|---|---|
| 2.1 | Schedule table extractor — parse equipment tables from schedule pages | P0 |
| 2.2 | Legend parser — extract symbol-to-name mapping automatically | P1 |
| 2.3 | Cross-reference engine — match floor plan tags to schedule specs | P0 |
| 2.4 | Correction UI — simple web interface for team to review and fix detections | P0 |
| 2.5 | Feedback loop — every human correction saved as training data | P0 |
| 2.6 | Multi-project support — handle different tag conventions across projects | P1 |

**Success metric:** >90% tag detection, full BOM output matching schedule data.

**Deliverable:** Web UI where estimator uploads PDF, reviews AI takeoff, corrects errors, exports BOM.

### Phase 3: ML Model Training (Weeks 9-14)

**Goal:** Train custom object detection model on accumulated labeled data.

| Task | Description | Priority |
|---|---|---|
| 3.1 | Training data pipeline — convert Bluebeam exports + corrections to YOLO/COCO format | P0 |
| 3.2 | Train symbol detection model (YOLOv8 or RT-DETR) on collected data | P0 |
| 3.3 | Hybrid detection — combine OCR tags + visual detection for higher accuracy | P1 |
| 3.4 | Spec book parser — LLM-based extraction from Division 23 specifications | P2 |
| 3.5 | Confidence scoring — per-detection confidence to prioritize human review | P1 |

**Success metric:** >95% detection with visual model, <5% false positives.

**Deliverable:** Model that detects both tagged and untagged equipment.

### Phase 4: Production SaaS (Weeks 15-22)

**Goal:** Public-facing product for external HVAC suppliers.

| Task | Description | Priority |
|---|---|---|
| 4.1 | Multi-tenant web application (auth, org isolation, project management) | P0 |
| 4.2 | Batch processing — handle 50+ page blueprint sets | P0 |
| 4.3 | Standard output formats — integrate with QuoteSoft, PriceSelect, etc. | P1 |
| 4.4 | Accuracy benchmarking dashboard | P1 |
| 4.5 | Billing & usage tracking | P0 |
| 4.6 | Onboarding & support tooling | P1 |
| 4.7 | Cross-reference with manufacturer catalogs (use existing Triune scraped data) | P2 |

**Deliverable:** SaaS product at app.triune.ai (or equivalent).

### Phase 5: Expansion (Months 6+)

- Plumbing takeoff support
- Electrical takeoff support
- Addendum/revision change detection
- Direct integration with Bluebeam plugin
- Mobile-friendly review interface
- API for third-party integrations

---

## 7. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.12+ | ML ecosystem, team familiarity |
| PDF Processing | PyMuPDF (fitz) | Fast, handles annotations, text extraction |
| OCR | PaddleOCR or EasyOCR | Best accuracy on engineering drawings; Tesseract as fallback |
| Computer Vision | OpenCV + Ultralytics (YOLOv8) | Template matching (Phase 1), object detection (Phase 3) |
| ML Framework | PyTorch | For custom model training in Phase 3 |
| LLM Integration | Claude API | Spec book parsing, ambiguous text understanding |
| Web Framework | FastAPI + React | Phase 2+ UI |
| Database | PostgreSQL + S3 | Structured data + document storage |
| Infrastructure | AWS (existing) or Vercel + Railway | Depends on team preference |
| Export | openpyxl (Excel), reportlab (PDF) | Standard output formats |

---

## 8. Data Strategy

### 8.1 Training Data Sources (in priority order)

1. **Bluebeam markup exports** — richest source: symbol names, positions, counts from human experts
2. **Human corrections in the tool** — every fix the team makes is a labeled sample
3. **Equipment schedule pages** — structured tables that serve as answer keys
4. **Existing project archive** — PDFs from completed projects (even without Bluebeam exports, tag OCR provides labels)

### 8.2 Data Flywheel

```
More Projects Processed
        │
        ▼
More Human Corrections ──> Better Training Data
        │                         │
        ▼                         ▼
Faster Estimating         Better ML Models
        │                         │
        ▼                         ▼
More Projects Quoted ───> Higher Accuracy
        │                         │
        └─────────────────────────┘
```

### 8.3 Data Requirements

- **Phase 1 launch:** 5-10 annotated projects (Bluebeam exports)
- **Phase 3 model training:** 50-100 annotated projects
- **Phase 4 public launch:** 200+ projects for robust generalization
- Each project generates 100-500 labeled equipment instances

---

## 9. Accuracy Targets

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Tag detection recall | >80% | >90% | >95% | >97% |
| False positive rate | <10% | <5% | <3% | <2% |
| Airflow value accuracy | >70% | >85% | >90% | >95% |
| Dimension accuracy | >70% | >85% | >90% | >95% |
| Time per project (human review) | 50% reduction | 65% reduction | 75% reduction | 80% reduction |

---

## 10. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OCR accuracy on dense drawings | Tags misread or missed | Use multiple OCR engines; high-DPI rendering; human review |
| Drawing style variation across firms | Model doesn't generalize | Start with Triune's projects (consistent firms); expand gradually |
| Rotated/angled text on drawings | OCR misses text | Multi-angle OCR; pre-rotation detection |
| French + English mixed content | OCR confused by language | Configure OCR for bilingual; tag patterns are language-agnostic |
| Equipment without tags (rare) | Missed in OCR-first approach | Visual detection as backup (Phase 3) |
| Team adoption resistance | Tool unused | Build AS the team's workflow, not separate from it; start with "assistant" not "replacement" |
| Rebar (withrebar.ai) competitive pressure | Market timing | Focus on internal value first; public launch when accuracy is proven |

---

## 11. Success Criteria for Each Phase

### Phase 1 — "Is this useful?"
- [ ] Tool detects >80% of equipment tags on 3+ real projects
- [ ] Output matches Bluebeam ground truth within 20%
- [ ] At least 2 estimators voluntarily use it on real work
- [ ] Saves measurable time per project

### Phase 2 — "Team depends on it"
- [ ] Every new project runs through the tool first
- [ ] Estimators spend more time reviewing than counting
- [ ] BOM output is directly usable (not just a starting point)
- [ ] 50+ projects processed, generating training data

### Phase 3 — "Ready for others"
- [ ] Visual detection works across different engineering firms' drawing styles
- [ ] <5% error rate on unseen projects
- [ ] Processing time under 2 minutes for a 30-page set

### Phase 4 — "Product-market fit"
- [ ] 5+ external customers using the tool
- [ ] Positive unit economics (revenue > compute + support costs)
- [ ] Net Promoter Score > 40

---

## 12. Non-Goals (Explicitly Out of Scope)

- **Ductwork takeoff** — counting duct lengths, fittings, and sizes (much harder, different product)
- **3D model generation** — we produce 2D counts, not BIM models
- **Automatic purchasing** — we generate BOMs, not purchase orders
- **General construction takeoff** — HVAC only (plumbing/electrical in Phase 5)
- **CAD/Revit file support** — PDF only for now (DWG support is a Phase 5 consideration)
- **Real-time collaboration** — single-user workflow first

---

## 13. Open Questions

1. **Bluebeam integration path** — Should Phase 2 UI be a standalone web app or a Bluebeam plugin? Plugin has lower adoption friction but higher dev cost.
2. **Pricing model** — Per project? Per page? Monthly subscription? Usage-based like Rebar?
3. **Bilingual support** — Quebec projects are French; Ontario/Western Canada are English. Need both from day 1 or start French-only?
4. **Manufacturer cross-reference** — Triune already has scraped product databases. When to integrate? Phase 2 (internal) or Phase 4 (public)?
5. **IP ownership** — If built with Claude Code, what are the IP implications for the public product?

---

## Appendix A: Tag Pattern Reference

Based on analysis of real Beaconsfield project:

| Prefix | Meaning (French) | Meaning (English) | Regex Pattern |
|---|---|---|---|
| D- | Diffuseur | Diffuser | `D-\d{1,2}` |
| GR- | Grille de retour | Return grille | `GR-\d{1,2}` |
| GE- | Grille d'évacuation | Exhaust grille | `GE-\d{1,2}` |
| GA- | Grille d'alimentation | Supply grille | `GA-\d{1,2}` |
| VAV- | Boîte à volume variable | VAV box | `VAV-\d{1,2}` |
| VM- | Volet motorisé | Motorized damper | `VM-\d{1,2}` |
| TA- | Transfert d'air | Air transfer | `TA-\d{1,2}` |
| BV | Bypass valve / Fire damper | Fire/bypass damper | `BV` |
| SE | Serpentin électrique | Electric coil | `SE` |
| SC- | Serpentin de chauffage | Heating coil | `SC-\d{1,2}` |
| HUM- | Humidificateur | Humidifier | `HUM-\d{1,2}` |
| RT- | Roue thermique | Heat wheel | `RT-\d{1,2}` |
| PF- | Pré-filtre | Pre-filter | `PF-\d{1,2}` |
| BFC- | Boîte fin de course | End-of-run box | `BFC-\d{1,2}` |
| BM- | Boîte de mélange | Mixing box | `BM-\d{1,2}` |
| EVAC | Évacuateur | Exhaust fan | `EVAC` |

## Appendix B: Value Patterns

| Value Type | Pattern | Examples |
|---|---|---|
| Airflow (metric) | `\d+ L/s` | 189 L/s, 42 L/s |
| Airflow (imperial) | `\d+ CFM` | 400 CFM |
| Rectangular size | `\d+x\d+` | 600x100, 300x250 |
| Round size | `ø\d+` or `Ø\d+` | ø250, ø150 |
| Quantity | `\d+x` (prefix) | 2x, 3x |
