# v10 Training Launch — Step-by-Step

**Goal:** Train YOLOv8s v10 on the combined 153-project corpus (124 existing + 29 new sample) and ship it as `models/hvac_yolov8s_v10.pt`.

**Baseline to beat:** v9 median product_recall = **3%**, max 33%, on the 34-project sample benchmark (see `docs/v9_baseline_2026-04-28.md`).

---

## Step 1 — Build the dataset locally (~30–60 min, CPU-only)

This renders every page of every labeled PDF at 200 DPI and tiles them at 640×640 with YOLO labels.

```bash
cd C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool
python train_yolo.py --all --prepare-only
```

**What this writes:**
- `yolo_dataset/images/{train,val}/*.jpg`  — tiles
- `yolo_dataset/labels/{train,val}/*.txt`  — YOLO bbox labels
- `yolo_dataset/dataset.yaml`               — class names + paths

**Sanity-check at the end of the run:**
- Total tiles should be in the **30,000–40,000** range (124 + 29 projects, more pages than v9's 25K).
- Class count should match the dryrun: roughly **35 classes**, with no surprise NEW entries.
- Heaviest classes should be `AD-T-BAR SUPPLY`, `AD-SURF SUPPLY`, `AD-T-BAR RETURN` (all bumped by the new corpus).

If tile count is way off or new unexpected classes appear, **stop** — re-run `dryrun_v10_extract.py` and review.

---

## Step 2 — Zip the dataset (~2–5 min)

```bash
python -c "import zipfile, os; z=zipfile.ZipFile('yolo_dataset.zip','w',zipfile.ZIP_STORED); [z.write(os.path.join(r,f), os.path.relpath(os.path.join(r,f),'.')) for r,_,fs in os.walk('yolo_dataset') for f in fs]; z.close()"
```

This produces `yolo_dataset.zip`. Expect ~1.3–1.8 GB. (`ZIP_STORED` = no compression — JPEGs don't compress further, and Kaggle uploads are bandwidth-limited not CPU-limited.)

---

## Step 3 — Upload to Kaggle as a Dataset

1. Go to **kaggle.com → Datasets → New Dataset**.
2. Drag `yolo_dataset.zip` in.
3. Name: `hvac-yolo-dataset-v10`.
4. Visibility: **Private**. Click **Create**.
5. Wait for "Dataset created successfully" (~5–10 min for 1.5 GB).

---

## Step 4 — Open the v10 training notebook on Kaggle

1. Kaggle → **Code → New Notebook**.
2. **File → Import Notebook** → upload `kaggle_train_v10.ipynb` from this repo.
3. Right panel → **Settings → Accelerator → GPU T4 ×2** (or P100 if T4 unavailable).
4. Right panel → **Add Data** → search `hvac-yolo-dataset-v10` → Add.
5. **Run All**. Training takes ~2–3 hr at 60 epochs.

**Watch for:** the per-epoch checkpoint callback writes `/kaggle/working/hvac_yolov8s_v10.pt` after every improvement. Even if Kaggle disconnects, the best model so far is preserved.

---

## Step 5 — Download the model

When training finishes (or at the end of your 9-hr Kaggle session):

1. Right panel → **Output** tab.
2. Download `hvac_yolov8s_v10.pt` (~22 MB).
3. Place it at `C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool\models\hvac_yolov8s_v10.pt`.

---

## Step 6 — Re-benchmark with v10

```bash
cd C:\Users\JFL\Downloads\Triune\hvac-takeoff-tool
python benchmark_samples.py --model models/hvac_yolov8s_v10.pt
```

(`--model` passthrough already exists in `benchmark_samples.py`.)

This re-runs the same 34 sample projects with v10. Cache is keyed by project, NOT by model — delete `benchmark_output/` first or use `--no-cache` if available, otherwise rename the folder so v10 results don't overwrite v9.

```bash
mv benchmark_output benchmark_output_v9
python benchmark_samples.py --model models/hvac_yolov8s_v10.pt
mv benchmark_output benchmark_output_v10
```

---

## Step 7 — Compare v9 vs v10

The headline metric: **median product_recall**. v9 = 3%. Target ≥ 25%.

```bash
python -c "
import csv
def load(p):
    with open(p) as f: return list(csv.DictReader(f))
v9 = {r['project']: r for r in load('docs/v9_baseline_2026-04-28_results.csv')}
v10 = {r['project']: r for r in load('benchmark_output_v10/benchmark_results.csv')}
print(f'{'project':50s} {'v9_recall':>10s} {'v10_recall':>10s} {'delta':>8s}')
for proj in sorted(v9):
    a = float(v9[proj].get('product_recall') or 0)
    b = float((v10.get(proj) or {}).get('product_recall') or 0)
    print(f'{proj[:50]:50s} {a:>10.0%} {b:>10.0%} {b-a:>+8.0%}')
"
```

**Expected outcomes:**

| Outcome | What it means | Next step |
|---|---|---|
| Median ≥ 25%, holdouts (Sola/Imperial/Alliance/KK) all improve | v10 is a clean win | Promote to production, archive v9 |
| Median ≥ 15%, holdouts improve but not all | Useful but data still thin | Add Label Studio + active learning, plan v11 |
| Median < 15% | More data didn't help → architecture problem | Try YOLOv8m or RT-DETR, look at confusion matrix |
| One holdout regresses badly | Domain mismatch | Investigate that project's drawings, may need more like it |

---

## Holdout discipline

The 4 projects below are **NOT** in the v10 training set. Their post-train numbers ARE the regression bar:

- `4.15.26 Sola Salons` (v9: 33%)
- `4.21.26 677 Imperial Street - TI` (v9: 0%)
- `4.17.26 Alliance Mass Stern Remodel` (v9: 0%)
- `4.13.26 Krispy Kreme #574 - Valencia` (v9: 3%)

If we ever decide to include these in training, we lose our regression bar. Don't.
