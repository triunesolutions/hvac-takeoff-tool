#!/usr/bin/env bash
# Re-run takeoff_cli to generate detections.json, then push to Label Studio
# for each of the 5 high-recall projects.
set -e
cd "C:/Users/JFL/Downloads/Triune/hvac-takeoff-tool"

SAMPLE_ROOT="C:/Users/JFL/Downloads/SAMPLE FILES 27.04.26/SAMPLE FILES 27.04.26"

run_one () {
  local proj_folder="$1"
  local pdf_name="$2"
  local out_dir_name="$3"
  local pdf="$SAMPLE_ROOT/$proj_folder/Plans_Specs/$pdf_name"
  local out="benchmark_output/$out_dir_name"

  echo "============================================================"
  echo "  $out_dir_name"
  echo "============================================================"
  if [ ! -f "$pdf" ]; then
    echo "  MISSING PDF: $pdf"
    return
  fi
  python takeoff_cli.py "$pdf" --model models/hvac_yolov8s_v10.pt --output-dir "$out" 2>&1 | tail -25
  echo
  echo "  → exporting to Label Studio ..."
  python export_to_label_studio.py "$out_dir_name" 2>&1 | tail -8
  echo
}

run_one "4.13.26 Erewhon - Pacific Palisades"           "4.15.26 Erewhon - Pacific Palisades.pdf"           "4.13.26 Erewhon - Pacific Palisades"
run_one "4.13.26 The Bungalow - San Diego"              "4.14.26 The Bungalow - San Diego.pdf"              "4.13.26 The Bungalow - San Diego"
run_one "4.14.26 BMO Santee CA De Novo 2026 Reno"       "4.16.26 BMO Santee CA De Novo 2026 Reno.pdf"       "4.14.26 BMO Santee CA De Novo 2026 Reno"
run_one "4.14.26 Saint Mary's Stadium Clubhouse"        "4.15.26 Saint Mary's Stadium Clubhouse.pdf"        "4.14.26 Saint Mary_s Stadium Clubhouse"
run_one "4.21.26 Anaheim 82"                            "4.22.26 Anaheim 82.pdf"                            "4.21.26 Anaheim 82"

echo "ALL DONE"
