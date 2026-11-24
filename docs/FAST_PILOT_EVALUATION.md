# Fast pilot evaluation using FURG fire videos

## Why this route
For a quick, reproducible pilot, this branch includes `evaluation/prepare_furg_events.py`. It converts FURG's OpenCV XML frame annotations into the event CSV required by `run_video_benchmark.py`.

FURG repository: https://github.com/steffensbola/furg-fire-dataset

The FURG README describes a video dataset with one XML annotation file per video; each XML contains frame-numbered flame bounding boxes. The repository declares CC0-1.0. This is a practical pilot dataset because the clips and annotations are directly available from GitHub and do not require the MIVIA registration route.

## Annotation conversion check
The first successful XML conversion processed all 23 annotation files: 17 videos contained flame bounding boxes, 6 contained no flame boxes, and 34 fire-event intervals were derived. These are counts from the dataset annotations only, not detection results or evidence of accuracy. The six videos without flame boxes should be described as unannotated/no-box clips unless their negative status is independently confirmed.

## Important limitations
- This is a small exploratory pilot, not a large or definitive benchmark.
- The bundled `best.pt` checkpoint's training-data provenance is undocumented. It is therefore not possible to prove that FURG videos were excluded from the model's training data. Report this limitation explicitly.
- XML box annotations are reduced to time intervals from the first and last annotated frames (with short annotation gaps merged). The evaluation is event-level and temporal/class-based; it does not compute bounding-box mAP or spatial IoU accuracy.
- The reported false-alert rate is per monitored video-hour because this is an offline video set. Do not label it camera-hour.
- Run all configurations and report the complete table. Do not select the most favourable row and present it as the only result.

## Windows commands

### 1. Clone the Fire Guard AI research branch
```powershell
git clone --branch research-evaluation-kit https://github.com/Atharva-p82/AI-fire-detection.git
cd AI-fire-detection
```

If the repository is already cloned, use `git switch research-evaluation-kit` and `git pull` instead.

### 2. Prepare Python 3.12
Run from the repository root:
```powershell
py -3.12 -m venv backend\venv
backend\venv\Scripts\python.exe -m pip install --upgrade pip
backend\venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" "torchvision==0.20.1+cpu"
backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```
This is the CPU setup. Do not replace a working CUDA setup with CPU-only packages. Verify that `backend\best.pt` exists before benchmarking.

### 3. One-command pilot (after Python setup)
The branch includes a Windows script that downloads FURG if needed, converts the annotations, and runs the first benchmark configuration:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\evaluation\run_furg_pilot.ps1
```

It expects `backend\venv\Scripts\python.exe` to exist from Step 2. It writes results to `evaluation/furg_pilot_results`.

### Manual alternative
If you prefer to run each stage separately, download the dataset:

```powershell
git clone --depth 1 https://github.com/steffensbola/furg-fire-dataset.git evaluation\furg_fire_dataset
```

Convert its XML annotations to event timestamps:

```powershell
backend\venv\Scripts\python.exe evaluation\prepare_furg_events.py --dataset evaluation\furg_fire_dataset --output evaluation\furg_events.csv
```
The utility prints counts of positive videos, negative videos and derived events. Open `evaluation/furg_events.csv` and spot-check a few rows against the corresponding videos. Do not edit timestamps to make the detector look better.

Run the pilot benchmark:

```powershell
backend\venv\Scripts\python.exe evaluation\run_video_benchmark.py --videos evaluation\furg_fire_dataset --events evaluation\furg_events.csv --output evaluation\furg_pilot_results --persistence 1 2 3 5 --track-iou 0.30
```
This runs the existing class-level persistence policy and the experimental IoU-linked comparator on the same detector outputs, across persistence values 1, 2, 3 and 5.

Optional sensitivity runs can use track IoU 0.50 and 0.70, each with a different output folder:

```powershell
backend\venv\Scripts\python.exe evaluation\run_video_benchmark.py --videos evaluation\furg_fire_dataset --events evaluation\furg_events.csv --output evaluation\furg_pilot_iou050 --persistence 1 2 3 5 --track-iou 0.50
backend\venv\Scripts\python.exe evaluation\run_video_benchmark.py --videos evaluation\furg_fire_dataset --events evaluation\furg_events.csv --output evaluation\furg_pilot_iou070 --persistence 1 2 3 5 --track-iou 0.70
```

### Inspect the outputs
- `summary.csv`: event precision, event recall, event F1, false alerts per video-hour and detection delay for each policy/configuration.
- `event_matches.csv`: TP/FP event match for each predicted event.
- `raw_detections.csv`: model boxes/confidences on sampled frames.
- `predicted_events.csv`: grouped events from each policy.
- `run_metadata.json`: checkpoint hash, model classes, Python/OpenCV/platform details and run settings.

Keep these files unchanged as raw evidence. Save a copy of the terminal output as well.

## How to use the results in the presentation
- Slide 17: identify the FURG pilot dataset and state the training-provenance limitation.
- Slide 18–19: report the event-level comparison and use the label "false alerts per video-hour."
- Slide 20: fill the results table only from `summary.csv`; do not report mAP from this script.
- Slide 21: insert genuine screenshots of the live camera feed, detection overlay/confidence and event history. These are implementation screenshots, not benchmark metrics.

## Suggested honest wording
"An exploratory event-level pilot was conducted on the FURG video dataset using its frame-level flame annotations converted into event intervals. Because the provenance of the bundled checkpoint's training data is unavailable, overlap between training and pilot data cannot be excluded. The results are therefore reported as a preliminary system evaluation, not as an independent generalization benchmark."
