# Fire Guard AI research evaluation protocol

## Purpose and evidence boundary
This branch adds an offline comparison between the existing class-level persistence rule and an experimental IoU-linked verifier. The verifier is NOT integrated into live alerts. The evaluation script runs the detector once on sampled video frames, then replays the same detections through both policies. It reports event-level metrics only; it does not train a detector or compute mAP.

The bundled checkpoint reports the Fire class only. Do not claim that this checkpoint detects smoke. No benchmark result is included with this branch; run the labelled test described below and report only the generated metrics.

## Dataset recommendation
For an initial event-level experiment, start with the MIVIA Fire Detection Dataset. Its official page describes 31 videos: 14 with fire and 17 negative clips, including difficult scenes with red objects, smoke and clouds. It is a small/older video set, so treat it as an exploratory benchmark, not evidence of broad generalization. MIVIA requires registration for dataset downloads; check permitted use, attribution and redistribution terms before using or sharing the files.

- Dataset: https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/
- Registration/download: https://mivia.unisa.it/datasets-request/

For a separate frame-level image evaluation, MIVIA LFDN describes 36,554 annotated images, including hard negatives such as sunlight, reflections, glare, fog, headlights, dust, clothes and flags: https://mivia.unisa.it/large-fire-dataset-with-negative-samples-lfdn/. D-Fire is another image dataset with YOLO-format bounding boxes: https://github.com/gaia-solutions-on-demand/DFireDataset. Image datasets do not replace the video clips required for event delay and false-alert-per-hour evaluation.

## Dataset partition and annotation
Create separate folders; split by original recording/camera/scene, never random adjacent frames:
\`\`\`
evaluation/
  validation_videos/
  test_videos/
  validation_events.csv
  test_events.csv
\`\`\`
Use validation videos to choose confidence, persistence and track-IoU thresholds. Freeze the configuration before running the held-out test set. If only 31 videos are available, disclose the small sample and report exact positive/negative counts.

Ground-truth CSV columns:
\`\`\`csv
video,event_id,class,start_ms,end_ms
warehouse_01.mp4,F01,fire,12500,28700
\`\`\`
Times are milliseconds from the beginning of the video. Use one row per continuous visible fire incident. Start is the first visually observable flame frame; end is the last visible frame. For a negative clip, add no event row; its full duration still contributes to exposure hours. Use the video filename exactly. The sample row is illustrative only.

## Windows setup
Clone this branch:
\`\`\`powershell
git clone --branch research-evaluation-kit https://github.com/Atharva-p82/AI-fire-detection.git
cd AI-fire-detection
\`\`\`
Use the backend virtual environment already documented by the project. If creating it from scratch:
\`\`\`powershell
cd backend
py -3.12 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" "torchvision==0.20.1+cpu"
pip install -r requirements.txt
cd ..
\`\`\`
Python 3.12 is the version documented as validated in the project README. The PyTorch command shown is the repository's CPU setup; do not overwrite a working CUDA environment with CPU packages. Run the benchmark from the repository root with the backend environment active. By default it loads backend/best.pt. To select another checkpoint set:
\`\`\`powershell
$env:FIRE_MODEL_PATH="C:\\full\\path\\to\\best.pt"
\`\`\`

## Run the comparison
Validation run:
\`\`\`powershell
python evaluation\\run_video_benchmark.py --videos evaluation\\validation_videos --events evaluation\\validation_events.csv --output evaluation\\results_validation --persistence 1 2 3 5 --track-iou 0.30
\`\`\`
Try track-IoU values such as 0.50 and 0.70 on validation only, storing each run in a different output folder. Select one setting using validation results.

Held-out test run (replace 0.50 with the selected validation setting):
\`\`\`powershell
python evaluation\\run_video_benchmark.py --videos evaluation\\test_videos --events evaluation\\test_events.csv --output evaluation\\results_test --persistence 1 2 3 5 --track-iou 0.50
\`\`\`
Defaults: detector confidence 0.25, NMS IoU 0.45, every third decoded frame, incident re-arm 10 seconds and event matching tolerance 30 seconds. Keep settings identical across policies. The optional \`--roi-json\` accepts a JSON object mapping video filenames to normalized polygon points. The script evaluates both ROI-off and ROI-on configurations when an ROI file is supplied.

## Output files
- \`raw_detections.csv\`: raw model boxes/confidence from sampled frames.
- \`predicted_events.csv\`: grouped events per policy/configuration.
- \`summary.csv\` and \`summary.json\`: event metrics and run interpretation.
- \`event_matches.csv\`: event TP/FP matching and matched delay.
- \`run_metadata.json\`: checkpoint hash, classes, platform/software, frame counts, video duration and configuration.

Keep all output files with the report. They are the reproducibility record.

## Metrics
- Event precision = matched predicted events / all predicted events.
- Event recall = matched ground-truth events / all ground-truth events.
- Event F1 = harmonic mean of event precision and recall.
- False alerts per camera-hour = unmatched predicted events / monitored video hours.
- Detection delay = predicted event start - annotated first-visible time for a matched event.

The matching rule accepts a prediction from the ground-truth start through ground-truth end plus the configured tolerance. Report this tolerance. Event precision is not frame-level detector precision. This script does not calculate bounding-box precision/recall or mAP@0.5 / mAP@0.5:0.95. Those require frame-level boxes and a separate object-detection evaluation.

Higher persistence may reduce false alerts but increase delay or miss short fires. IoU linking can reject an unrelated object inheriting another track's history, but may also lose a track if the flame shape/position changes or a sampled detection disappears. Report trade-offs, not only the best-looking metric.

## Runtime experiment is separate
The offline evaluator does not measure full live-camera latency, external email/Telegram delivery, multi-camera throughput or reconnect performance. Measure these in the actual app. Record CPU/GPU, RAM, OS, camera resolution, codec, source FPS and stream type. Test one and two cameras (and four only if hardware permits). Record camera API metrics \`frames_processed\`, \`inference_ms\`, \`processing_fps\` and \`last_frame_at\` at fixed intervals. Separate raw camera FPS from model inference FPS. Inject a controlled camera disconnect and log reconnect time. Use a test email receiver/bot and record provider acknowledgements; a database event does not prove delivery. Repeat each setup at least three times and report all runs and the median.

Do not create a real fire for experiments. Use legally accessible prerecorded footage or qualified, institution-approved facilities. This system is not certified life-safety equipment.

## Software checks
From backend:
\`\`\`powershell
cd backend
python -m unittest discover -s tests -v
\`\`\`
From frontend:
\`\`\`powershell
cd frontend
npm ci
npm run lint
npm run build
\`\`\`
Save full terminal output and report exact pass/fail results. A static count of test methods is not evidence the suite passes.

## Submission checklist
- Confirm the guides' formal designations with the department. The manuscript uses the user-supplied names: Dr Kaushiki Mam and Dr Mehboob Alam Sir.
- Run all backend tests and frontend lint/build; retain logs.
- Split by original video/scene and freeze thresholds before held-out testing.
- Populate paper result tables only from test outputs; preserve raw CSV/JSON and checkpoint hash.
- Report mAP only if a separate frame-level box evaluation is done.
- Keep the current Fire-only claim unless a different model's metadata and evaluation support more.
- Capture genuine screenshots of model classes, ROI editor, live feed, event history and test outputs. Hide credentials and private camera content.
- Confirm author contributions, citations, dataset terms, institutional format and permissions.
- Describe Fire Guard AI as a prototype/decision-support tool, not a certified alarm replacement.
