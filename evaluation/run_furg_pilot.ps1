$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetPath = Join-Path $PSScriptRoot "furg_fire_dataset"
$eventsPath = Join-Path $PSScriptRoot "furg_events.csv"
$outputPath = Join-Path $PSScriptRoot "furg_pilot_results"
$python = Join-Path $repoRoot "backend\venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Backend Python environment not found. Create backend\venv and install backend\requirements.txt first. See docs\FAST_PILOT_EVALUATION.md."
}

if (-not (Test-Path $datasetPath)) {
    Write-Host "Downloading the FURG fire-video dataset..."
    git clone --depth 1 https://github.com/steffensbola/furg-fire-dataset.git $datasetPath
    if ($LASTEXITCODE -ne 0) { throw "Could not clone the FURG dataset. Check Git and internet connectivity." }
}

Write-Host "Converting per-frame XML annotations to event intervals..."
& $python (Join-Path $PSScriptRoot "prepare_furg_events.py") --dataset $datasetPath --output $eventsPath
if ($LASTEXITCODE -ne 0) { throw "Annotation conversion failed. Review the message above." }

Write-Host "Running Fire Guard AI event-level pilot benchmark..."
& $python (Join-Path $PSScriptRoot "run_video_benchmark.py") --videos $datasetPath --events $eventsPath --output $outputPath --persistence 1 2 3 5 --track-iou 0.30
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed. Review the message above." }

Write-Host ""
Write-Host "Pilot finished. Results folder:"
Write-Host $outputPath
Write-Host "Open summary.csv and event_matches.csv. Preserve all output files and terminal logs."
Write-Host "Reminder: this is exploratory; checkpoint training-data overlap cannot be ruled out."
