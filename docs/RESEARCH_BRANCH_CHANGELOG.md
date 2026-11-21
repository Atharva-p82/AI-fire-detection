# Research branch change log

Branch: `research-evaluation-kit`

## Integrated changes
1. `backend/camera_manager.py` passes camera_id to the notification worker.
2. `backend/notification.py` replaces the fixed process-global 60-second gate with a configurable per-camera email/Telegram attempt cooldown using `ai_cooldown_seconds`, monotonic time and a lock. Test notifications use a separate scope. Event creation is not throttled; the in-memory limiter resets after a backend restart.
3. `frontend/src/components/SettingsModal.tsx` labels the setting as a per-camera notification cooldown and states that dashboard event records are managed separately.
4. `backend/experimental_tracking.py` adds a simple offline, class-constrained, greedy one-to-one IoU association verifier between adjacent processed inference frames. It is not connected to live alerts.
5. `evaluation/run_video_benchmark.py` runs the detector once on sampled video frames, replays predictions through both policies, and computes event-level metrics from a supplied ground-truth event CSV.
6. Focused unit tests were added for the verifier and per-camera notification cooldown.
7. The research evaluation protocol and CSV/ROI templates document experiment execution and annotation.

## Not completed / must be measured
- No benchmark was executed on a labelled held-out corpus when this branch was prepared.
- No detector accuracy, mAP, event-quality improvement, inference FPS, end-to-end latency, reconnect success rate or notification delivery success rate is claimed.
- Training configuration and data provenance for the bundled checkpoint have not been recovered.
- The bundled checkpoint reports Fire only.
- Run the full backend test suite and frontend lint/build in the project environment; retain actual logs.

## Review before merge
The experimental verifier uses greedy IoU matching on consecutive processed frames; it does not predict motion or bridge missed frames. The cooldown limiter is process-local and resets on restart. Review these limitations and the tests before merging the branch into `main`.
