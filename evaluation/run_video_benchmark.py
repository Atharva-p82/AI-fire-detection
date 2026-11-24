#!/usr/bin/env python3
"""Offline event-level Fire Guard AI benchmark.

Runs the model once on each sampled frame, then replays the same detections through
class persistence and an experimental IoU-linked verifier. This script reports
event-level metrics only; it does not compute object-detection mAP.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
from detection import FireDetector
from detection_policy import DetectionSettings, advance_persistence, validate_roi
from experimental_tracking import advance_track_persistence


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_videos(path):
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
    if path.is_file():
        if path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported video extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def read_events(path):
    if path is None:
        return None
    required = {"video", "event_id", "class", "start_ms", "end_ms"}
    output = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Ground-truth CSV must include {sorted(required)}")
        for row in reader:
            start, end = float(row["start_ms"]), float(row["end_ms"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError(f"Invalid time interval: {row}")
            output.append({"video": Path(row["video"]).name, "event_id": row["event_id"],
                           "class": row["class"].strip().lower(), "start_ms": start, "end_ms": end})
    return output


def read_rois(path):
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("ROI JSON must map video filenames to polygon points")
    return {str(name): validate_roi(points) for name, points in values.items()}


def infer_video(path, detector, stride, confidence, iou):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    records, frame_index, reads, last_time = [], -1, 0, -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            reads += 1
            if reads % stride:
                continue
            time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if not math.isfinite(time_ms) or time_ms < 0 or time_ms <= last_time:
                time_ms = frame_index / fps * 1000.0
            last_time = time_ms
            _, detections = detector.process_frame(frame, confidence=confidence, iou=iou)
            clean = []
            for d in detections:
                box = d.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                clean.append({"class": str(d.get("class", "")).strip().lower(),
                              "confidence": float(d.get("confidence", 0)),
                              "bbox": [float(v) for v in box]})
            records.append({"frame_index": frame_index, "time_ms": time_ms,
                            "shape": tuple(frame.shape), "detections": clean})
    finally:
        cap.release()
    duration = count / fps * 1000 if count > 0 else (frame_index + 1) / fps * 1000
    return records, fps, reads, max(duration, 0.0)


def replay(records, policy, persistence, roi, confidence, iou, track_iou):
    settings = DetectionSettings(confidence=confidence, iou=iou, persistence_frames=persistence)
    state, output = {}, []
    for rec in records:
        if policy == "class":
            confirmed = list(advance_persistence(state, rec["detections"], settings,
                                roi=roi, frame_shape=rec["shape"]).values())
        else:
            confirmed = advance_track_persistence(state, rec["detections"], settings,
                                roi=roi, frame_shape=rec["shape"], track_iou_threshold=track_iou)
        output.append({**rec, "confirmed": confirmed})
    return output


def group_events(video, records, rearm_ms):
    active, completed = {}, []
    def close(cls):
        item = active.pop(cls)
        item["end_ms"] = item["last_seen_ms"]
        completed.append(item)
    for rec in records:
        now = rec["time_ms"]
        classes = {}
        for d in rec["confirmed"]:
            cls = d["class"]
            if cls not in classes or d["confidence"] > classes[cls]["confidence"]:
                classes[cls] = d
        for cls, d in classes.items():
            item = active.get(cls)
            if item is not None and now - item["last_seen_ms"] >= rearm_ms:
                close(cls)
                item = None
            if item is None:
                active[cls] = {"video": video, "class": cls, "start_ms": now,
                               "last_seen_ms": now, "confidence": d["confidence"]}
            else:
                item["last_seen_ms"] = now
                item["confidence"] = max(item["confidence"], d["confidence"])
        for cls, item in list(active.items()):
            if cls not in classes and now - item["last_seen_ms"] >= rearm_ms:
                close(cls)
    for cls in list(active):
        close(cls)
    return completed


def percentile(values, q):
    if not values: return None
    ordered = sorted(values)
    if len(ordered) == 1: return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi-pos) + ordered[hi] * (pos-lo)


def score_events(predicted, truth, duration_by_video, tolerance_ms):
    by_video = defaultdict(list)
    for gt in truth: by_video[gt["video"]].append(gt)
    remaining = {id(gt) for gt in truth}
    tp, fp, delays, matches = 0, 0, [], []
    for pred in sorted(predicted, key=lambda x: (x["video"], x["start_ms"])):
        choices = [gt for gt in by_video.get(pred["video"], [])
                   if id(gt) in remaining and gt["class"] == pred["class"]
                   and gt["start_ms"] <= pred["start_ms"] <= gt["end_ms"] + tolerance_ms]
        if not choices:
            fp += 1
            matches.append({**pred, "match": "FP", "ground_truth_event_id": "", "delay_ms": ""})
            continue
        gt = min(choices, key=lambda x: abs(pred["start_ms"] - x["start_ms"]))
        remaining.remove(id(gt))
        tp += 1
        delay = pred["start_ms"] - gt["start_ms"]
        delays.append(delay)
        matches.append({**pred, "match": "TP", "ground_truth_event_id": gt["event_id"], "delay_ms": delay})
    fn = len(remaining)
    hours = sum(duration_by_video.values()) / 3_600_000
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"event_tp": tp, "event_fp": fp, "event_fn": fn,
            "event_precision": precision, "event_recall": recall, "event_f1": f1,
            "false_alerts_per_video_hour": fp/hours if hours else None,
            "matched_detection_delay_median_ms": statistics.median(delays) if delays else None,
            "matched_detection_delay_p95_ms": percentile(delays, 95),
            "monitored_camera_hours": hours, "matches": matches}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--events", type=Path, help="CSV columns: video,event_id,class,start_ms,end_ms")
    ap.add_argument("--output", type=Path, default=Path("evaluation/results"))
    ap.add_argument("--roi-json", type=Path)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--persistence", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--track-iou", type=float, default=0.30)
    ap.add_argument("--rearm-seconds", type=float, default=10.0)
    ap.add_argument("--event-delay-window-seconds", type=float, default=30.0)
    args = ap.parse_args()
    if args.stride < 1 or any(p < 1 for p in args.persistence): ap.error("stride and persistence must be positive")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1 or not 0 <= args.track_iou <= 1:
        ap.error("confidence, IoU and track-IoU must be in [0,1]")
    if args.rearm_seconds < 0 or args.event_delay_window_seconds < 0: ap.error("time windows cannot be negative")
    videos = find_videos(args.videos)
    if not videos: ap.error("No supported videos found")
    if len({v.name for v in videos}) != len(videos): ap.error("Video filenames must be unique")
    truth, rois = read_events(args.events), read_rois(args.roi_json)
    # Normalize annotation video names against the actual files. The FURG XML
    # metadata uses Car2.mp4 / Car3.mp4 while the repository stores car2.mp4 /
    # car3.mp4; Linux paths are case-sensitive.
    video_name_lookup = {video.name.casefold(): video.name for video in videos}
    if truth is not None:
        unknown_videos = sorted({
            event["video"] for event in truth
            if event["video"].casefold() not in video_name_lookup
        })
        if unknown_videos:
            raise ValueError(f"Ground-truth CSV contains videos not present in --videos input: {unknown_videos}")
        for event in truth:
            event["video"] = video_name_lookup[event["video"].casefold()]
    model_path = Path(os.environ.get("FIRE_MODEL_PATH", BACKEND / "best.pt")).expanduser().resolve()
    if not model_path.is_file(): raise FileNotFoundError(f"Checkpoint missing: {model_path}")
    detector = FireDetector(str(model_path))
    if truth is not None:
        loaded_classes = {str(name).strip().lower() for name in detector.model_info()["classes"]}
        unknown_classes = sorted({event["class"] for event in truth} - loaded_classes)
        if unknown_classes:
            raise ValueError(f"Ground-truth classes {unknown_classes} are not in loaded checkpoint classes {sorted(loaded_classes)}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_path), "model_sha256": sha256(model_path),
                "model_name": detector.model_name, "model_classes": detector.model_info()["classes"],
                "smoke_supported": detector.model_info()["smoke_supported"],
                "python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__,
                "cpu": platform.processor(), "stride": args.stride, "confidence": args.confidence,
                "nms_iou": args.iou, "persistence": args.persistence, "track_iou": args.track_iou,
                "rearm_seconds": args.rearm_seconds, "event_tolerance_seconds": args.event_delay_window_seconds}
    raw_rows, raw_by_video, durations = [], {}, {}
    for video in videos:
        records, fps, reads, duration = infer_video(video, detector, args.stride, args.confidence, args.iou)
        raw_by_video[video.name], durations[video.name] = records, duration
        metadata.setdefault("videos", []).append({"video": video.name, "fps": fps,
                    "decoded_frames": reads, "inference_frames": len(records), "duration_ms": duration})
        for rec in records:
            for d in rec["detections"]:
                x1,y1,x2,y2 = d["bbox"]
                raw_rows.append({"video": video.name, "frame_index": rec["frame_index"],
                    "time_ms": rec["time_ms"], "class": d["class"], "confidence": d["confidence"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    write_csv(args.output/"raw_detections.csv", raw_rows,
              ["video","frame_index","time_ms","class","confidence","x1","y1","x2","y2"])
    summary, all_events, all_matches = [], [], []
    roi_modes = [False, True] if rois else [False]
    for roi_enabled in roi_modes:
        for policy in ("class", "track_iou"):
            for persistence in args.persistence:
                predicted = []
                for name, records in raw_by_video.items():
                    roi = rois.get(name, []) if roi_enabled else []
                    evaluated = replay(records, policy, persistence, roi,
                                        args.confidence, args.iou, args.track_iou)
                    events = group_events(name, evaluated, args.rearm_seconds*1000)
                    for event in events:
                        event.update({"policy": policy, "persistence_frames": persistence,
                                      "roi_enabled": roi_enabled, "roi_vertices": len(roi)})
                    predicted.extend(events)
                row = {"policy": policy, "persistence_frames": persistence,
                       "roi_enabled": roi_enabled, "predicted_event_count": len(predicted)}
                if truth is not None:
                    metrics = score_events(predicted, truth, durations, args.event_delay_window_seconds*1000)
                    all_matches.extend({**m, "policy": policy, "persistence_frames": persistence,
                                        "roi_enabled": roi_enabled} for m in metrics.pop("matches"))
                    row.update(metrics)
                summary.append(row)
                all_events.extend(predicted)
    write_csv(args.output/"predicted_events.csv", all_events,
              ["video","class","start_ms","last_seen_ms","end_ms","confidence","policy",
               "persistence_frames","roi_enabled","roi_vertices"])
    write_csv(args.output/"summary.csv", summary, list(summary[0]))
    if truth is not None:
        write_csv(args.output/"event_matches.csv", all_matches,
                  ["video","class","start_ms","last_seen_ms","end_ms","confidence","match",
                   "ground_truth_event_id","delay_ms","policy","persistence_frames","roi_enabled"])
    (args.output/"run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps({
        "metadata": metadata, "results": summary,
        "interpretation": "Event metrics from supplied labels." if truth is not None else
        "No ground-truth CSV supplied: predictions only, not measured accuracy."}, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(videos), "raw_detections": len(raw_rows),
                      "configurations": len(summary), "output": str(args.output.resolve()),
                      "ground_truth_supplied": truth is not None}, indent=2))


if __name__ == "__main__":
    main()
