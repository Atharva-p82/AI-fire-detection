#!/usr/bin/env python3
"""Convert FURG OpenCV XML flame-box annotations to Fire Guard event intervals.

The FURG XML files contain per-frame rectangles. This utility turns consecutive
annotated frames into event intervals for the project's event-level benchmark.
It does not create new ground truth: it derives temporal intervals from the
published frame annotations. Review the output CSV before using it.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def basename_any_os(value: str) -> str:
    return re.split(r"[\\/]", value.strip().strip('"'))[-1]


def is_annotated(frame_node: ET.Element) -> bool:
    annotations = frame_node.find("annotations")
    if annotations is None:
        return False
    for box in list(annotations):
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", box.text or "")
        if len(values) >= 4:
            try:
                _, _, width, height = map(float, values[:4])
            except ValueError:
                continue
            if width > 0 and height > 0:
                return True
    return False


def annotated_frame_numbers(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    fps_text = root.findtext("frameRate")
    if fps_text is None:
        raise ValueError(f"{xml_path.name}: missing frameRate")
    fps = float(fps_text)
    if fps <= 0:
        raise ValueError(f"{xml_path.name}: frameRate must be positive")

    frames_parent = root.find("frames")
    if frames_parent is None:
        raise ValueError(f"{xml_path.name}: missing frames section")

    annotated = []
    total_frames = 0
    for frame_node in list(frames_parent):
        frame_text = frame_node.findtext("frameNumber")
        if frame_text is None:
            continue
        frame_number = int(frame_text)
        total_frames += 1
        if is_annotated(frame_node):
            annotated.append(frame_number)
    return fps, total_frames, sorted(set(annotated))


def group_runs(frame_numbers, merge_gap_frames):
    if not frame_numbers:
        return []
    runs = []
    start = previous = frame_numbers[0]
    for current in frame_numbers[1:]:
        if current - previous > merge_gap_frames + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Root folder containing FURG .mp4 and .xml files")
    parser.add_argument("--output", type=Path, default=Path("evaluation/furg_events.csv"))
    parser.add_argument("--merge-gap-frames", type=int, default=2,
                        help="Merge annotated segments separated by at most this many unannotated frames")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset folder does not exist: {args.dataset}")
    if args.merge_gap_frames < 0:
        parser.error("--merge-gap-frames must be non-negative")

    xml_files = sorted(args.dataset.rglob("*.xml"))
    if not xml_files:
        parser.error(f"No XML annotation files found under {args.dataset}")

    rows = []
    positives, negatives, missing_videos = 0, 0, []
    seen_videos = set()

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        source_name = root.findtext("fileName") or (xml_path.stem + ".mp4")
        video_name = basename_any_os(source_name)
        if not video_name.lower().endswith(".mp4"):
            video_name = xml_path.stem + ".mp4"

        if video_name in seen_videos:
            raise ValueError(f"Duplicate video annotation filename: {video_name}")
        seen_videos.add(video_name)

        video_candidates = list(args.dataset.rglob(video_name))
        if not video_candidates:
            missing_videos.append(video_name)
            continue

        fps, total_frames, annotated = annotated_frame_numbers(xml_path)
        if not annotated:
            negatives += 1
            continue

        positives += 1
        runs = group_runs(annotated, args.merge_gap_frames)
        for index, (first_frame, last_frame) in enumerate(runs, start=1):
            start_ms = first_frame / fps * 1000.0
            end_ms = (last_frame + 1) / fps * 1000.0
            rows.append({
                "video": video_name,
                "event_id": f"{Path(video_name).stem}_F{index:02d}",
                "class": "fire",
                "start_ms": round(start_ms, 3),
                "end_ms": round(end_ms, 3),
            })

    if missing_videos:
        raise FileNotFoundError(
            "Missing corresponding MP4 files for XML annotations: " +
            ", ".join(missing_videos)
        )
    if positives + negatives == 0:
        raise RuntimeError("No usable XML annotations were processed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video", "event_id", "class", "start_ms", "end_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"XML annotations processed: {positives + negatives}")
    print(f"Videos with flame boxes: {positives}")
    print(f"Videos with no flame boxes: {negatives}")
    print(f"Derived fire-event intervals: {len(rows)}")
    print(f"CSV written to: {args.output.resolve()}")
    print("Review the CSV and confirm event intervals before benchmarking.")


if __name__ == "__main__":
    main()
