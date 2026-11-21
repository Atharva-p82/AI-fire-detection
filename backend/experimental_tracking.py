"""Experimental IoU-linked temporal verifier for Fire Guard AI.
This module is deliberately separate from the production camera pipeline.
"""
from detection_policy import _finite_number, point_in_roi


def box_iou(first, second):
    """Calculate IoU for two [x1, y1, x2, y2] boxes."""
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return 0.0
    if len(first) != 4 or len(second) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (_finite_number(v) for v in first)
        bx1, by1, bx2, by2 = (_finite_number(v) for v in second)
    except ValueError:
        return 0.0
    if ax1 > ax2 or ay1 > ay2 or bx1 > bx2 or by1 > by2:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def advance_track_persistence(state, detections, settings, roi=(), frame_shape=None,
                              track_iou_threshold=0.30):
    """Confirm detections only when same-class boxes match across consecutive frames.

    Greedy one-to-one IoU matching is used. This is a simple experiment baseline,
    not a general multi-object tracker: no motion prediction, appearance embeddings,
    optical flow or missed-frame recovery are implemented.
    """
    threshold = _finite_number(track_iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("track_iou_threshold must be between 0 and 1")
    if not isinstance(state, dict):
        raise ValueError("state must be a dictionary")
    state.setdefault("next_track_id", 1)
    state.setdefault("tracks", {})

    eligible = []
    for detection in detections:
        name = str(detection.get("class", "")).strip().lower()
        try:
            confidence = _finite_number(detection.get("confidence"))
        except ValueError:
            continue
        if not name or not 0 <= confidence <= 1 or confidence < settings.threshold_for(name):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            left, top, right, bottom = (_finite_number(v) for v in bbox)
        except ValueError:
            continue
        if left > right or top > bottom:
            continue
        if roi:
            if frame_shape is None or len(frame_shape) < 2 or not frame_shape[0] or not frame_shape[1]:
                continue
            center = ((left + right) / (2 * frame_shape[1]), (top + bottom) / (2 * frame_shape[0]))
            if not point_in_roi(center, roi):
                continue
        eligible.append({**detection, "class": name, "confidence": confidence,
                         "bbox": [left, top, right, bottom]})

    previous = state["tracks"]
    candidates = []
    for track_id, track in previous.items():
        for det_index, detection in enumerate(eligible):
            if track["class"] == detection["class"]:
                overlap = box_iou(track["bbox"], detection["bbox"])
                if overlap >= threshold:
                    candidates.append((overlap, track_id, det_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_tracks, matched_detections = set(), set()
    new_tracks, confirmed = {}, []
    for overlap, track_id, det_index in candidates:
        if track_id in matched_tracks or det_index in matched_detections:
            continue
        matched_tracks.add(track_id)
        matched_detections.add(det_index)
        detection = eligible[det_index]
        hits = previous[track_id]["hits"] + 1
        new_tracks[track_id] = {"class": detection["class"], "bbox": detection["bbox"], "hits": hits}
        if hits >= settings.persistence_frames:
            confirmed.append({**detection, "track_id": track_id, "track_hits": hits})

    for det_index, detection in enumerate(eligible):
        if det_index in matched_detections:
            continue
        track_id = state["next_track_id"]
        state["next_track_id"] += 1
        new_tracks[track_id] = {"class": detection["class"], "bbox": detection["bbox"], "hits": 1}
        if settings.persistence_frames <= 1:
            confirmed.append({**detection, "track_id": track_id, "track_hits": 1})

    # Dropping unmatched tracks enforces consecutive-frame evidence.
    state["tracks"] = new_tracks
    return confirmed
