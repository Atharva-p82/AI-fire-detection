import sys
from pathlib import Path
import unittest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from detection_policy import DetectionSettings
from experimental_tracking import advance_track_persistence, box_iou


def det(box, confidence=0.9, cls="fire"):
    return {"class": cls, "confidence": confidence, "bbox": box}


class TrackVerifierTests(unittest.TestCase):
    def test_iou_exact_and_disjoint(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_same_track_reaches_confirmation(self):
        state, settings = {}, DetectionSettings(persistence_frames=2)
        self.assertEqual(advance_track_persistence(state, [det([10, 10, 40, 40])], settings), [])
        confirmed = advance_track_persistence(state, [det([11, 10, 41, 40])], settings)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["track_hits"], 2)

    def test_different_object_does_not_inherit_old_track_hits(self):
        state, settings = {}, DetectionSettings(persistence_frames=2)
        advance_track_persistence(state, [det([0, 0, 10, 10])], settings)
        confirmed = advance_track_persistence(state, [det([50, 50, 60, 60])], settings)
        self.assertEqual(confirmed, [])
        self.assertEqual([track["hits"] for track in state["tracks"].values()], [1])

    def test_multiple_objects_are_associated_one_to_one(self):
        state, settings = {}, DetectionSettings(persistence_frames=2)
        advance_track_persistence(state, [det([0, 0, 10, 10]), det([30, 30, 40, 40])], settings)
        confirmed = advance_track_persistence(state, [det([1, 0, 11, 10]), det([31, 30, 41, 40])], settings)
        self.assertEqual(len(confirmed), 2)
        self.assertEqual(len({item["track_id"] for item in confirmed}), 2)

    def test_missing_detection_resets_consecutive_track(self):
        state, settings = {}, DetectionSettings(persistence_frames=2)
        advance_track_persistence(state, [det([0, 0, 10, 10])], settings)
        advance_track_persistence(state, [], settings)
        self.assertEqual(advance_track_persistence(state, [det([0, 0, 10, 10])], settings), [])

    def test_threshold_and_roi_are_applied_before_association(self):
        state = {}
        settings = DetectionSettings(confidence=0.5, persistence_frames=2)
        roi = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
        shape = (100, 100, 3)
        advance_track_persistence(state, [det([70, 20, 90, 40])], settings, roi, shape)
        self.assertEqual(state["tracks"], {})
        self.assertEqual(advance_track_persistence(state, [det([10, 10, 20, 20], 0.2)], settings), [])

    def test_invalid_track_iou_is_rejected(self):
        with self.assertRaises(ValueError):
            advance_track_persistence({}, [], DetectionSettings(), track_iou_threshold=1.5)


if __name__ == "__main__":
    unittest.main()
