import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import notification


class NotificationCooldownTests(unittest.TestCase):
    def setUp(self):
        notification.last_notification.clear()
        self.settings = {"email_enabled": "true", "telegram_enabled": "false", "ai_cooldown_seconds": "5"}
        self.get_setting = patch("notification.get_setting", side_effect=lambda key, default=None: self.settings.get(key, default))
        self.get_setting.start()
        self.email = patch("notification.send_email", return_value=True)
        self.email_mock = self.email.start()
        self.telegram = patch("notification.send_telegram", return_value=True)
        self.telegram.start()

    def tearDown(self):
        self.get_setting.stop()
        self.email.stop()
        self.telegram.stop()
        notification.last_notification.clear()

    def test_cooldown_is_independent_per_camera(self):
        with patch("notification.time.monotonic", side_effect=[100.0, 100.5, 100.5]):
            self.assertTrue(notification.trigger_notifications("fire", 0.9, None, 1))
            self.assertFalse(notification.trigger_notifications("fire", 0.9, None, 1))
            self.assertTrue(notification.trigger_notifications("fire", 0.9, None, 2))
        self.assertEqual(self.email_mock.call_count, 2)

    def test_zero_cooldown_allows_repeated_attempts(self):
        self.settings["ai_cooldown_seconds"] = "0"
        with patch("notification.time.monotonic", side_effect=[10.0, 10.0]):
            self.assertTrue(notification.trigger_notifications("fire", 0.9, None, 1))
            self.assertTrue(notification.trigger_notifications("fire", 0.9, None, 1))
        self.assertEqual(self.email_mock.call_count, 2)

    def test_no_enabled_channels_does_not_reserve_cooldown(self):
        self.settings["email_enabled"] = "false"
        with patch("notification.time.monotonic", return_value=5.0):
            self.assertFalse(notification.trigger_notifications("fire", 0.9, None, 1))
        self.assertNotIn("camera:1", notification.last_notification)

    def test_parallel_workers_reserve_only_one_attempt(self):
        entered, release, calls = threading.Event(), threading.Event(), []
        def slow_send(*args, **kwargs):
            calls.append(1)
            entered.set()
            release.wait(timeout=2)
            return True
        with patch("notification.send_email", side_effect=slow_send), patch("notification.time.monotonic", return_value=50.0):
            first = threading.Thread(target=notification.trigger_notifications, args=("fire", 0.9, None, 7))
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second_result = notification.trigger_notifications("fire", 0.9, None, 7)
            release.set()
            first.join(timeout=2)
        self.assertFalse(second_result)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
