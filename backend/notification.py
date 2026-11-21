import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests
import os
import threading
import time
from database import get_setting

# Camera-scoped notification rate limiter. Monotonic time avoids wall-clock jumps.
last_notification = {}
_notification_lock = threading.Lock()
DEFAULT_NOTIFICATION_COOLDOWN_SECONDS = 5.0

def send_email(subject, body, snapshot_path=None):
    """Send an email using SMTP settings from database."""
    try:
        smtp_server = get_setting("smtp_server")
        smtp_port = get_setting("smtp_port")
        smtp_user = get_setting("smtp_user")
        smtp_password = get_setting("smtp_password")
        receiver_email = get_setting("receiver_email")
        
        if not all([smtp_server, smtp_port, smtp_user, smtp_password, receiver_email]):
            print("Email notification skipped: Missing SMTP settings.")
            return False

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = receiver_email
        
        msg.attach(MIMEText(body, "plain"))
        
        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as f:
                img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(snapshot_path))
                msg.attach(image)
        
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            
        print(f"Email sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def send_telegram(message, snapshot_path=None):
    """Send a Telegram message using Bot API settings from database."""
    try:
        bot_token = get_setting("telegram_bot_token")
        chat_id = get_setting("telegram_chat_id")
        
        if not all([bot_token, chat_id]):
            print("Telegram notification skipped: Missing Bot settings.")
            return False

        # Send Photo if available
        if snapshot_path and os.path.exists(snapshot_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(snapshot_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": message}
                response = requests.post(url, data=data, files=files)
        else:
            # Send Text only
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            response = requests.post(url, data=data)
            
        if response.status_code == 200:
            print(f"Telegram message sent to {chat_id}")
            return True
        else:
            print(f"Telegram failed: {response.text}")
            return False

    except Exception as e:
        print(f"Failed to send telegram: {e}")
        return False

def trigger_notifications(event_type, confidence, snapshot_path=None, camera_id=None):
    """Send enabled alerts subject to a configurable per-camera cooldown.

    A camera's cooldown does not suppress another camera's alert. Test
    notifications without a camera ID use a separate 'test' scope.
    """
    email_enabled = get_setting("email_enabled") == "true"
    telegram_enabled = get_setting("telegram_enabled") == "true"
    if not email_enabled and not telegram_enabled:
        print("Notifications skipped: no notification channel is enabled.")
        return False

    raw_cooldown = get_setting("ai_cooldown_seconds", str(DEFAULT_NOTIFICATION_COOLDOWN_SECONDS))
    try:
        cooldown_seconds = float(raw_cooldown)
        if not (cooldown_seconds >= 0 and cooldown_seconds < float("inf")):
            raise ValueError("Cooldown must be finite and non-negative")
    except (TypeError, ValueError, OverflowError):
        cooldown_seconds = DEFAULT_NOTIFICATION_COOLDOWN_SECONDS

    scope = f"camera:{camera_id}" if camera_id is not None else "test"
    current_time = time.monotonic()
    with _notification_lock:
        previous_time = last_notification.get(scope)
        if previous_time is not None and current_time - previous_time < cooldown_seconds:
            print(f"Notification skipped due to cooldown for {scope}.")
            return False
        # Reserve the window before network I/O so parallel worker threads do
        # not send duplicate alerts for the same camera at the same time.
        last_notification[scope] = current_time

    confidence_percent = f"{confidence * 100:.1f}%"
    subject = f"EMERGENCY: {event_type.upper()} Detected!"
    body = (
        f"WARNING!\n\nThe system detected {event_type} with {confidence_percent} confidence."
        f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nPlease check the dashboard immediately."
    )

    results = []
    if email_enabled:
        results.append(send_email(subject, body, snapshot_path))
    if telegram_enabled:
        results.append(send_telegram(body, snapshot_path))
    # Cooldown is an attempt-rate limit, regardless of provider success.
    return any(results)
