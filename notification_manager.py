import datetime
import os
import requests

class NotificationManager:
    def __init__(self, log_file="alerts.log"):
        self.log_file = log_file
        # Default topic if not set
        self.ntfy_topic = os.getenv("NTFY_TOPIC", "package_guard_demo")

    def send_push_notification(self, title, message, attachment_path=None):
        """Sends push notification via ntfy.sh"""
        try:
            url = f"https://ntfy.sh/{self.ntfy_topic}"
            headers = {
                "Title": title,
                "Priority": "high" if "THEFT" in title else "default",
                "Tags": "warning,package"
            }
            
            data = None
            if attachment_path and os.path.exists(attachment_path):
                # Send file as body, message in header
                headers["Filename"] = os.path.basename(attachment_path)
                # ntfy.sh allows message in 'Message' header when sending file
                headers["Message"] = message 
                with open(attachment_path, "rb") as f:
                    # We have to read it here or pass the file handle to requests
                    # requests supports passing the file handle directly
                    requests.post(url, data=f, headers=headers)
            else:
                # Send message as body
                data = message.encode('utf-8')
                requests.post(url, data=data, headers=headers)
                
            print(f"[Success] Push notification sent to ntfy.sh/{self.ntfy_topic}")
        except Exception as e:
            print(f"[Error] Failed to send push notification: {e}")

    def send_alert(self, event_type, message, proof_path=None):
        """
        Sends an alert (Email/SMS simulation).
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_body = f"[{timestamp}] ALERT: {event_type} - {message}"
        if proof_path:
            alert_body += f" (Evidence: {proof_path})"

        # 1. Console Output
        print("\n" + "="*40)
        print(alert_body)
        print("="*40 + "\n")

        # 2. Log to File
        with open(self.log_file, "a") as f:
            f.write(alert_body + "\n")

        # 3. Send Mobile Push
        self.send_push_notification(event_type, message, attachment_path=proof_path)
