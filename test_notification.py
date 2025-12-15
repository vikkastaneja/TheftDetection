from notification_manager import NotificationManager
from dotenv import load_dotenv
import os

# Load env manually for test script
load_dotenv()

print("Testing Notification Manager...")
print(f"Topic: {os.getenv('NTFY_TOPIC')}")

notifier = NotificationManager()
notifier.send_alert("TEST EVENT", "This is a test notification from the Theft Detection System.")
