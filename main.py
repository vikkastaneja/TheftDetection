import cv2
import sys

from detector import Detector
from event_manager import EventManager
from llm_verifier import LlmVerifier
from face_manager import FaceManager
from notification_manager import NotificationManager
import threading
import cv2
import sys
import os

import argparse
import shutil

def perform_cleanup():
    print("\n[RESET] Performing Factory Reset...")
    
    # 1. Clean Faces (Learned Data)
    if os.path.exists("faces"):
        shutil.rmtree("faces")
        os.makedirs("faces")
        print(" - Deleted all learned faces")
    
    # 2. Clean Logs
    if os.path.exists("alerts.log"):
        os.remove("alerts.log")
        print(" - Deleted alerts.log")

    # 3. Clean Evidence
    if os.path.exists("evidence.jpg"):
        os.remove("evidence.jpg")
        print(" - Deleted evidence.jpg")

    # 4. Remove Model (Force Re-download)
    if os.path.exists("yolov8n.pt"):
        os.remove("yolov8n.pt")
        print(" - Deleted yolov8n.pt (System will re-download the latest official model)")

    print("[RESET] Complete. System is now fresh.\n")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Package Theft Detection System")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI/Window)")
    parser.add_argument("--reset", action="store_true", help="Factory reset: Delete all learned faces, logs, and models before starting")
    args = parser.parse_args()

    if args.reset:
        perform_cleanup()

    # Initialize Notifier first to report startup errors
    notifier = NotificationManager()

    # Initialize video capture with IP Camera
    rtsp_url = "rtsp://admin:Test1234%21@192.168.86.42:554/11"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        msg = f"Error: Could not open video capture from {rtsp_url}"
        print(msg)
        notifier.send_alert("SYSTEM ERROR", msg)
        sys.exit(1)

    # Initialize components
    detector = Detector()
    face_manager = FaceManager()
    event_manager = EventManager(face_manager=face_manager)
    llm_verifier = LlmVerifier() # Will look for OPENAI_API_KEY env var
    # notifier already initialized above

    print("Starting video feed with detection.")
    if args.headless:
        print("Running in HEADLESS mode. Press Ctrl+C to quit.")
    else:
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Register current face as Owner")
    
    llm_status = "LLM Status: Idle"

    def run_llm_check(frames, event_msg):
        nonlocal llm_status
        llm_status = "LLM Status: Analyzing..."
        
        # 1. Ask LLM
        result = llm_verifier.verify_event(frames)
        llm_status = f"LLM Result: {result}"
        
        # 2. Notify
        print(f"\n[AI REPORT] {result}\n")
        
        # Heuristic check on LLM response to see if confirmed
        is_confirmed = "YES" in result.upper()
        alert_type = "THEFT CONFIRMED" if is_confirmed else "Suspicious Event"
        
        notifier.send_alert(alert_type, f"AI Verdict: {result}", proof_path="See logs")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                msg = "Error: Can't receive frame (stream end?). Camera might be down."
                print(msg)
                notifier.send_alert("CAMERA DOWN", msg)
                break

            # Run object detection
            results = detector.detect(frame)
            
            # Update event logic
            status_text, tracked_packages, theft_event = event_manager.update(results, frame)
            
            # Check for Theft Event -> Trigger LLM
            if theft_event:
                msg = theft_event['msg']
                print(f"THEFT DETECTED: {msg}")
                
                # Save Evidence Frame
                evidence_path = None
                if theft_event['frames']:
                    evidence_path = os.path.abspath("evidence.jpg")
                    
                    # SMART EVENT: Find the frame with the best face
                    best_frame = face_manager.find_best_face_frame(theft_event['frames'])
                    
                    cv2.imwrite(evidence_path, best_frame)
                
                # Initial Alert
                notifier.send_alert("POTENTIAL THEFT", msg, proof_path=evidence_path)
                
                # Run LLM in a separate thread
                # We pass the evidence path to the thread so it can reuse it if needed
                threading.Thread(target=run_llm_check, args=(theft_event['frames'], msg)).start()
            
            # HEADLESS CHECK
            if not args.headless:
                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Draw status text
                cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(annotated_frame, llm_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Visualize Face ID (Optional: Draw name near person)
                # We can extract person boxes from YOLO results to optimize
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0: # Person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        name, dist = face_manager.identify_person(frame, [x1, y1, x2, y2])
                        label = f"{name} ({dist:.2f})"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw tracked packages 
                for pkg_id, pkg in tracked_packages.items():
                    box = pkg['box']
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    cv2.putText(annotated_frame, f"ID: {pkg_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Display the resulting frame
                cv2.imshow('Theft Detection Feed', annotated_frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Register the largest face in the frame as "Owner"
                    # In a real app, you'd ask for a name. Here we default to "Owner"
                    print("Attempting to register face...")
                    success, msg = face_manager.register_face(frame, "Owner")
                    print(msg)
                    # Re-load to update memory
                    if success:
                        face_manager.load_known_faces()
    except KeyboardInterrupt:
        print("\nStopping...")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
