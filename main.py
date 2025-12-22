import cv2
import sys
import os
import argparse
import shutil
import threading
import time

from detector import Detector
from event_manager import EventManager
from llm_verifier import LlmVerifier
from face_manager import FaceManager
from notification_manager import NotificationManager
from dotenv import load_dotenv
import queue

load_dotenv()

class StreamWorker:
    """
    Parallel Worker: Handles one camera stream, performs detection, 
    and processes event logic in its own thread.
    """
    def __init__(self, url, name, detector, face_manager, notifier):
        self.url = url
        self.name = name
        self.detector = detector
        self.face_manager = face_manager
        self.notifier = notifier
        self.event_manager = EventManager(face_manager=face_manager)
        
        self.running = True
        self.cap = cv2.VideoCapture(url)
        self.latest_frame = None
        self.annotated_frame = None
        self.status_text = "Checking..."
        self.lock = threading.Lock()
        self.error_count = 0
        
        # Start Heartbeat
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log("Worker initialized and thread started.")

    def log(self, msg):
        print(f"[{self.name}] {msg}")

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.error_count += 1
                if self.error_count % 100 == 0:
                    self.log("Stream lost. Attempting auto-reconnect...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url)
                time.sleep(0.1)
                continue

            self.error_count = 0
            
            # --- PARALLEL BRAIN START ---
            try:
                # 1. Detect
                results = self.detector.detect(frame)
                
                # 2. Process logic
                status, pkgs, theft_event = self.event_manager.update(results, frame)
                
                # 3. Handle Alert (Async)
                if theft_event:
                    self.log(f"!!! {theft_event['msg']} !!!")
                    self.notifier.send_alert("POTENTIAL THEFT", f"[{self.name}] {theft_event['msg']}")
                    # Logic for LLM check could be added here or in main
                
                # 4. Prepare visualization frame
                res_list = results if isinstance(results, list) else [results]
                ann_frame = res_list[0].plot()
                cv2.putText(ann_frame, f"{self.name}: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Update shared state
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.annotated_frame = ann_frame
                    self.status_text = status

            except Exception as e:
                self.log(f"Processing Error: {e}")
                time.sleep(1)

    def get_display_frame(self):
        with self.lock:
            if self.annotated_frame is None:
                return None
            return self.annotated_frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()
        self.log("Worker stopped.")

def perform_cleanup():
    print("\n[RESET] Performing Factory Reset...")
    for folder in ["faces"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
    for f in ["alerts.log", "evidence.jpg"]:
        if os.path.exists(f): os.remove(f)
    for model in ["yolov8n.pt", "rtdetr-l.pt"]:
        if os.path.exists(model): os.remove(model)
    print("[RESET] Complete.\n")

import time

def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Package Theft Detection")
    parser.add_argument("--streams", nargs="+", help="RTSP URLs for cameras")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--active-minutes", type=float, help="Cycle Mode: Minutes ON")
    parser.add_argument("--sleep-minutes", type=float, help="Cycle Mode: Minutes OFF")
    parser.add_argument("--reset", action="store_true", help="Factory Reset")
    parser.add_argument("--no-llm", action="store_true", help="Disable OpenAI")
    parser.add_argument("--detector", choices=['yolo', 'rtdetr', 'ensemble'], default='ensemble')
    args = parser.parse_args()

    if args.reset: perform_cleanup()

    notifier = NotificationManager()
    detector = Detector(model_type=args.detector)
    face_manager = FaceManager()
    llm_verifier = LlmVerifier() if not args.no_llm else None

    # Priority: 1. Command Line args, 2. Env variable, 3. Defaults
    if args.streams:
        urls = args.streams
    else:
        env_streams = os.getenv("CAMERA_STREAMS")
        if env_streams:
            urls = [s.strip() for s in env_streams.split(",") if s.strip()]
        else:
            # Fallback default
            urls = ["rtsp://192.168.86.35:57800/44b620aab5dfcd3d"]

    def run_detection_session(duration_minutes=None):
        inf_mode = "Infinite" if duration_minutes is None else f"{duration_minutes} min"
        print(f"\n--- Parallel Detection Active ({len(urls)} cameras, {inf_mode}) ---")
        
        # Initialize Workers (Threads)
        workers = [StreamWorker(url, f"CAM_{i+1}", detector, face_manager, notifier) for i, url in enumerate(urls)]
        
        start_time = time.time()

        try:
            while True:
                if duration_minutes and (time.time() - start_time)/60 >= duration_minutes:
                    print("Session time limit reached.")
                    break

                # The Main Thread now only handles the UI display and keyboard input
                # All detection and processing is happening in background threads!
                
                if not args.headless:
                    frames = []
                    for w in workers:
                        f = w.get_display_frame()
                        if f is not None:
                            frames.append(f)
                    
                    if frames:
                        # Stack images horizontally for the grid
                        display_grid = cv2.hconcat(frames) if len(frames) > 1 else frames[0]
                        
                        # Scale down for desktop if too large
                        h, w = display_grid.shape[:2]
                        target_w = 1280
                        if w > target_w:
                            scale = target_w / w
                            display_grid = cv2.resize(display_grid, (int(w*scale), int(h*scale)))

                        cv2.imshow('ANTIGRAVITY: Multi-Cam Parallel Guard', display_grid)
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        return True, False
                    elif key == ord('r'):
                        # Register owner using the first available camera frame
                        for w in workers:
                            with w.lock:
                                if w.latest_frame is not None:
                                    success, msg = face_manager.register_face(w.latest_frame, "Owner")
                                    print(f"[{w.name}] API: {msg}")
                                    if success: face_manager.load_known_faces()
                                    break
                else:
                    # Headless: Just wait and let threads work
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\nShutting down workers...")
            return True, False
        finally:
            for w in workers: w.stop()
            cv2.destroyAllWindows()
            
        return False, False

    # Runtime Execution
    if args.active_minutes and args.sleep_minutes:
        while True:
            should_quit, err = run_detection_session(args.active_minutes)
            if should_quit: break
            time.sleep(args.sleep_minutes * 60)
    else:
        run_detection_session(None)

if __name__ == "__main__":
    main()

