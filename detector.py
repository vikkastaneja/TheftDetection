from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 model.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect(self, frame):
        """
        Run object detection on a single frame.
        Returns the frame with annotations and the list of results.
        """
        results = self.model(frame)
        return results
