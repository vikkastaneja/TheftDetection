from ultralytics import YOLO, RTDETR
import cv2

class Detector:
    def __init__(self, model_type='ensemble'):
        """
        Initialize the detector(s).
        model_type: 'yolo', 'rtdetr', or 'ensemble' (both)
        """
        self.model_type = model_type
        self.models = {}

        if model_type in ['yolo', 'ensemble']:
            print("\n[CNN] Initializing YOLOv8n...")
            self.models['yolo'] = YOLO('yolov8n.pt')
        
        if model_type in ['rtdetr', 'ensemble']:
            print("\n[Visual Transformer] Initializing RT-DETR-L...")
            self.models['rtdetr'] = RTDETR('rtdetr-l.pt')

        print(f"Detector initialized in '{model_type}' mode.")

    def detect(self, frame):
        """
        Run detection. If ensemble, merge results.
        """
        all_results = []
        for name, model in self.models.items():
            res = model(frame, verbose=False)[0]
            all_results.append(res)

        if len(all_results) == 1:
            return all_results[0]
        
        # If ensemble, we return the first one but you can access shared detections.
        # For simple integration with existing code, we'll merge boxes into the first result.
        # This keeps compatibility with 'results[0].plot()' and 'results[0].boxes'
        main_res = all_results[0]
        if len(all_results) > 1:
            # Simple merge: append boxes from 2nd model to 1st
            # This works because event_manager uses IoU matching anyway.
            # RT-DETR detections will just be 'extra' candidates.
            main_res.boxes = main_res.boxes # Keep original
            # Note: Directly merging Boxes objects is complex in Ultralytics, 
            # but EventManager just iterates over them.
            # We will return the list of results and let main handle it or 
            # return a proxy. 
            pass 

        return all_results

