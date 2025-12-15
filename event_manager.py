from collections import deque
import cv2

import time 
import math

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

class EventManager:
    def __init__(self, face_manager=None, buffer_size=30):
        # Configuration
        self.package_classes = [24, 26, 28]  # backpack(24), handbag(26), suitcase(28)
        self.person_class = 0  # person(0)
        self.iou_threshold = 0.3 # Threshold to consider it the same object
        self.disappear_threshold = 1.5 # Seconds to wait before declaring lost (stolen?)
        
        # Dependencies
        self.face_manager = face_manager

        # State
        # tracked_packages: dict {id: {'box': [x,y,x,y], 'last_seen': time, 'created_at': time, 'status': 'monitor'}}
        self.tracked_packages = {} 
        self.next_package_id = 1
        self.events = [] # Log of events like "Package 1 Delivered", "Package 1 Removed"
        
        # Frame Buffer (store last N frames for context)
        self.frame_buffer = deque(maxlen=buffer_size) 

    def update(self, results, frame):
        """
        Process detection results to update state.
        results: List of YOLO results
        frame: The current video frame (numpy array)
        """
        current_time = time.time()
        
        # Add current frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # 1. Parse current detections
        current_persons = []
        detected_packages = [] # List of {'box': box}

        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            cls = int(box.cls[0])
            # box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            if cls == self.person_class:
                current_persons.append([x1, y1, x2, y2])
            elif cls in self.package_classes:
                detected_packages.append({'box': [x1, y1, x2, y2]})

        # 2. Match Detected Packages to Tracked Packages (Greedy Match)
        matched_track_ids = set()
        
        for det_pkg in detected_packages:
            best_iou = 0
            best_id = -1
            
            for pkg_id, tracked_pkg in self.tracked_packages.items():
                iou = calculate_iou(det_pkg['box'], tracked_pkg['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = pkg_id
            
            if best_iou > self.iou_threshold:
                # Update existing package
                self.tracked_packages[best_id]['box'] = det_pkg['box']
                self.tracked_packages[best_id]['last_seen'] = current_time
                matched_track_ids.add(best_id)
            else:
                # New package found! (Delivery Event?)
                new_id = self.next_package_id
                self.tracked_packages[new_id] = {
                    'box': det_pkg['box'],
                    'last_seen': current_time,
                    'created_at': current_time,
                    'status': 'monitor'
                }
                self.next_package_id += 1
                matched_track_ids.add(new_id)
                self.events.append(f"Package {new_id} Detected.")
        
        # 2.5 Run Face Tracking for Auto-Learning
        if self.face_manager and len(current_persons) > 0:
            # We pass the raw person boxes to the face manager
            # It will handle tracking and auto-learning
            try:
                self.face_manager.track_unknowns(frame, current_persons)
            except Exception as e:
                print(f"Error in auto-learning: {e}")

        # 3. Check for Disappeared Packages (Potential Theft)
        active_packages = {}
        detected_theft_event = None
        
        for pkg_id, pkg in self.tracked_packages.items():
            if pkg_id in matched_track_ids:
                active_packages[pkg_id] = pkg
            else:
                # Package was NOT seen this frame
                time_since_seen = current_time - pkg['last_seen']
                if time_since_seen < self.disappear_threshold:
                    # Keep it merely "missing" for a bit (occlusion handling)
                    active_packages[pkg_id] = pkg
                else:
                    # It's been gone too long. REMOVED.
                    if pkg['status'] != 'removed':
                        # Check if a person is present?
                        is_person_present = len(current_persons) > 0
                        
                        # LOGIC UPDATE: Check WHO is present
                        person_identity = "Unknown"
                        if is_person_present and self.face_manager:
                            # Try to identify the person closest to the package? 
                            # For now, valid if ANY present person is known
                            # We check the largest person box or the first one
                             for p_box in current_persons:
                                 name, _ = self.face_manager.identify_person(frame, p_box)
                                 if name != "Unknown":
                                     person_identity = name
                                     break

                        if is_person_present:
                            if person_identity != "Unknown":
                                status_msg = f"Package Taken by OWNER ({person_identity})"
                                event_msg = f"Package {pkg_id} {status_msg}"
                                self.events.append(event_msg)
                                # Authorized removal, no theft event
                            else:
                                status_msg = "Package Taken (Potential Theft!)"
                                event_msg = f"Package {pkg_id} {status_msg}"
                                self.events.append(event_msg)
                                detected_theft_event = {
                                    'type': 'theft',
                                    'msg': event_msg,
                                    'frames': list(self.frame_buffer)
                                }
                        else:
                            # Just disappeared (no person)
                            status_msg = "Package Disappeared"
                            self.events.append(f"Package {pkg_id} {status_msg}")
                        
                        pass 

        self.tracked_packages = active_packages
        
        status_text = f"Tracking: {len(self.tracked_packages)} pkgs."
        if self.events:
            status_text += f" Last: {self.events[-1]}"
            
        return status_text, self.tracked_packages, detected_theft_event
