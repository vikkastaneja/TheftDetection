import face_recognition
import cv2
import os
import pickle
import numpy as np

class FaceManager:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Auto-Learning State
        # active_unknowns: dict { 'id': {'box': box, 'seen_count': 0, 'last_seen': time, 'encoding': enc} }
        self.active_unknowns = {}
        self.next_unknown_id = 1
        
        self.load_known_faces()

    def load_known_faces(self):
        """Loads face encodings from the faces directory."""
        print("Loading known faces...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            return

        for filename in os.listdir(self.faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(name)
                        print(f"Loaded face: {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        print(f"Total known faces: {len(self.known_face_names)}")

    def register_face(self, frame, name, box=None):
        """
        Registers a new face from a frame.
        box: (top, right, bottom, left) or None to find face automatically.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if box:
            x1, y1, x2, y2 = map(int, box)
            face_locations = [(y1, x2, y2, x1)]
        else:
            face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return False, "No face detected"

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not encodings:
            return False, "Could not encode face"

        # Check if this face is ALREADY known (to prevent duplicates)
        distances = face_recognition.face_distance(self.known_face_encodings, encodings[0])
        if len(distances) > 0 and np.min(distances) < 0.5:
             return False, f"Face already known as {self.known_face_names[np.argmin(distances)]}"

        # Save image for database
        # Note: We save the image because the 'model' is just a fixed extractor.
        # The 'learning' is effectively adding this reference image/encoding to our list.
        # Saving the file allows you to audit 'Who did it learn?'.
        save_path = os.path.join(self.faces_dir, f"{name}.jpg")
        cv2.imwrite(save_path, frame)
        
        # Update memory
        self.known_face_encodings.append(encodings[0])
        self.known_face_names.append(name)
        
        return True, f"Registered {name}"

    def track_unknowns(self, frame, person_boxes):
        """
        Updates tracking for unknown faces to handle Auto-Learning.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = frame.shape
        
        # 1. Identify current faces
        # We need to associate person_boxes with faces.
        # For simplicity, we just look for faces in the frame and try to track them.
        # Ideally, we crop the person_box and find the face inside.
        
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract person crop to find face inside (more robust)
            person_crop = rgb_frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
            if person_crop.size == 0: continue
            
            face_locs = face_recognition.face_locations(person_crop)
            if not face_locs: continue
            
            # Encodings in the crop
            face_encs = face_recognition.face_encodings(person_crop, face_locs)
            if not face_encs: continue
            
            encoding = face_encs[0] # Assume one face per person box
            
            # Check if KNOWN
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
            if True in matches:
                continue # Already known, ignore

            # UNKNOWN FACE -> TRACK IT
            # 1. Simple matching to existing persistence tracker (using encoding distance)
            matched_id = None
            for uid, udata in self.active_unknowns.items():
                dist = face_recognition.face_distance([udata['encoding']], encoding)[0]
                if dist < 0.5:
                    matched_id = uid
                    break
            
            if matched_id is not None:
                # Update existing
                self.active_unknowns[matched_id]['seen_count'] += 1
                self.active_unknowns[matched_id]['encoding'] = encoding # Update to latest
            else:
                # New Unknown
                matched_id = self.next_unknown_id
                self.next_unknown_id += 1
                self.active_unknowns[matched_id] = {
                    'encoding': encoding,
                    'seen_count': 1,
                    'start_time': 0 # Could use time
                }

            # AUTO-LEARN LOGIC
            # 1. Proximity Check: Face width relative to frame width
            # face_locs[0] is (top, right, bottom, left) relative to CROP
            # We need width relative to FULL FRAME
            # Actually, the 'box' (x1, y1, x2, y2) is the person body size.
            # Let's use the PERSON BODY width as proxy for proximity? 
            # Or better, the face size.
            top, right, bottom, left = face_locs[0]
            face_width = right - left
            width_ratio = face_width / frame_w
            
            # Thresholds
            PROXIMITY_THRESHOLD = 0.10 # Face must be 10% of screen width (Close)
            PERSISTENCE_THRESHOLD = 50 # 50 Frames (~2-3 seconds at 20fps logic)
            
            if width_ratio > PROXIMITY_THRESHOLD:
                if self.active_unknowns[matched_id]['seen_count'] > PERSISTENCE_THRESHOLD:
                    # HEURISTIC MET! TRUST THIS PERSON.
                    new_name = f"Resident_Auto_{matched_id}"
                    print(f"[AUTO-LEARN] Trusting new face: {new_name}")
                    
                    # Register
                    # We need the full frame coordinates for register_face helper, 
                    # OR just append manually.
                    # Adapting register_face to take encoding would be cleaner, but let's just append.
                    
                    # Save reference image
                    save_path = os.path.join(self.faces_dir, f"{new_name}.jpg")
                    cv2.imwrite(save_path, frame[y1:y2, x1:x2]) # Save person crop
                    
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(new_name)
                    
                    # cleanup
                    del self.active_unknowns[matched_id]

    def identify_person(self, frame, box=None):
        """
        Identifies a person in the frame.
        Returns: name ("Unknown" if not found), distance
        """
        if not self.known_face_encodings:
            return "Unknown", 1.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if box:
            x1, y1, x2, y2 = map(int, box)
            # Add padding/constraints?
            face_locations = [(y1, x2, y2, x1)]
        else:
            face_locations = face_recognition.face_locations(rgb_frame)

        if not face_locations:
            return "Unknown (No Face)", 1.0

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            return "Unknown (Encoding Fail)", 1.0

        # We assume one person in the box, so check the first encoding
        encoding = face_encodings[0]
        
        # Compare with known faces
        matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return self.known_face_names[best_match_index], face_distances[best_match_index]
        
        return "Unknown", min(face_distances) if len(face_distances) > 0 else 1.0

    def find_best_face_frame(self, frames):
        """
        Scans a list of frames and returns the one with the largest/clearest face.
        If no faces found, returns the last frame.
        """
        best_frame = frames[-1] # Default to last frame
        max_face_area = 0

        # Optimization: Don't scan every single frame if buffer is huge, but 30 is fine.
        # We can skip frames to be faster if needed (e.g. every 2nd frame)
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Find largest face in this frame
                for top, right, bottom, left in face_locations:
                    area = (bottom - top) * (right - left)
                    if area > max_face_area:
                        max_face_area = area
                        best_frame = frame
        
        if max_face_area > 0:
            print(f"[Smart Evidence] Found best face with area {max_face_area}")
        else:
            print("[Smart Evidence] No faces found in buffer. Using last frame.")
            
        return best_frame
