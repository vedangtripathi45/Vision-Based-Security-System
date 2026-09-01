import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime
import os
from mtcnn import MTCNN
import pickle
import warnings
import logging
from keras_facenet import FaceNet
import serial
import threading

# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class DeepFaceRecognitionSystem:
    def __init__(self, tolerance=0.85):
        """
        Initialize Deep Learning Face Recognition System
        tolerance: Distance threshold for face matching (default: 0.7)
        Lower = stricter, Higher = more lenient
        Recommended: 0.6-0.8 for FaceNet
        """
        print("Loading models...")
        self.detector = MTCNN()
        self.embedder = FaceNet()  # Pre-trained FaceNet model
        self.tolerance = tolerance
        self.authorized_faces = []
        self.authorized_names = []
        self.min_detection_confidence = 0.9
        print("Models loaded successfully!")
        
    def extract_face_encoding(self, image, box):
        """Extract face embedding using FaceNet"""
        x, y, w, h = box
        
        # Handle negative coordinates
        x, y = max(0, x), max(0, y)
        h_img, w_img = image.shape[:2]
        
        # Ensure box is within image bounds
        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)
        
        if x >= x2 or y >= y2:
            return None
        
        face = image[y:y2, x:x2]
        
        if face.size == 0:
            return None
        
        try:
            # FaceNet expects 160x160 RGB images
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Expand dimensions for batch processing
            face_batch = np.expand_dims(face_rgb, axis=0)
            
            # Get embedding (128-dimensional vector)
            embedding = self.embedder.embeddings(face_batch)[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def register_face(self, image_path, name):
        """Register an authorized face from an image file"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image {image_path}")
            return False
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_image)
        except Exception as e:
            print(f"❌ Detection error in {image_path}: {e}")
            return False
        
        if len(detections) == 0:
            print(f"❌ No face detected in {image_path}")
            return False
        
        # Filter high confidence detections
        high_conf = [d for d in detections if d['confidence'] >= self.min_detection_confidence]
        
        if len(high_conf) == 0:
            print(f"❌ No high-confidence face in {image_path} (best: {detections[0]['confidence']:.2f})")
            return False
        
        # Use highest confidence detection
        best_detection = max(high_conf, key=lambda x: x['confidence'])
        box = best_detection['box']
        
        embedding = self.extract_face_encoding(image, box)
        
        if embedding is None:
            print(f"❌ Failed to extract embedding from {image_path}")
            return False
        
        self.authorized_faces.append(embedding)
        self.authorized_names.append(name)
        print(f"✓ Registered: {name} (conf: {best_detection['confidence']:.2f})")
        return True
    
    def register_face_multiple_samples(self, image_paths, name):
        """Register multiple images of same person for robustness"""
        success_count = 0
        
        for image_path in image_paths:
            if self.register_face(image_path, name):
                success_count += 1
        
        if success_count > 0:
            print(f"✓ Total: {success_count} samples registered for {name}")
            return True
        return False
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings using Euclidean distance"""
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance, distance <= self.tolerance
    
    def recognize_face(self, image):
        """
        Recognize faces in an image
        Returns: list of (box, name, is_authorized, confidence, distance) tuples
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_image)
        except:
            return []
        
        if not detections:
            return []
        
        results = []
        
        for detection in detections:
            try:
                box = detection['box']
                det_conf = detection['confidence']
                
                # Skip low confidence detections
                if det_conf < self.min_detection_confidence:
                    continue
                
                embedding = self.extract_face_encoding(image, box)
                
                if embedding is None:
                    continue
                
                # Find best match
                is_authorized = False
                recognized_name = "Unknown"
                best_distance = float('inf')
                
                for auth_embedding, name in zip(self.authorized_faces, self.authorized_names):
                    distance = np.linalg.norm(embedding - auth_embedding)
                    print("D:",distance)
                    
                    if distance < best_distance:
                        best_distance = distance
                        if distance <= self.tolerance:
                            is_authorized = True
                            recognized_name = name
                
                # Convert distance to confidence percentage
                # FaceNet distances: 0 = identical, 1.0+ = different
                match_confidence = max(0, min(100, (1 - best_distance) * 100))
                
                results.append((box, recognized_name, is_authorized, match_confidence, best_distance))
                
            except Exception as e:
                continue
        
        return results
    
    def save_database(self, filepath='deepface_database.pkl'):
        """Save authorized faces database"""
        data = {
            'faces': self.authorized_faces,
            'names': self.authorized_names,
            'tolerance': self.tolerance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Database saved to {filepath}")
    
    def load_database(self, filepath='deepface_database.pkl'):
        """Load authorized faces database"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.authorized_faces = data['faces']
            self.authorized_names = data['names']
            if 'tolerance' in data:
                self.tolerance = data['tolerance']
            print(f"✓ Loaded {len(self.authorized_names)} face samples")
            print(f"  Unique persons: {len(set(self.authorized_names))}")
            return True
        print(f"❌ Database not found: {filepath}")
        return False
    
    def recognize_from_webcam(self, show_distance=True):
        """Real-time face recognition from webcam"""
        cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("\n" + "="*50)
        print("🎥 FACE RECOGNITION ACTIVE")
        print("="*50)
        print(f"Tolerance: {self.tolerance}")
        print(f"Registered: {len(set(self.authorized_names))} person(s)")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'i' - Show info")
        print("\n💡 Tip: Face camera directly with good lighting\n")
        
        frame_count = 0
        show_info = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 2 frames for speed
            if frame_count % 1 == 0:
                results = self.recognize_face(frame)
                
                # Show "no face" message
                if len(results) == 0 and frame_count > 20:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Draw results
                for box, name, is_authorized, confidence, distance in results:
                    x, y, w, h = box
                    
                    # Color based on authorization
                    color = (0, 255, 0) if is_authorized else (0, 0, 255)
                    status = "✓ AUTHORIZED" if is_authorized else "✗ UNAUTHORIZED"
                    while True:
                        if is_authorized:
                            arduino.write(bytes([int(1)]))
                        else:
                            arduino.write(bytes([int(0)]))
                        break
                    # Draw box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw label background
                    label_h = 70 if show_distance else 55
                    cv2.rectangle(frame, (x, y-label_h), (x+w, y), color, -1)
                    
                    # Draw text
                    y_text = y - label_h + 15
                    cv2.putText(frame, name, (x+5, y_text), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    y_text += 20
                    cv2.putText(frame, status, (x+5, y_text), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    y_text += 18
                    cv2.putText(frame, f"Conf: {confidence:.1f}%", (x+5, y_text), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    if show_distance:
                        y_text += 15
                        cv2.putText(frame, f"Dist: {distance:.3f}", (x+5, y_text), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Show info overlay
            if show_info:
                info_bg = np.zeros((120, 300, 3), dtype=np.uint8)
                cv2.putText(info_bg, f"Tolerance: {self.tolerance}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_bg, f"Registered: {len(set(self.authorized_names))}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_bg, f"Samples: {len(self.authorized_names)}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(info_bg, "Press 'i' to hide", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
                frame[10:130, 10:310] = cv2.addWeighted(frame[10:130, 10:310], 0.3, info_bg, 0.7, 0)
            
            cv2.imshow('Deep Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                show_info = not show_info
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Shutdown complete")
    


        """Recognize faces in a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read: {image_path}")
            return
        
        results = self.recognize_face(image)
        
        print(f"\n{'='*50}")
        print(f"📸 Recognition Results: {image_path}")
        print(f"{'='*50}")
        
        if len(results) == 0:
            print("No faces detected")
        
        for i, (box, name, is_authorized, confidence, distance) in enumerate(results, 1):
            x, y, w, h = box
            color = (0, 255, 0) if is_authorized else (0, 0, 255)
            status = "AUTHORIZED ✓" if is_authorized else "UNAUTHORIZED ✗"
            
            # Draw on image
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            label_h = 70
            cv2.rectangle(image, (x, y-label_h), (x+w, y), color, -1)
            
            y_text = y - label_h + 18
            cv2.putText(image, name, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_text += 22
            cv2.putText(image, status, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_text += 18
            cv2.putText(image, f"Distance: {distance:.3f}", (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Print to console
            print(f"\nFace {i}:")
            print(f"  Name: {name}")
            print(f"  Status: {status}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Distance: {distance:.4f} (threshold: {self.tolerance})")
        
        print(f"\n{'='*50}\n")
        
        cv2.imshow('Recognition Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def recognize_from_image(self, image_path):
        """Recognize faces in a single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read: {image_path}")
            return
        
        results = self.recognize_face(image)
        
        print(f"\n{'='*50}")
        print(f"📸 Recognition Results: {image_path}")
        print(f"{'='*50}")
        
        if len(results) == 0:
            print("No faces detected")
        
        for i, (box, name, is_authorized, confidence, distance) in enumerate(results, 1):
            x, y, w, h = box
            color = (0, 255, 0) if is_authorized else (0, 0, 255)
            status = "AUTHORIZED ✓" if is_authorized else "UNAUTHORIZED ✗"
            
            # Draw on image
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
            label_h = 70
            cv2.rectangle(image, (x, y-label_h), (x+w, y), color, -1)
            
            y_text = y - label_h + 18
            cv2.putText(image, name, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_text += 22
            cv2.putText(image, status, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_text += 18
            cv2.putText(image, f"Distance: {distance:.3f}", (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Print to console
            print(f"\nFace {i}:")
            print(f"  Name: {name}")
            print(f"  Status: {status}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Distance: {distance:.4f} (threshold: {self.tolerance})")
        
        print(f"\n{'='*50}\n")
        
        cv2.imshow('Recognition Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
class AnomalyDetector:
    def __init__(self):
        """Anomaly Detection using MediaPipe Pose"""
        print("Loading Anomaly Detection models...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        
        self.fall_detected = False
        self.violence_detected = False
        self.danger_detected = False
        
        self.ground_time = 0
        self.last_ground_check = time.time()
        
        self.FALL_ANGLE_THRESHOLD = 60
        self.VIOLENCE_VELOCITY_THRESHOLD = 0.15
        self.GROUND_TIME_THRESHOLD = 3
        
        self.video_writer = None
        self.is_recording = False
        self.recording_filename = None
        self.event_log = []
        print("Anomaly Detection loaded!")
        
    def start_recording(self, frame_width, frame_height, fps=20):
        """Start video recording"""
        if not self.is_recording:
            if not os.path.exists('recordings'):
                os.makedirs('recordings')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_filename = f"recordings/security_{timestamp}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                self.recording_filename, fourcc, fps,
                (frame_width, frame_height)
            )
            
            self.is_recording = True
            self.event_log = []
            print(f"[RECORDING] Started: {self.recording_filename}")
            return True
        return False
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            
            log_filename = self.recording_filename.replace('.avi', '_log.txt')
            with open(log_filename, 'w') as f:
                f.write("Security System Event Log\n")
                f.write("=" * 50 + "\n")
                f.write(f"Recording: {self.recording_filename}\n")
                f.write(f"Total Events: {len(self.event_log)}\n")
                f.write("=" * 50 + "\n\n")
                for event in self.event_log:
                    f.write(f"{event}\n")
            
            print(f"[RECORDING] Stopped: {self.recording_filename}")
            print(f"[LOG] Saved: {log_filename}")
            return True
        return False
    
    def log_event(self, event_type, details):
        """Log event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_str = f"[{timestamp}] {event_type}: {details}"
        self.event_log.append(event_str)
        return event_str
    
    def calculate_angle(self, point1, point2):
        """Calculate angle from vertical"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        return min(angle, 180 - angle)
    
    def calculate_velocity(self, current_pos, previous_positions):
        """Calculate movement velocity"""
        if len(previous_positions) < 2:
            return 0
        
        velocities = []
        for i in range(len(previous_positions) - 1):
            if previous_positions[i] and previous_positions[i+1]:
                dx = previous_positions[i+1][0] - previous_positions[i][0]
                dy = previous_positions[i+1][1] - previous_positions[i][1]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0
    
    def detect_fall(self, landmarks, image_shape):
        """Detect if person has fallen"""
        h, w = image_shape[:2]
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2,
                       (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2,
                   (left_hip.y + right_hip.y) / 2)
        
        body_angle = self.calculate_angle(shoulder_mid, hip_mid)
        hip_below_knee = (hip_mid[1] > left_knee.y and hip_mid[1] > right_knee.y)
        fall_condition = body_angle > self.FALL_ANGLE_THRESHOLD or hip_below_knee
        
        return fall_condition, body_angle
    
    def detect_violence(self, landmarks, image_shape):
        """Detect violent movements"""
        h, w = image_shape[:2]
        
        left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        current_positions = {
            'left_hand': (left_hand.x, left_hand.y),
            'right_hand': (right_hand.x, right_hand.y),
            'left_foot': (left_foot.x, left_foot.y),
            'right_foot': (right_foot.x, right_foot.y)
        }
        
        self.position_history.append(current_positions)
        
        if len(self.position_history) >= 3:
            max_velocity = 0
            for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
                positions = [pos[limb] for pos in self.position_history if limb in pos]
                velocity = self.calculate_velocity(None, positions)
                max_velocity = max(max_velocity, velocity)
            
            violence_condition = max_velocity > self.VIOLENCE_VELOCITY_THRESHOLD
            return violence_condition, max_velocity
        
        return False, 0
    
    def detect_danger(self, landmarks, fall_detected):
        """Detect if person is in danger"""
        current_time = time.time()
        
        if fall_detected:
            if current_time - self.last_ground_check > 1:
                self.ground_time += 1
                self.last_ground_check = current_time
            
            if self.ground_time >= self.GROUND_TIME_THRESHOLD:
                return True, self.ground_time
        else:
            self.ground_time = 0
            self.last_ground_check = current_time
        
        return False, self.ground_time
    
    def process_frame(self, frame):
        """Process frame for anomaly detection"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        self.fall_detected = False
        self.violence_detected = False
        self.danger_detected = False
        
        fall_angle = 0
        violence_velocity = 0
        danger_time = 0
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            self.fall_detected, fall_angle = self.detect_fall(landmarks, frame.shape)
            self.violence_detected, violence_velocity = self.detect_violence(landmarks, frame.shape)
            self.danger_detected, danger_time = self.detect_danger(landmarks, self.fall_detected)
        
        return frame, {
            'fall': self.fall_detected,
            'violence': self.violence_detected,
            'danger': self.danger_detected,
            'fall_angle': fall_angle,
            'violence_velocity': violence_velocity,
            'danger_time': danger_time
        }
    
    def write_frame(self, frame):
        """Write frame to video"""
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

class ArduinoInterface:
    def __init__(self, port='COM8', baudrate=9600):
        """Initialize Arduino/ESP serial connection"""
        self.serial_conn = None
        self.connected = False
        self.face_recognition_trigger = False
        
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for connection
            self.connected = True
            print(f"✓ Arduino/ESP connected on {port}")
        except Exception as e:
            print(f"⚠ Arduino/ESP not connected: {e}")
            print("  System will work without Arduino trigger")
    
    def read_trigger(self):
        """Read trigger signal from Arduino/ESP"""
        if not self.connected or not self.serial_conn:
            return False
        
        try:
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.readline().decode('utf-8').strip()
                if data == "FACE_RECOGNITION":
                    return True
        except:
            pass
        return False
    
    def send_status(self, message):
        """Send status back to Arduino/ESP"""
        if self.connected and self.serial_conn:
            try:
                self.serial_conn.write(f"{message}\n".encode())
            except:
                pass
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn:
            self.serial_conn.close()

class IntegratedSecuritySystem:
    def __init__(self, arduino_port='/dev/ttyUSB0'):
        """Initialize integrated security system"""
        print("="*60)
        print("  INTEGRATED SECURITY SYSTEM")
        print("  Anomaly Detection + Face Recognition")
        print("="*60 + "\n")
        
        self.anomaly_detector = AnomalyDetector()
        self.face_recognition = FaceRecognitionModule(tolerance=0.85)
        self.arduino = ArduinoInterface(arduino_port)
        
        # Load face database
        self.face_recognition.load_database('deepface_database.pkl')
        
        self.frame_width = 640
        self.frame_height = 480
        
    def draw_interface(self, frame, anomaly_results, face_results):
        """Draw unified interface"""
        h, w = frame.shape[:2]
        
        # Anomaly Detection Status Box
        cv2.rectangle(frame, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 180), (255, 255, 255), 2)
        
        y_offset = 35
        
        # Recording indicator
        if self.anomaly_detector.is_recording:
            cv2.circle(frame, (30, 25), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (45, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Fall detection
        fall_color = (0, 0, 255) if anomaly_results['fall'] else (0, 255, 0)
        fall_text = f"FALL: {'DETECTED' if anomaly_results['fall'] else 'Normal'}"
        cv2.putText(frame, fall_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fall_color, 2)
        
        # Violence detection
        y_offset += 35
        violence_color = (0, 0, 255) if anomaly_results['violence'] else (0, 255, 0)
        violence_text = f"VIOLENCE: {'DETECTED' if anomaly_results['violence'] else 'Normal'}"
        cv2.putText(frame, violence_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, violence_color, 2)
        
        # Danger detection
        y_offset += 35
        danger_color = (0, 0, 255) if anomaly_results['danger'] else (0, 255, 0)
        danger_text = f"DANGER: {'DETECTED' if anomaly_results['danger'] else 'Normal'}"
        cv2.putText(frame, danger_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, danger_color, 2)
        
        # Face Recognition Status
        y_offset += 35
        face_status = "ACTIVE" if self.face_recognition.active else "STANDBY"
        face_color = (0, 255, 255) if self.face_recognition.active else (128, 128, 128)
        cv2.putText(frame, f"FACE REC: {face_status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Draw face recognition results
        for box, name, is_authorized, confidence, distance in face_results:
            x, y, w_box, h_box = box
            color = (0, 255, 0) if is_authorized else (0, 0, 255)
            status = "✓ AUTHORIZED" if is_authorized else "✗ UNAUTHORIZED"
            
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 3)
            
            label_h = 70
            cv2.rectangle(frame, (x, y-label_h), (x+w_box, y), color, -1)
            
            y_text = y - label_h + 15
            cv2.putText(frame, name, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_text += 20
            cv2.putText(frame, status, (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            y_text += 18
            cv2.putText(frame, f"Conf: {confidence:.1f}%", (x+5, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Overall alert
        if any([anomaly_results['fall'], anomaly_results['violence'], anomaly_results['danger']]):
            alert_text = "! ANOMALY DETECTED !"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, alert_text, (text_x, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return frame
    
    def run(self):
        """Main system loop"""
        print("\n" + "="*60)
        print("SYSTEM CONTROLS:")
        print("="*60)
        print("  'r' - Start/Stop Recording")
        print("  'f' - Toggle Face Recognition (manual)")
        print("  'q' - Quit")
        print("  Arduino/ESP sends 'FACE_RECOGNITION' to trigger")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("✓ System Active!\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        prev_fall = False
        prev_violence = False
        prev_danger = False
        
        face_recognition_cooldown = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            frame_count = fps_counter
            
            # Always run anomaly detection
            processed_frame, anomaly_results = self.anomaly_detector.process_frame(frame)
            
            # Check Arduino trigger
            if self.arduino.read_trigger():
                self.face_recognition.active = True
                face_recognition_cooldown = 150  # Run for 150 frames (~5 seconds)
                self.arduino.send_status("FACE_REC_ACTIVE")
                print("[TRIGGER] Face recognition activated by Arduino/ESP")
            
            # Run face recognition if active
            face_results = []
            if self.face_recognition.active:
                if frame_count % 2 == 0:  # Process every 2 frames
                    face_results = self.face_recognition.recognize_face(processed_frame)
                    
                    # Log unauthorized access
                    for box, name, is_authorized, confidence, distance in face_results:
                        if not is_authorized:
                            event = self.anomaly_detector.log_event(
                                "UNAUTHORIZED ACCESS", 
                                f"Unknown person detected (Conf: {confidence:.1f}%)"
                            )
                            print(event)
                            self.arduino.send_status("UNAUTHORIZED")
                        else:
                            print(f"[ACCESS] Authorized: {name}")
                            self.arduino.send_status(f"AUTHORIZED:{name}")
                
                face_recognition_cooldown -= 1
                if face_recognition_cooldown <= 0:
                    self.face_recognition.active = False
                    self.arduino.send_status("FACE_REC_STANDBY")
            
            # Log anomaly events
            if anomaly_results['fall'] and not prev_fall:
                event = self.anomaly_detector.log_event(
                    "FALL", f"Angle: {anomaly_results['fall_angle']:.1f}°"
                )
                print(event)
                self.arduino.send_status("FALL_DETECTED")
            
            if anomaly_results['violence'] and not prev_violence:
                event = self.anomaly_detector.log_event(
                    "VIOLENCE", f"Velocity: {anomaly_results['violence_velocity']:.3f}"
                )
                print(event)
                self.arduino.send_status("VIOLENCE_DETECTED")
            
            if anomaly_results['danger'] and not prev_danger:
                event = self.anomaly_detector.log_event(
                    "DANGER", f"Ground time: {anomaly_results['danger_time']:.1f}s"
                )
                print(event)
                self.arduino.send_status("DANGER_DETECTED")
            
            prev_fall = anomaly_results['fall']
            prev_violence = anomaly_results['violence']
            prev_danger = anomaly_results['danger']
            
            # Draw interface
            processed_frame = self.draw_interface(processed_frame, anomaly_results, face_results)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(processed_frame, f"FPS: {fps}", (10, self.frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write to video if recording
            self.anomaly_detector.write_frame(processed_frame)
            
            # Display
            cv2.imshow('Integrated Security System', processed_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.anomaly_detector.is_recording:
                    self.anomaly_detector.start_recording(self.frame_width, self.frame_height)
                else:
                    self.anomaly_detector.stop_recording()
            elif key == ord('f'):
                self.face_recognition.active = not self.face_recognition.active
                status = "activated" if self.face_recognition.active else "deactivated"
                print(f"[MANUAL] Face recognition {status}")
                face_recognition_cooldown = 150 if self.face_recognition.active else 0
        
        # Cleanup
        if self.anomaly_detector.is_recording:
            self.anomaly_detector.stop_recording()
        
        self.arduino.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System shutdown complete")

if __name__ == "__main__":

    arduino = serial.Serial('COM8', 9600)  
    time.sleep(2)  # Wait for connection to establish
    value=0
    flag=1
    prev_value=0
    while True:
        if arduino.in_waiting > 0:          # Check if data is available
            value = arduino.read()          # Read one byte
            value = int.from_bytes(value, 'big')  # Convert byte to int (0 or 1)
            if value==1 and prev_value==0:
                flag=0
            #print(value)  # Prints 0 or 1
            prev_value=value

        if value == 1 and flag!=1:
            flag=1
            print("="*60)
            print("  DEEP LEARNING FACE RECOGNITION SYSTEM")
            print("  Using FaceNet for accurate face embeddings")
            print("="*60 + "\n")
    
            # Initialize with FaceNet
            system = DeepFaceRecognitionSystem(tolerance=0.85)
    
            print("\n" + "="*60)
            print("STEP 1: Register Authorized Faces")
            print("="*60)
    
            # Register faces
            #system.register_face('utkarsh.jpg', 'Utkarsh')
            #system.register_face('vedang.jpg', 'Vedang')
            
            # Optional: Register multiple samples per person (RECOMMENDED)
            # system.register_face_multiple_samples([
            #     'john1.jpg', 'john2.jpg', 'john3.jpg'
            # ], 'John Doe')
            
            # Save database
            #system.save_database('deepface_database.pkl')
            
            # Or load existing
            system.load_database('deepface_database.pkl')
            
            print("\n" + "="*60)
            print("STEP 2: Test Recognition")
            print("="*60)
            
            # Test with images first
            print("\n📸 Testing with authorized person...")
            #system.recognize_from_image('N.jpg')
            
            '''print("\n📸 Testing with different person...")
            system.recognize_from_image('different_person.jpg')'''
    
            # Start webcam
            #input("\nPress Enter to start webcam recognition...")
            system.recognize_from_webcam(show_distance=True)
        else:
            cap = cv2.VideoCapture(1)
        
            if not cap.isOpened():
                print("❌ Could not open webcam")
            
            while True:
                ret, frame = cap.read()
                sys=AnomalyDetector()
                sys.process_frame(frame)
