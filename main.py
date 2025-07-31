import cv2
import streamlit as st
import numpy as np
import time
from PIL import Image
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Custom FaceMeshDetector class to replace cvzone
class FaceMeshDetector:
    def __init__(self, maxFaces=1, staticMode=False):
        self.maxFaces = maxFaces
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=staticMode,
            max_num_faces=maxFaces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)
        faces = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
                
                if draw:
                    mp_drawing.draw_landmarks(
                        img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        None, mp_drawing_styles.get_default_face_mesh_contours_style())
        
        return img, faces
    
    def findDistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance, [(x1 + x2) // 2, (y1 + y2) // 2]

# Custom LivePlot class to replace cvzone
class LivePlot:
    def __init__(self, w, h, yLimit, invert=False):
        self.yLimit = yLimit
        self.w = w
        self.h = h
        self.invert = invert
        self.xP = 0
        self.yList = []
        
    def update(self, y):
        if len(self.yList) < self.w // 20:
            self.yList.append(y)
        else:
            self.yList.pop(0)
            self.yList.append(y)
            
        imgPlot = np.zeros((self.h, self.w, 3), np.uint8)
        
        # Draw plot lines
        for i in range(1, len(self.yList)):
            x1 = int((i - 1) * 20)
            x2 = int(i * 20)
            
            y1 = int(np.interp(self.yList[i-1], self.yLimit, [self.h, 0] if self.invert else [0, self.h]))
            y2 = int(np.interp(self.yList[i], self.yLimit, [self.h, 0] if self.invert else [0, self.h]))
            
            cv2.line(imgPlot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return imgPlot

# Custom text rectangle function to replace cvzone
def putTextRect(img, text, pos, scale=1, thickness=1, colorT=(255, 255, 255), colorR=(0, 0, 0), 
                font=cv2.FONT_HERSHEY_SIMPLEX, offset=10, border=None, colorBorder=(0, 0, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorBorder, border)
    
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    return img, [x1, y2, x2, y1]

# Custom stack images function to replace cvzone
def stackImages(imgArray, cols, scale):
    rows = len(imgArray) // cols
    
    for x in range(0, rows):
        for y in range(0, cols):
            imgArray[x * cols + y] = cv2.resize(imgArray[x * cols + y], 
                                               (0, 0), None, scale, scale)
    
    imageBlank = np.zeros((imgArray[0].shape[0], imgArray[0].shape[1], 3), np.uint8)
    hor = [imageBlank] * rows
    
    for x in range(0, rows):
        hor[x] = np.hstack(imgArray[x * cols:x * cols + cols])
    
    ver = np.vstack(hor)
    return ver

# Initialize face mesh detector (cached for performance)
@st.cache_resource
def get_detector():
    return FaceMeshDetector(maxFaces=1, staticMode=False)

detector = get_detector()

# Set up the app
st.set_page_config(page_title="Advanced Eye Blink Detection", layout="wide")
st.title("Advanced Eye Blink Detection with Head Movement Rejection")

# Initialize session state
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'threshold': 35,
        'camera_index': 0,
        'available_cameras': [],
        'last_warning': 0,
        'stability_threshold': 5
    }
    st.session_state.detection = {
        'blink_count': 0,
        'counter': 0,
        'ratio_list': [],
        'running': False,
        'cap': None,
        'head_stable': True,
        'prev_landmarks': None,
        'fps': 0
    }

# Camera discovery
def find_available_cameras(max_tests=3):
    available = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Camera selection
    st.session_state.settings['available_cameras'] = find_available_cameras()
    if st.session_state.settings['available_cameras']:
        cam_idx = st.selectbox("Select Camera", 
                             st.session_state.settings['available_cameras'],
                             index=0)
        if cam_idx != st.session_state.settings['camera_index']:
            st.session_state.settings['camera_index'] = cam_idx
            if st.session_state.detection['cap'] is not None:
                st.session_state.detection['cap'].release()
            st.session_state.detection['cap'] = cv2.VideoCapture(cam_idx)
    
    use_webcam = st.checkbox("Use Webcam", value=True)
    video_file = st.file_uploader("Or upload a video file", type=["mp4", "avi", "mov"])
    
    st.session_state.settings['threshold'] = st.slider("Blink Threshold", 20, 50, 35)
    st.session_state.settings['stability_threshold'] = st.slider("Head Stability Threshold", 1, 10, 5)
    show_plot = st.checkbox("Show Blink Ratio Plot", value=True)
    show_warnings = st.checkbox("Show Movement Warnings", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Detection", key="start_button"):
            st.session_state.detection['running'] = True
    with col2:
        if st.button("Stop Detection", key="stop_button"):
            st.session_state.detection['running'] = False
    
    if st.button("Reset Counter", key="reset_button"):
        st.session_state.detection['blink_count'] = 0

# Landmark indices for MediaPipe (different from cvzone)
# Eye landmarks for blink detection
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Specific landmarks for blink ratio calculation
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 23
LEFT_EYE_LEFT = 130
LEFT_EYE_RIGHT = 243

# Landmarks for head stability
stability_landmarks = [1, 152, 33, 263, 61, 291]

# Initialize plot
plotY = LivePlot(640, 360, [20, 50], invert=True)

# Create placeholders
col1, col2 = st.columns([2, 1])
video_placeholder = col1.empty()
metrics_placeholder = col2.empty()
warning_placeholder = st.empty()

def initialize_capture():
    if use_webcam:
        try:
            st.session_state.detection['cap'] = cv2.VideoCapture(st.session_state.settings['camera_index'])
            if not st.session_state.detection['cap'].isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.detection['running'] = False
                return False
        except Exception as e:
            st.error(f"Error accessing webcam: {e}")
            st.session_state.detection['running'] = False
            return False
    else:
        if video_file is not None:
            try:
                with open("temp_video.mp4", "wb") as f:
                    f.write(video_file.read())
                st.session_state.detection['cap'] = cv2.VideoCapture("temp_video.mp4")
            except Exception as e:
                st.error(f"Error processing video file: {e}")
                st.session_state.detection['running'] = False
                return False
        else:
            st.warning("Please enable webcam or upload a video file to start detection.")
            st.session_state.detection['running'] = False
            return False
    return True

def check_head_stability(face):
    """Check if head is stable enough for blink detection"""
    current_landmarks = np.array([face[i] for i in stability_landmarks])
    
    if st.session_state.detection['prev_landmarks'] is None:
        st.session_state.detection['prev_landmarks'] = current_landmarks
        return True
    
    # Calculate movement between frames
    movement = np.mean(np.abs(current_landmarks - st.session_state.detection['prev_landmarks']))
    st.session_state.detection['prev_landmarks'] = current_landmarks
    
    return movement < st.session_state.settings['stability_threshold']

# Main processing loop
if st.session_state.detection['running']:
    if st.session_state.detection['cap'] is None or not st.session_state.detection['cap'].isOpened():
        if not initialize_capture():
            st.stop()
    
    prev_time = time.time()
    frame_count = 0
    
    while st.session_state.detection['running'] and st.session_state.detection['cap'].isOpened():
        success, img = st.session_state.detection['cap'].read()
        if not success:
            if not use_webcam:
                st.warning("Video ended. Upload another video or switch to webcam.")
                st.session_state.detection['running'] = False
                break
            continue

        # Flip horizontally for mirror effect
        img = cv2.flip(img, 1) if use_webcam else img

        # Detect face mesh
        img, faces = detector.findFaceMesh(img, draw=False)
        ratioAvg = 0
        status = "No face detected"
        blink_detected = False

        if faces:
            face = faces[0]
            
            # Check head stability
            st.session_state.detection['head_stable'] = check_head_stability(face)
            
            # Draw eye landmarks
            for id in LEFT_EYE + RIGHT_EYE:
                if id < len(face):
                    cv2.circle(img, tuple(face[id]), 2, (255, 0, 255), cv2.FILLED)

            # Blink detection using left eye
            if all(id < len(face) for id in [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT]):
                leftUp = face[LEFT_EYE_TOP]
                leftDown = face[LEFT_EYE_BOTTOM]
                leftLeft = face[LEFT_EYE_LEFT]
                leftRight = face[LEFT_EYE_RIGHT]

                lengthVer, _ = detector.findDistance(leftUp, leftDown)
                lengthHor, _ = detector.findDistance(leftLeft, leftRight)

                cv2.line(img, tuple(leftUp), tuple(leftDown), (0, 200, 0), 2)
                cv2.line(img, tuple(leftLeft), tuple(leftRight), (0, 200, 0), 2)

                ratio = int((lengthVer / lengthHor) * 100) if lengthHor > 0 else 0
                st.session_state.detection['ratio_list'].append(ratio)

                if len(st.session_state.detection['ratio_list']) > 3:
                    st.session_state.detection['ratio_list'].pop(0)

                ratioAvg = sum(st.session_state.detection['ratio_list']) / len(st.session_state.detection['ratio_list']) if st.session_state.detection['ratio_list'] else 0
            
            # Status determination
            if not st.session_state.detection['head_stable']:
                status = "HEAD MOVING"
                if show_warnings and time.time() - st.session_state.settings['last_warning'] > 2:
                    warning_placeholder.warning("Please keep your head still for accurate detection")
                    st.session_state.settings['last_warning'] = time.time()
            elif ratioAvg < st.session_state.settings['threshold']:
                status = "BLINK DETECTED"
                blink_detected = True
            else:
                status = "READY"
                warning_placeholder.empty()

            # Blink counting logic (only when head is stable)
            if st.session_state.detection['head_stable'] and blink_detected:
                if st.session_state.detection['counter'] == 0:
                    st.session_state.detection['blink_count'] += 1
                    st.session_state.detection['counter'] = 1
                
                if st.session_state.detection['counter'] != 0:
                    st.session_state.detection['counter'] += 1
                    if st.session_state.detection['counter'] > 10:
                        st.session_state.detection['counter'] = 0
            else:
                st.session_state.detection['counter'] = 0

            # Display blink count on image
            img, _ = putTextRect(img, f'Blinks: {st.session_state.detection["blink_count"]}', 
                               (50, 50), scale=1.5, thickness=2, colorR=(0, 200, 0))
            img, _ = putTextRect(img, f'Status: {status}', (50, 100), 
                               scale=1.0, thickness=1, 
                               colorR=(200, 0, 0) if "BLINK" in status else (0, 200, 0))
            
            # Update plot if enabled
            if show_plot:
                imgPlot = plotY.update(ratioAvg)
                img = cv2.resize(img, (640, 360))
                imgStack = stackImages([img, imgPlot], 2, 1)
            else:
                imgStack = cv2.resize(img, (640, 360))
        else:
            imgStack = cv2.resize(img, (640, 360))
            if show_plot:
                imgStack = stackImages([imgStack, np.zeros((360, 640, 3), dtype=np.uint8)], 2, 1)
            warning_placeholder.empty()

        # Convert to RGB for Streamlit
        imgStack = cv2.cvtColor(imgStack, cv2.COLOR_BGR2RGB)
        
        # Display video feed
        video_placeholder.image(imgStack, channels="RGB", use_container_width=True)
        
        # Display metrics
        with metrics_placeholder.container():
            st.metric("Total Blinks", st.session_state.detection['blink_count'])
            st.metric("Current Ratio", f"{ratioAvg:.1f}" if faces else "N/A")
            st.metric("Head Stable", "Yes" if st.session_state.detection['head_stable'] else "No")
            st.metric("Detection Threshold", st.session_state.settings['threshold'])
            
            if faces:
                progress_value = ratioAvg / 50 if ratioAvg < 50 else 1.0
                st.progress(progress_value, text=f"Eye Aspect Ratio: {ratioAvg:.1f}")
        
        # Performance monitoring
        frame_count += 1
        if frame_count % 10 == 0:
            st.session_state.detection['fps'] = 10 / (time.time() - prev_time)
            prev_time = time.time()
        
        # Small delay to prevent high CPU usage
        time.sleep(0.01)

# Clean up when stopping
if not st.session_state.detection['running'] and st.session_state.detection['cap'] is not None:
    st.session_state.detection['cap'].release()
    st.session_state.detection['cap'] = None
