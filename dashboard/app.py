import streamlit as st
import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance as dist
from collections import deque
import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(page_title="Driver Monitoring Dashboard", layout="wide")

st.title("🚗 AI Driver Monitoring Dashboard")

# -----------------------------
# DRIVER DATABASE
# -----------------------------

drivers = {
    "Rahul": {"vehicle": "TN01A1234", "location": "Chennai"},
    "Arjun": {"vehicle": "KA05B8899", "location": "Bangalore"},
    "Priya": {"vehicle": "MH12D4567", "location": "Mumbai"},
}

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Driver Selection")

driver_name = st.sidebar.selectbox("Select Driver", list(drivers.keys()))

vehicle_id = drivers[driver_name]["vehicle"]
driver_location = drivers[driver_name]["location"]

start = st.sidebar.button("Start Monitoring")

st.sidebar.markdown("---")

st.sidebar.subheader("Driver Info")
st.sidebar.write("Driver:", driver_name)
st.sidebar.write("Vehicle:", vehicle_id)
st.sidebar.write("Location:", driver_location)

# -----------------------------
# FLEET STATUS HEADER
# -----------------------------

st.caption(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}")

# -----------------------------
# MAIN LAYOUT
# -----------------------------

col1, col2 = st.columns([3,1])

frame_window = col1.empty()

with col2:

    st.header("Summary")

    ear_metric = st.metric("EAR", "0")
    mar_metric = st.metric("MAR", "0")
    blink_metric = st.metric("Blink Count", "0")
    fatigue_metric = st.metric("Fatigue Score", "0")

    st.subheader("Fatigue Level")
    fatigue_bar = st.progress(0)

    status_box = st.empty()

    st.subheader("Location")
    st.info(driver_location)

# -----------------------------
# EAR TREND
# -----------------------------

st.markdown("---")
st.subheader("EAR Trend")
ear_chart = st.empty()

# -----------------------------
# FATIGUE GRAPH + SUMMARY
# -----------------------------

colA, colB = st.columns([3,1])

with colA:
    st.subheader("Driver Fatigue History")
    fatigue_chart = st.empty()

with colB:
    st.subheader("Fatigue Summary")
    fatigue_text = st.empty()

# -----------------------------
# LOAD MODEL
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "shape_predictor_68_face_landmarks.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# -----------------------------
# FUNCTIONS
# -----------------------------

def calculate_ear(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)


def calculate_mar(mouth):

    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])

    return (A + B + C) / (2.0 * D)

# -----------------------------
# CAMERA START
# -----------------------------

if start:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
        st.stop()

    blink_count = 0
    frame_counter = 0

    EYE_THRESHOLD = 0.25
    CONSEC_FRAMES = 15

    ear_values = deque(maxlen=50)
    fatigue_history = deque(maxlen=100)

    while True:

        ret, frame = cap.read()

        if not ret:
            st.error("Camera not returning frames")
            break

        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:

            shape = predictor(gray, face)

            coords = np.zeros((68,2))

            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            coords = coords.astype(int)

            # FACE BOX

            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

            # EYES

            leftEye = coords[36:42]
            rightEye = coords[42:48]

            ear_left = calculate_ear(leftEye)
            ear_right = calculate_ear(rightEye)

            ear = (ear_left + ear_right) / 2.0

            # MOUTH

            mouth = coords[48:68]

            mar = calculate_mar(mouth)

            # BLINK DETECTION

            if ear < EYE_THRESHOLD:

                frame_counter += 1

                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1

            else:
                frame_counter = 0

            # FATIGUE SCORE

            fatigue_score = min(1.0, blink_count / 50)

            if ear < 0.22:

                cv2.putText(
                    frame,
                    "WARNING: DRIVER DROWSY",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

            # UPDATE HISTORY

            ear_values.append(ear)
            fatigue_history.append(fatigue_score)

            # UPDATE METRICS

            ear_metric.metric("EAR", round(ear,3))
            mar_metric.metric("MAR", round(mar,3))
            blink_metric.metric("Blink Count", blink_count)
            fatigue_metric.metric("Fatigue Score", round(fatigue_score,2))

            fatigue_bar.progress(int(fatigue_score*100))

            # STATUS

            if fatigue_score > 0.7:
                status_box.error("⚠ Driver Drowsy")
            else:
                status_box.success("✅ Driver Alert")

            # SUMMARY TEXT

            if fatigue_score < 0.3:
                fatigue_text.success("Driver is alert and focused.")
            elif fatigue_score < 0.7:
                fatigue_text.warning("Driver showing signs of fatigue.")
            else:
                fatigue_text.error("High fatigue detected! Take a break.")

        # DISPLAY FRAME

        frame_window.image(frame, channels="BGR")

        # UPDATE GRAPHS

        ear_chart.line_chart(list(ear_values))
        fatigue_chart.line_chart(list(fatigue_history))

    cap.release()