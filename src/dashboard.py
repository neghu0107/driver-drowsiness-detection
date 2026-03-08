import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from src.utils import calculate_ear

st.set_page_config(layout="wide")

st.title("🚗 Driver Fatigue Monitoring Dashboard")

# Sidebar
st.sidebar.title("Driver Info")
driver = st.sidebar.text_input("Driver Name")
vehicle = st.sidebar.text_input("Vehicle ID")

# Layout
col1, col2 = st.columns([3,1])

frame_window = col1.image([])

metric_ear = col2.empty()
metric_blink = col2.empty()
metric_fatigue = col2.empty()
metric_status = col2.empty()

# Load model
model = pickle.load(open("models/fatigue_model.pkl","rb"))

# Mediapipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Eye landmarks
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

cap = cv2.VideoCapture(0)

blink_count = 0
ear_threshold = 0.23

run = st.checkbox("Start Camera")

while run:

    ret, frame = cap.read()

    if not ret:
        st.write("Camera error")
        break

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    ear = 0
    fatigue_score = 0

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            points = []

            for lm in face_landmarks.landmark:

                x = int(lm.x*w)
                y = int(lm.y*h)

                points.append((x,y))

            # Draw face box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            cv2.rectangle(frame,
                          (min(xs),min(ys)),
                          (max(xs),max(ys)),
                          (0,255,0),2)

            # Draw eye landmarks
            for i in LEFT_EYE+RIGHT_EYE:
                cv2.circle(frame,points[i],2,(0,255,255),-1)

            left_eye = np.array([points[i] for i in LEFT_EYE])
            right_eye = np.array([points[i] for i in RIGHT_EYE])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2

            if ear < ear_threshold:
                blink_count += 1

            features = np.array([[ear, ear*1.5, blink_count, 0]])

            fatigue_score = model.predict_proba(features)[0][1]

            if fatigue_score > 0.7:

                cv2.putText(frame,
                            "WARNING: DROWSY DRIVER",
                            (40,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

    frame_window.image(frame,channels="BGR")

    metric_ear.metric("EAR", round(ear,3))
    metric_blink.metric("Blink Count", blink_count)
    metric_fatigue.metric("Fatigue Score", round(float(fatigue_score),2))

    if fatigue_score > 0.7:
        metric_status.metric("Driver Status","DROWSY")
    else:
        metric_status.metric("Driver Status","ALERT")