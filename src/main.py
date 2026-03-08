import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from utils import calculate_EAR, calculate_MAR, head_tilt_angle

pygame.mixer.init()
pygame.mixer.music.load("assets/alarm.wav")

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [78,81,13,311,308,402,14,178,88,95,185]

cap = cv2.VideoCapture(0)

blink_count = 0
sleep_frames = 0

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.6

fatigue_history = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mesh = []

            for p in face_landmarks.landmark:
                mesh.append((int(p.x*w), int(p.y*h)))

            # face box
            xs = [p[0] for p in mesh]
            ys = [p[1] for p in mesh]

            cv2.rectangle(frame,
                          (min(xs),min(ys)),
                          (max(xs),max(ys)),
                          (0,255,0),2)

            # eye landmarks
            for idx in LEFT_EYE+RIGHT_EYE:
                cv2.circle(frame, mesh[idx],2,(0,255,255),-1)

            # mouth landmarks
            for idx in MOUTH:
                cv2.circle(frame, mesh[idx],2,(255,0,255),-1)

            left_eye = [mesh[i] for i in LEFT_EYE]
            right_eye = [mesh[i] for i in RIGHT_EYE]
            mouth = [mesh[i] for i in MOUTH]

            EAR = (calculate_EAR(left_eye)+calculate_EAR(right_eye))/2
            MAR = calculate_MAR(mouth)

            # blink detection
            if EAR < EAR_THRESHOLD:
                sleep_frames += 1
            else:
                if sleep_frames > 3:
                    blink_count += 1
                sleep_frames = 0

            # yawning detection
            yawning = MAR > MAR_THRESHOLD

            # head tilt
            nose = mesh[1]
            chin = mesh[152]

            tilt = head_tilt_angle(nose,chin)

            # fatigue score
            fatigue = int((1-EAR)*100)

            fatigue_history.append(fatigue)

            if len(fatigue_history) > 100:
                fatigue_history.pop(0)

            # warning
            if sleep_frames > 20:

                play_alarm()

                cv2.putText(frame,
                            "WARNING DRIVER SLEEPING",
                            (40,60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,0,255),3)

            if yawning:
                cv2.putText(frame,
                            "Yawning Detected",
                            (40,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,0,255),2)

            # attention detection
            if abs(tilt) > 25:

                cv2.putText(frame,
                            "Driver Not Looking Forward",
                            (40,140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,(0,165,255),2)

            # dashboard
            cv2.rectangle(frame,(10,10),(320,180),(0,0,0),-1)

            cv2.putText(frame,f"EAR: {EAR:.2f}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,255),2)

            cv2.putText(frame,f"Blink Rate: {blink_count}",
                        (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2)

            cv2.putText(frame,f"Head Tilt: {tilt:.1f}",
                        (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(255,255,0),2)

            cv2.putText(frame,f"Fatigue Score: {fatigue}",
                        (20,130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,255),2)

    cv2.imshow("Driver Monitoring System",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()