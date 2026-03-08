import cv2
import dlib
import numpy as np
import threading
import winsound

from scipy.spatial import distance as dist
from imutils import face_utils

# -------------------------------
# Alarm Function (Windows Sound)
# -------------------------------
def sound_alarm():
    duration = 1000   # milliseconds
    frequency = 2500  # Hz
    winsound.Beep(frequency, duration)


# -------------------------------
# Eye Aspect Ratio Calculation
# -------------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# -------------------------------
# Threshold Settings
# -------------------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10

COUNTER = 0
ALARM_ON = False


# -------------------------------
# Load Face Detector
# -------------------------------
print("[INFO] Loading models...")

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# -------------------------------
# Start Webcam
# -------------------------------
print("[INFO] Starting camera...")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # -------------------------------
        # Drowsiness Logic
        # -------------------------------
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    3,
                )

                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm).start()

        else:
            COUNTER = 0
            ALARM_ON = False

        # Display EAR value
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break


cap.release()
cv2.destroyAllWindows()