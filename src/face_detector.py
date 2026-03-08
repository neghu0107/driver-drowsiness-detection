import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

previous_faces = -1   # store last detected count


def detect_face(frame):
    global previous_faces

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Print only if count changes
    if len(faces) != previous_faces:
        print("Faces detected:", len(faces))
        previous_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y),
                      (x+w, y+h),
                      (0, 255, 0), 2)

    return frame