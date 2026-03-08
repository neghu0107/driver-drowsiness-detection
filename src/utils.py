import numpy as np

def calculate_ear(eye):

    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    ear = (A + B) / (2.0 * C)

    return ear


def calculate_mar(mouth):

    A = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[10]))
    B = np.linalg.norm(np.array(mouth[4]) - np.array(mouth[8]))
    C = np.linalg.norm(np.array(mouth[0]) - np.array(mouth[6]))

    mar = (A + B) / (2.0 * C)

    return mar


def head_tilt(nose, chin):

    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]

    angle = np.degrees(np.arctan2(dy, dx))

    return angle