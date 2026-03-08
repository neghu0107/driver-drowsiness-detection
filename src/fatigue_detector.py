import csv
import os

DATA_PATH = "data/fatigue_training.csv"

def init_dataset():

    if not os.path.exists(DATA_PATH):

        with open(DATA_PATH, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "EAR",
                "MAR",
                "BlinkRate",
                "HeadTilt",
                "SleepDuration",
                "Fatigue"
            ])


def log_training_data(ear, mar, blink_rate, head_tilt, sleep_duration, fatigue):

    with open(DATA_PATH, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            ear,
            mar,
            blink_rate,
            head_tilt,
            sleep_duration,
            fatigue
        ])