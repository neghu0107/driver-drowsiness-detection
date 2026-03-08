import pandas as pd
import random

def simulate_movement(drivers):

    drivers["latitude"] += [
        random.uniform(-0.002,0.002) for _ in range(len(drivers))
    ]

    drivers["longitude"] += [
        random.uniform(-0.002,0.002) for _ in range(len(drivers))
    ]

    return drivers
