import random

drivers = {
    "Driver 1": {"lat": 13.0827, "lon": 80.2707},
    "Driver 2": {"lat": 13.05, "lon": 80.22},
    "Driver 3": {"lat": 13.07, "lon": 80.24},
    "Driver 4": {"lat": 13.09, "lon": 80.26},
}

def move_vehicle(lat, lon):

    lat += random.uniform(-0.0005, 0.0005)
    lon += random.uniform(-0.0005, 0.0005)

    return lat, lon