import random

drivers = {
    "Driver1": [13.0827, 80.2707],
    "Driver2": [13.0527, 80.2507],
    "Driver3": [13.1027, 80.2907],
    "Driver4": [13.0727, 80.2407],
}

def move_vehicle():

    for d in drivers:

        drivers[d][0] += random.uniform(-0.001,0.001)
        drivers[d][1] += random.uniform(-0.001,0.001)

    return drivers