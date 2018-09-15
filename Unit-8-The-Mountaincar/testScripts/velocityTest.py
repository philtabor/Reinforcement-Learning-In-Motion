import numpy as np
import math
max_speed = 0.07
min_position = -1.2
max_position = 0.6
velocity = 0.054
position = 0.43
#velocity = 0.001
#position = -1.1
steps = 0
action = 2
while position < 0.5:
    velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
    velocity = np.clip(velocity, -max_speed, max_speed)
    position += velocity
    position = np.clip(position, min_position, max_position)

    steps += 1
print(steps)