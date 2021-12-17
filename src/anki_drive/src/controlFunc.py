from overdrive import Overdrive
import time

def lateral_CL(car_obj, delta_d, tol):
    "Take in car object, desired delta from center delta_d, and tolerance tol and drive to lateral position"
    #while loop to drive to position
    e = delta_d-car_obj.delta #define error
    kp = 1 #proportional control constant
    while abs(e)>tol:
        #define control input for this state, to be input to change_lane
        pos_input = kp*e
        car_obj.set_offset(1000, 1000, pos_input) #set continous offset with max speed and accel
        e = car_obj.delta-delta_d