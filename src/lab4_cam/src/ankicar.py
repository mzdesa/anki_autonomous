#!/usr/bin/env python

from overdrive import Overdrive
import numpy as np
from numpy.lib.npyio import save
from mapping_track_to_model import *

class Ankicar:
    def __init__(self, addr):
        self.CON_SPEED, self.ACCEL = 1000, 1000
        self.width, self.length = 4.5, 9 #cm
        self.car = Overdrive(addr)
        self.curr_xy = np.array([4, -50]) #xy coords
        self.curr_vxvy = np.array([0, 31]) #keep direction in mind
        self.prev_xy = np.array([4, -56])
        self.theta_car = 0

        self.time_step = 0.25
        
        #System ID
        self.A = []
        self.B = []
        self.getCarPos = GetCarState() #initialize to be used later

        #Store state history for real time smoothing
        self.state_data = np.zeros([5, 1]) #initialize as a 5x1 array of zeros

        #Smoothed state
        self.state = np.zeros((5, 1))
        self.prev_state = np.zeros((5, 1))
        self.curr_state = np.array([]) #SMOOTHED STATE VECTOR

        self.s = None #s parameter - get from state vector
        self.delta = None #n parameter

    def reset(self):
        self.s, self.delta = self.get_position()
        self.speed = self.car.speed
        return

    def disconnect(self):
        self.car._disconnect()
        print("Disconnected")
        # self.car.changeLane(self.CON_SPEED, self.ACCEL, amt)
        return

    def set_speed(self, speed):
        self.car.changeSpeed(speed, self.ACCEL)

    def set_offset(self, offset):
        self.car.changeLane(1000, 1000, offset)

    #apply system ID for car
    def get_system_ID(self):
        """
        Update A and B matrices after data collection
        """
        self.A = np.load("savedAMatrix2.npy") #load the saved A and B filed
        self.B = np.load("savedBMatrix2.npy") #must run from project root directory

    def get_position(self, pointFromPixel, last_image_service):
        #function to get the real time unsmoothed position and add it to state data
        getCarPos = self.getCarPos
        try:
            #GET CAR STATE
            ros_img_msg = last_image_service().image_data
            getCarPos.mask_car(
                ros_img_msg, landmarks=pointFromPixel.points)
            car_pos_xy = pointFromPixel.pixel_to_track_cartesian(np.array([getCarPos.blue_y, getCarPos.blue_x]))
            car_pos_sn_raw = coord_solver(car_pos_xy) #transform to sn coordinates
            #print("XY coordinates: ", car_pos_xy)
            #print("Modded SN coordinates: ", car_pos_sn_raw)
            s_curr = car_pos_sn_raw[0]
            n_curr = car_pos_sn_raw[1]

            def s_dot_solver(s_curr, s_prev):
                if s_curr-s_prev < -100: #define -100 as a threshold value
                    return (s_curr+226.097-s_prev)/self.time_step
                return (s_curr-s_prev)/self.time_step                         
            n_dot_solver = lambda n_curr, n_prev: (n_curr-n_prev)/self.time_step

            #solve for necessary variables
            s_dot = s_dot_solver(s_curr, self.prev_state[0])
            n_dot = n_dot_solver(n_curr, self.prev_state[1])
            phi = np.arctan2(n_dot, s_dot)
            #redefine current state, previous state, and state_data history
            self.prev_state = self.state
            self.state = np.array([s_curr, n_curr, s_dot, n_dot, phi]).reshape((5,1))
            #print("Current State: ", self.state)
            self.state_data = np.hstack((self.state_data, self.state))
        except rospy.ServiceException as e:
            print("image_process: Service call failed: %s" % e)

    def get_smoothed_position(self, pointFromPixel, last_image_service):
        #function that should be run at all times to update position
        #uses real time exponential smoothing
        alpha = 0.5
        self.get_position(pointFromPixel, last_image_service)
        raw_data = self.state_data #load vector of state data
        smoothed_data_exp = raw_data
        #print(raw_data.shape)
        for row_index in [2, 3]:
            for t_step in range(raw_data.shape[1]):
                if t_step == 0:
                    smoothed_data_exp[row_index, t_step] = raw_data[row_index, t_step]
                else:
                    smoothed_data_exp[row_index, t_step] = (1-alpha)*smoothed_data_exp[row_index, t_step-1] + (alpha)*raw_data[row_index, t_step]
        smoothed_data_exp[4, :] = np.arctan2(smoothed_data_exp[3, :], smoothed_data_exp[2, :])
        #current smoothed state is the final value of the total smoothed array
        self.curr_state = smoothed_data_exp[:, -1]
        self.s = self.curr_state[0]
        self.delta = self.curr_state[1]
        #print(self.curr_state[0])

if __name__ == '__main__':
    car = Ankicar("C9:08:8F:21:77:E3")
    #car.get_system_ID()
    #print(car.A)
    # Waits for the image service to become available
    rospy.wait_for_service('last_color_image')
    # rospy.wait_for_service('last_depth')
    rospy.init_node('car_mask', anonymous=True, disable_signals=True)

    pub = rospy.Publisher('/car_position', CarPosition, queue_size=10)

    last_image_service = rospy.ServiceProxy('last_color_image', ImageSrv)
    # last_depth_service = rospy.ServiceProxy('last_depth', ImageSrv)
    #x_arr, y_arr, speed_array, offset_array
    pointFromPixel = PointFromPixel()

    while not rospy.is_shutdown():
        try:
            ros_img_msg = last_image_service().image_data
            pointFromPixel.get_6_points(ros_img_msg)
            flag = True
            break

        except KeyboardInterrupt:
            print('Keyboard Interrupt, exiting')
            break

        # Catch if anything went wrong with the Image Service
        except rospy.ServiceException as e:
            print("image_process: Service call failed: %s" % e)

    if flag:
        while not rospy.is_shutdown():
            try:
                getCarPos = GetCarState()
                ros_img_msg = last_image_service().image_data
                car.set_speed(300) #set car speed
                car.get_smoothed_position(pointFromPixel, last_image_service)

            except KeyboardInterrupt:
                print('Keyboard Interrupt, exiting')
                break

            # Catch if anything went wrong with the Image Service
            except rospy.ServiceException as e:
                print("image_process: Service call failed: %s" % e)

