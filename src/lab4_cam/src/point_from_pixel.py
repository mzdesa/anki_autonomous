#!/usr/bin/env python

from overdrive import Overdrive
import rospy
import cv2
from cv_bridge import CvBridge
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point, PointStamped, Vector3Stamped, TransformStamped
import numpy as np
import tf.transformations as tr
from image_geometry import PinholeCameraModel
from lab4_cam.srv import ImageSrv, ImageSrvResponse
from PIL import Image as img
import matplotlib.pyplot as plt
from scipy import optimize
# from ankicar import Ankicar
from curviSolver import coord_solver
import time

class PointFromPixel():
    """ Given a pixel location, find its 3D location in the world """
    def __init__(self):
        self.need_camera_info = True
        self.need_depth_image = True
        self.depth = None
        self.model = PinholeCameraModel()
        self.bridge = CvBridge()

    def callback_camera_info(self, info):
        """ Define Pinhole Camera Model parameters using camera info msg """
        if self.need_camera_info:
            # rospy.loginfo('Got camera info!')
            self.model.fromCameraInfo(info)  # define model params
            self.frame = info.header.frame_id
            # self.need_camera_info = False

    def callback_depth_image(self, depth_image):
        """ Get depth at chosen pixel using depth image """
        if self.need_depth_image:
            # rospy.loginfo('Got depth image!')
            # rospy.loginfo(depth_image)
            # self.depth = img.frombytes("I", (depth_image.width, depth_image.height), depth_image.data)
            cv_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='32FC1')
            # Convert the depth image to a Numpy array since most cv2 functions
            # require Numpy arrays.
            cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
            # Normalize the depth image to fall between 0 (black) and 1 (white)
            # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
            cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
            # Resize to the desired size
            # cv_image_resized = cv2.resize(cv_image_norm, (depth_image.width, depth_image.height), interpolation = cv2.INTER_CUBIC)
            self.depth = cv_image_array
            # plt.imshow(self.depth)
            # plt.show()
            # rospy.loginfo(cv_image_norm.shape)

            # cv2.imshow("Image from my node", self.depth)
            # cv2.waitKey(1)
            # self.need_depth_image = False

    def calculate_3d_point(self, pixel):
        """ Project ray through chosen pixel, then use pixel depth to get 3d point """
        depth = self.depth[pixel[1], pixel[0]]  # lookup pixel in depth image
        # print(depth)
        # import pdb; pdb.set_trace()
        ray = self.model.projectPixelTo3dRay(tuple(pixel))  # get 3d ray of unit length through desired pixel
        #ray_z = [el / ray[2] for el in ray]  # normalize the ray so its Z-component equals 1.0
        pt = [el * depth for el in ray]  # multiply the ray by the depth; its Z-component should now equal the depth value
        # point = PointStamped()
        # point.header.frame_id = self.frame
        # point.point.x = pt[0]
        # point.point.y = pt[1]
        # point.point.z = pt[2]
        # print([pt[0], pt[1], pt[2]])
        return pt

class GetCarState():
    def __init__(self):
        self.blue_x = None
        self.blue_y = None
        self.green_x = None
        self.green_y = None
        self.prev_img = None
        self.bridge = CvBridge()

    def mask_car(self, ros_image, blue=True):
        # rospy.loginfo('Got color image!')
        img = self.ros_to_np_img(ros_image)
        img_height = img.shape[0]
        img_width = img.shape[1]
        # rospy.loginfo(img.shape)
        mask = np.zeros(img.shape)
        if blue:
            mask[img[:,:,0] > 200] = 1
            mask[img[:,:,2] > 150] = 0  #breaks around 190, starts to be visible at 50
        else:
            mask[img[:,:,1] > 200] = 1
            mask[img[:,:,2] > 180] = 0  #90 - 180
            mask[img[:,:,0] > 220] = 0
            # plt.imshow(mask)
            # plt.show()
        mask = np.mean(mask, axis=2)
        bbox = self.get_loop_box(mask)
        if len(bbox) == 0:
            self.blue_x = -1
            self.blue_y = -1
            return

        mask2 = np.zeros((img.shape[0], img.shape[1]))
        mask2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1.0
        mask2 = np.stack((mask2, mask2, mask2), axis=2)
        mask2 = mask2.astype(np.uint8)
        car_mask = np.multiply(img, mask2)
        center = (0, 0)
        if blue:
          self.blue_y = 479 if int(bbox[1] + bbox[3]/2) >= 480 else int(bbox[1] + bbox[3]/2)
          self.blue_x = 639 if int(bbox[0] + bbox[2]/2) >= 640 else int(bbox[0] + bbox[2]/2)
          center = (self.blue_x, self.blue_y)
        else:
          self.green_y = 479 if int(bbox[1] + bbox[3]/2) >= 480 else int(bbox[1] + bbox[3]/2)
          self.green_x = 639 if int(bbox[0] + bbox[2]/2) >= 640 else int(bbox[0] + bbox[2]/2)
          center = (self.green_x, self.green_y)
        img = cv2.circle(img, center, 10, (255,0,0), 5)
        cv2.imshow("Image from my node", img)
        cv2.waitKey(1)

    def ros_to_np_img(self, ros_img_msg):
        return np.array(self.bridge.imgmsg_to_cv2(ros_img_msg,'bgr8'))

    def get_loop_box(self, mask, largest=1):
        bboxes = []
        while True:
          mask = mask.astype(np.uint8)
          n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
          if n_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            bbox = stats[largest_label, :]
            bboxes.append(bbox)
          else:
            break

          mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0

          if bbox[cv2.CC_STAT_AREA] < 200:
            break
        
        if len(bboxes) == 0:
            return bboxes
        if largest == 1:
          return bboxes[0]
        return bboxes[:largest]

    def vel_solver(self, prev_state, dt):
      """
      Function to solve for the orientation angle psi of the car (Needed for Jun's code)
      Takes in the state of the car at the previous time step

      Inputs;
      prev_state: [x_prev, y_prev]
      dt: position sampling time step of CV code
      """
      return [(self.x-prev_state[0])/dt, (self.y-prev-state[0])/dt]

    def coord_solver(self, car, ar_state1, car_state, r, straight_len):
        """
        Solve for absolute s, n coordinates of car object
        car - overdrive object
        ar_state1- (x, y, z) of ar tag at the start of the 1st turn
        ar_state2- (x, y, z) of ar tag at the start of the 2nd turn
        car_state - (x, y, z) of car
        r - track turn radius
        straight_len - length of straight section
        """
        piece = 0 # later pass this in
        x = car_state[0]
        y = car_state[1]
        if piece == 0: #car.piece == 0:
            print('straight')
            s = (y - ar_state1[1])+straight_len + (straight_len + 2*np.pi*r) #have offset values to avoid negative
            n = x-ar_state1[0]
        elif piece == 1 or piece == 2:                       
            print("turn")
            n = np.sqrt((x + r)**2 + y**2) - r
            arr1 = np.array([x+r, y])
            arr2 = np.array([r, 0])
            theta = np.arccos(np.dot(arr1, arr2) / (r*np.linalg.norm(arr1)))
            s = r * theta
        elif piece == 3:
            print('straight')
            s = abs(y) + np.pi*r
            n = -(x + 2*r)
        elif piece == 4 or piece == 5:
            print('turn')
            n = np.sqrt((x + r)**2 + (y + straight_len)**2) - r
            arr1 = np.array([x+r, y+straight_len])
            arr2 = np.array([-r, 0])
            theta = np.arccos(np.dot(arr1, arr2) / (np.linalg.norm(arr2)*np.linalg.norm(arr1)))
            s = r * theta + straight_len + np.pi*r
        return s, n


if __name__ == '__main__':
    rospy.init_node('car_mask', anonymous=True)
    point_from_pixel = PointFromPixel()
    getCarPos = GetCarState()

    # rospy.Subscriber("/camera/depth/camera_info", CameraInfo, point_from_pixel.callback_camera_info)

    # Waits for the image service to become available
    rospy.wait_for_service('/last_color_image')
    # rospy.wait_for_service('/last_depth_image')

    # Creates a function used to call the image capture service: ImageSrv is the service type
    last_image_service = rospy.ServiceProxy('/last_color_image', ImageSrv)
    # last_depth_service = rospy.ServiceProxy('/last_depth_image', ImageSrv)

    #Connect to the realsense camera and get the transform to the track frame
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)

    def get_artag_gmatrix(msg):
        p = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        q = np.array([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
        q_norm = np.linalg.norm(q)
        if np.abs(q_norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                    str(q), np.linalg.norm(q)))
        elif np.abs(q_norm - 1.0) > 1e-6:
            q = q / q_norm
        g = np.matrix(tr.quaternion_matrix(q))
        p = np.expand_dims(p, axis=1)
        g[0:3, -1] = p
        return g

    def get_artag_pos(msg):
        p = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z, 1])
        return p

    def sysID():
        #solve for A and B matrices of system
        #get the transform to the track from the cam
        step = 0 #counter variable for iterations
        sample_period = 0.1 #position sampling period for each loop
        total_steps = 1000 #total number of seconds for sampling
        record_states = np.zeros((2, total_steps)) #Matrix that records XY of car
        sine_freq = np.pi/2 #frequency for sine oscillation
        amp = 44.5*2 #amplitude in overdrive units
        #initialize overdrive object
        car = Overdrive("XX:XX:XX:XX:XX") #overdrive object and address
        while not rospy.is_shutdown() and step<total_steps:
            #collect image data
            t = step*sample_period
            car.changeLane(1000, 1000, amp*np.sin(sine_freq*t)) #move car in a sine function
            step += 1#increment step

            #CAR TRACKING
            ros_img_msg = last_image_service().image_data
            ros_depth_msg = last_depth_service().image_data
            point_from_pixel.callback_depth_image(ros_depth_msg)

            #Blue Car
            car_pos = getCarPos.mask_car(ros_img_msg) #TODO are these the right coords?
            if getCarPos.blue_x == -1 and getCarPos.blue_y == -1:
                continue

            #Get regression data for car
            #update position in regression array
            record_states[:, step] = car_pos #write position into array at appropriate location
            time.sleep(sample_period) #have a sleep rate of sample period to have regular time intervals for data

        #JUN REGRESSION
        A, B = regression_func(record_states)
        


    while not rospy.is_shutdown():
        #Blue Car
        getCarPos.mask_car(ros_img_msg)
        if getCarPos.blue_x == -1 and getCarPos.blue_y == -1:
          continue

