#!/usr/bin/env python

from functools import partial
from numpy.core.numeric import ones
from obstacle import Obstacle
import rospy
import tf2_ros
import sys
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import image_geometry
from PIL import Image as img
# import pyrealsense2 as rs
from lab4_cam.srv import ImageSrv, ImageSrvResponse
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans

from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Twist
from lab4_cam.msg import CarPosition
from overdrive import Overdrive

import datetime  # get datetime for veloicty measurements
import time
from scipy.signal.signaltools import wiener

from matplotlib import pyplot as plt

from PIL import Image

bridge = CvBridge()
# depth_cam_info = None
cam_model = None
# X = None
# Y = None
world_coords = None


class PointFromPixel():
    def __init__(self):
        self.scale_x = 1  # transform from pixel unit to cm
        self.scale_y = 1
        self.origin = None  # centered at point0
        self.axis_x = None  # unit x-axis expressed in pixel coordinates
        self.axis_y = None  # unit y-axis expressed in pixel coordinates
        self.list_of_detected_points = []

    def get_6_points(self, img):
        img = self.ros_to_np_img(img)
        img_height = img.shape[0]
        img_width = img.shape[1]
        # rospy.loginfo(img.shape) # (720, 1280, 3)
        mask = np.zeros(img.shape)

        mask[img[:, :, 0] > 160] = 1
        mask[img[:, :, 2] > 120] = 0
        # mask[img[:,:,1] > 120] = 0

        #   3,    2
        # 4 p35 C p20  1
        #   5,    0
        # d50 = 56.5 cm
        # d02 = 36 cm
        # d14 = 92.5 cm
        mask = np.mean(mask, axis=2)  # (720, 1280)

        self.list_of_detected_points = []
        for i in range(img_height):
            for j in range(img_width):
                if mask[i, j] == 1:
                    self.list_of_detected_points.append([i, j])
        X = np.array(self.list_of_detected_points)
        kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
        points_center_array = kmeans.cluster_centers_  # (6, 2) array

        self.point4 = points_center_array[np.argmin(
            points_center_array[:, 1]), :]
        self.point1 = points_center_array[np.argmax(
            points_center_array[:, 1]), :]
        left_points = points_center_array[points_center_array[:, 1] < img_width/2]
        right_points = points_center_array[points_center_array[:, 1] > img_width/2]
        plt.imshow(mask)
        plt.show()
        assert left_points.shape[0] == 3, str(left_points.shape[0])
        assert right_points.shape[0] == 3, str(right_points.shape[0])
        self.point3 = left_points[np.argmin(left_points[:, 0]), :]
        self.point2 = right_points[np.argmin(right_points[:, 0]), :]
        self.point5 = left_points[np.argmax(left_points[:, 0]), :]
        self.point0 = right_points[np.argmax(right_points[:, 0]), :]
        self.points = [self.point0, self.point1, self.point2,
                       self.point3, self.point4, self.point5]

        point0 = self.point0
        point1 = self.point1
        point2 = self.point2
        point3 = self.point3
        point4 = self.point4
        point5 = self.point5

        print(point0)
        print(point1)
        print(point2)
        print(point3)
        print(point4)
        print(point5)

        plt.imshow(mask)
        plt.show()

        p35 = (point3 + point5) / 2
        p20 = (point2 + point0) / 2
        radius = np.linalg.norm(p35 - point5)
        assert np.linalg.norm(
            p35 - point4) - radius < 20, str(np.linalg.norm(p35 - point4) - radius)
        assert np.linalg.norm(
            p20 - point1) - radius < 20, str(np.linalg.norm(p20 - point1) - radius)
        assert np.linalg.norm(
            p20 - point0) - radius < 20, str(np.linalg.norm(p20 - point0) - radius)

        print((np.arctan((point1[1] - point4[1])/(point1[0] - point4[0]))))
        print((np.arctan((point2[1] - point3[1])/(point2[0] - point3[0]))))
        assert np.abs(np.arctan((point1[1] - point4[1])/(point1[0] - point4[0]))) - np.abs(
            np.arctan((point2[1] - point3[1])/(point2[0] - point3[0]))) < 0.1

        d02 = 36
        self.scale_x = d02 / np.linalg.norm(point0 - point2)
        d14 = 92.5
        self.scale_y = d14 / np.linalg.norm(point1 - point4)
        self.axis_y = (point1 - point4) / np.linalg.norm(point1 - point4)
        self.axis_x = np.array([self.axis_y[1], -self.axis_y[0]])
        self.origin = point0
        print(self.axis_y)
        print(self.axis_x)
        print(self.origin)

    def pixel_to_track_cartesian(self, point):
        point_hom_cam = np.append(point, np.array(1))
        scaling = np.array([[self.scale_x, 0, 0],
                            [0, self.scale_y, 0],
                            [0, 0, 1]])
        g_cam_track = np.array([[self.axis_x[0], self.axis_y[0], self.origin[0]],
                                [self.axis_x[1], self.axis_y[1], self.origin[1]],
                                [0, 0, 1]])
        point_hom_track = scaling.dot(
            np.linalg.inv(g_cam_track)).dot(point_hom_cam)
        # print(point_hom_track)
        return point_hom_track[:2]

    def ros_to_np_img(self, ros_img_msg):
        return np.array(bridge.imgmsg_to_cv2(ros_img_msg, 'bgr8'))


class GetCarState():

    def __init__(self):
        self.blue_x = None
        self.blue_y = None
        self.blue_x_history = np.array([])
        self.blue_y_history = np.array([])
        # self.green_x = None
        # self.green_y = None
        self.prev_img = None
        self.blue_x_prev = None
        self.blue_y_prev = None
        self.bridge = CvBridge()

        #List of obstacle objects
        self.obstacles = []
        
        # a list of numbers
        #self.red_obstacle_x = None
        #self.red_obstacle_y = None
        # a list of 4x2 arrays containing the XY global fr corners of an obstacle
        #self.obstacle_corners = np.array([])
        #self.obstacle_corners_pix = np.array([])
        # a list of (2,) numpy array (x, y) of obstacle center in cm
        #self.obstacle_xy_cm = np.array([])
        #self.obstacleL = None  # obstacle length
        #self.obstacleW = None  # obstacle width

    def mask_car(self, ros_image, landmarks, blue=True):
        """landmarks is a list of points representing the blue markers for the track.
        All of those pixels will become black before we try to locate the blue car.
        """
        # rospy.loginfo('Got color image!')
        img = self.ros_to_np_img(ros_image)
        img_height = img.shape[0]
        img_width = img.shape[1]

        # print(self.blue_x)
        # print(self.blue_y)
        # rospy.loginfo(img.shape)
        self.blue_x_prev = self.blue_x
        self.blue_y_prev = self.blue_y  # Keep in temporary variables in case state is lost

        for i in range(6):
            if i == 4 or i == 1:
                img[int(landmarks[i][0])-25:int(landmarks[i][0])+25,
                    int(landmarks[i][1])-15:int(landmarks[i][1])+15, :] = [0, 0, 0]
            else:
                img[int(landmarks[i][0])-15:int(landmarks[i][0])+15,
                    int(landmarks[i][1])-25:int(landmarks[i][1])+25, :] = [0, 0, 0]

        mask = np.zeros(img.shape)
        if blue:
            mask[img[:, :, 0] > 200] = 1
            # breaks around 190, starts to be visible at 50
            mask[img[:, :, 2] > 150] = 0
        # else:
        #     mask[img[:, :, 1] > 200] = 1
        #     mask[img[:, :, 2] > 180] = 0  # 90 - 180
        #     mask[img[:, :, 0] > 220] = 0
            # plt.imshow(mask)
            # plt.show()
        mask = np.mean(mask, axis=2)
        bbox = self.get_loop_box(mask)
        if len(bbox) == 0:
            print("no car detected")
            self.blue_x = self.blue_x_prev
            self.blue_y = self.blue_y_prev
            # self.blue_x = -1
            # self.blue_y = -1
            # print(self.blue_x)
            # print(self.blue_y)
            # return
        else:
            # mask2 = np.zeros((img.shape[0], img.shape[1]))
            # mask2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1.0
            # mask2 = np.stack((mask2, mask2, mask2), axis=2)
            # mask2 = mask2.astype(np.uint8)
            # car_mask = np.multiply(img, mask2)
            center = (0, 0)
            if blue:
                self.blue_y = 719 if int(
                    bbox[1] + bbox[3]/2) >= 720 else int(bbox[1] + bbox[3]/2)
                self.blue_x = 1279 if int(
                    bbox[0] + bbox[2]/2) >= 1280 else int(bbox[0] + bbox[2]/2)
                center = (self.blue_x, self.blue_y)
            # else:
            #     self.green_y = 719 if int(
            #         bbox[1] + bbox[3]/2) >= 720 else int(bbox[1] + bbox[3]/2)
            #     self.green_x = 1279 if int(
            #         bbox[0] + bbox[2]/2) >= 1280 else int(bbox[0] + bbox[2]/2)
            #     center = (self.green_x, self.green_y)
            img = cv2.circle(img, center, 10, (255, 0, 0), 5)
            # cv2.imshow("Blue Car", img)
            # cv2.waitKey(1)
        self.blue_x_history = np.append(self.blue_x_history, self.blue_x)
        self.blue_y_history = np.append(self.blue_y_history, self.blue_y)

    def mask_red_obstacle(self, ros_image, pointFromPixel):
        
        # rospy.loginfo('Got color image!')
        img = self.ros_to_np_img(ros_image)
        img_height = img.shape[0]
        img_width = img.shape[1]

        img_masked = np.copy(img)

        # re-detect the obstacles at each timestep
        self.obstacles = []
        obstacle_detected = True
        while obstacle_detected:
            #print(3)
            mask = np.zeros(img.shape)
            mask[img_masked[:, :, 2] > 170] = 1
            # plt.imshow(img_masked)
            # plt.show()
            mask[img_masked[:, :, 0] > 150] = 0
            # mask[img[:, :, 0] > 150] = 0
            # plt.imshow(mask)
            # plt.show()
            mask = np.mean(mask, axis=2)
            bbox = self.get_loop_box(mask, largest=1, threshold=1500) # the threshold for obstacle is 1500
            # print(bbox)
            # plt.imshow(mask)
            # plt.show()
            if len(bbox) == 0:
                if len(self.obstacles) >= 1:
                    return True
                else:
                    print("no obstacle detected")
                    return False
            else:
                # create an Obstacle object from the bounding box and mask the img_masked
                # mask2 = np.zeros((img.shape[0], img.shape[1]))
                # mask2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1.0
                # mask2 = np.stack((mask2, mask2, mask2), axis=2)
                # mask2 = mask2.astype(np.uint8)
                # plt.imshow(mask2)
                # plt.show()
                obstacle_corners_pix, obstacle_corners = self.getContours(
                    img, bbox, mask, pointFromPixel) # passing in the img is just for its shape
                

                red_obstacle_bb_center_y = 719 if int(
                    bbox[1] + bbox[3]/2) >= 720 else int(bbox[1] + bbox[3]/2)
                red_obstacle_bb_center_x = 1279 if int(
                    bbox[0] + bbox[2]/2) >= 1280 else int(bbox[0] + bbox[2]/2)
                
                # plot the center of the bounding box
                # center = (red_obstacle_bb_center_x, red_obstacle_bb_center_y)
                # img = cv2.circle(img, center, 10, (255, 0, 0), 5)
                for corner in obstacle_corners_pix:
                    corner = tuple(corner)
                    corner = corner[1], corner[0]
                    img = cv2.circle(img, corner, 2, (255, 0, 0), 5)
                    img_masked = cv2.circle(img_masked, corner, 2, (255, 0, 0), 5)
                # plot the bounding box
                # img = cv2.rectangle(
                #     img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
                cv2.imshow("Red Obstacle", img)
                cv2.waitKey(1)
                # plt.imshow(np.array(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)))
                # plt.show()

                obs_obj = Obstacle(0, 0, 0, 0) #create obstacle object
                obs_obj.red_obstacle_bb_center_x = red_obstacle_bb_center_x
                obs_obj.red_obstacle_bb_center_y = red_obstacle_bb_center_y
                obs_obj.obstacle_corners = obstacle_corners
                obs_obj.obstacle_corners_pix = obstacle_corners_pix

                #add to Self
                self.obstacles.append(obs_obj)
                print("mapping_track_to_model", self.obstacles)
                # print(int(np.min(obstacle_corners_pix[:, 0])))
                # print(int(np.max(obstacle_corners_pix[:, 0])))
                # print(int(np.min(obstacle_corners_pix[:, 1])))
                # print(int(np.max(obstacle_corners_pix[:, 1])))
                # mask the img_masked
                img_masked[int(np.min(obstacle_corners_pix[:, 0])):int(np.max(obstacle_corners_pix[:, 0])),
                    int(np.min(obstacle_corners_pix[:, 1])):int(np.max(obstacle_corners_pix[:, 1])), :] = [0, 0, 0]
                # plt.imshow(img_masked)
                # plt.show()

                # img_masked = np.multiply(img_masked, mask2)

        return True

    def ros_to_np_img(self, ros_img_msg):
        return np.array(self.bridge.imgmsg_to_cv2(ros_img_msg, 'bgr8'))

    def get_loop_box(self, mask, largest=1, threshold=50):
        #threshold specifies area of bounding box (200 for car, 800 for obstacle)
        bboxes = []
        while True:
            mask = mask.astype(np.uint8)
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=4)
            if n_labels > 1:
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                bbox = stats[largest_label, :]
                bboxes.append(bbox)
                # print(bboxes)
            else:
                break

            mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0

            if bbox[cv2.CC_STAT_AREA] < threshold:
                break
        
        if len(bboxes) == 0:
            return bboxes
        if bboxes[0][4] < threshold:
            return []
        
        if largest == 1:
            #print(bboxes)
            return bboxes[0]
        return bboxes[:largest]

    def getContours(self, img, bbox, mask, pointFromPixel):
        # print("CONTOUR")
        box_offset = 40
        center = [bbox[1]+int(bbox[3]/2), bbox[0]+int(bbox[2])/2]
        box_offset_left = bbox[0] if bbox[0] < box_offset else box_offset
        box_offset_right = img.shape[1]-bbox[0] if bbox[0] + \
            box_offset > img.shape[1] else box_offset
        box_offset_top = bbox[1] if bbox[1] < box_offset else box_offset
        box_offset_bottom = img.shape[0]-bbox[1] if bbox[1] + \
            box_offset > img.shape[0] else box_offset
        partial_img = mask[center[0]-box_offset_top:center[0]+box_offset_bottom,
                           center[1]-box_offset_left:center[1]+box_offset_right]
        obstacle_points_list = []

        for i in range(partial_img.shape[0]):
            for j in range(partial_img.shape[1]):
                if partial_img[i, j] == 1:
                    obstacle_points_list.append([i, j])

        dist_list_topleft = map(
            lambda x: x[0]**2+x[1]**2, obstacle_points_list)
        dist_list_topright = map(
            lambda x: x[0]**2+(partial_img.shape[1]-x[1])**2, obstacle_points_list)
        dist_list_bottomleft = map(lambda x: (
            partial_img.shape[0]-x[0])**2+x[1]**2, obstacle_points_list)
        dist_list_bottomright = map(lambda x: (
            partial_img.shape[0]-x[0])**2+(partial_img.shape[1]-x[1])**2, obstacle_points_list)

        obstacle_points_list = np.array(obstacle_points_list)
        topleft_rel_coord = obstacle_points_list[np.argsort(
            dist_list_topleft)[:20]].mean(axis=0).astype(int)
        topright_rel_coord = obstacle_points_list[np.argsort(
            dist_list_topright)[:20]].mean(axis=0).astype(int)
        bottomleft_rel_coord = obstacle_points_list[np.argsort(
            dist_list_bottomleft)[:20]].mean(axis=0).astype(int)
        bottomright_rel_coord = obstacle_points_list[np.argsort(
            dist_list_bottomright)[:20]].mean(axis=0).astype(int)

        # print(obstacle_points_list[np.argsort(dist_list_topleft)[:20]])
        # print(topleft_rel_coord)
        # print(topright_rel_coord)
        # print(bottomleft_rel_coord)
        # print(bottomright_rel_coord)

        plotting = False  # option to plot image with extrema
        if plotting:
            partial_img[topleft_rel_coord[0], topleft_rel_coord[1]] = 0.5
            partial_img[topright_rel_coord[0], topright_rel_coord[1]] = 0.5
            partial_img[bottomleft_rel_coord[0],
                        bottomleft_rel_coord[1]] = 0.5
            partial_img[bottomright_rel_coord[0],
                        bottomright_rel_coord[1]] = 0.5

            plt.imshow(partial_img, cmap='gray')
            plt.show()
        # Convert to XY coordinates
        # convert to global image coords
        offset = np.array(
            [center[0]-box_offset_top, center[1]-box_offset_left])
        # print(offset)
        edge_arr = np.vstack((topleft_rel_coord, topright_rel_coord,
                             bottomleft_rel_coord, bottomright_rel_coord)) + offset
        # obstacle_corners_pix is a (4, 2) array
        obstacle_corners_pix = np.copy(edge_arr)
        # print(self.obstacle_corners_pix)
        for i in range(4):
            # edge_arr[i, :] = edge_arr[i, :] + offset #account for offset
            edge_arr[i, :] = pointFromPixel.pixel_to_track_cartesian(
                edge_arr[i, :])
        obstacle_corners = edge_arr
        # print(self.obstacle_corners)
        return obstacle_corners_pix, obstacle_corners

    def get_lw_xy(self, ros_image, pointFromPixel):
        """
        Function to get the length and width & XY coordinates of obstacle center
        """
        obstacle_detected = self.mask_red_obstacle(ros_image, pointFromPixel)  # call mask function to update position
        if obstacle_detected:
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                # solve for length and width - take the average of the two possibilities
                # map the norm over the two lengths
                l_arr = map(np.linalg.norm, np.array(
                    [obstacle.obstacle_corners[0, :]-obstacle.obstacle_corners[1, :], obstacle.obstacle_corners[2, :]-obstacle.obstacle_corners[3, :]]))
                self.obstacles[i].obstacleL = np.mean(l_arr)
                self.obstacles[i].length = self.obstacles[i].obstacleL
                w_arr = map(np.linalg.norm, np.array(
                    [obstacle.obstacle_corners[0, :]-obstacle.obstacle_corners[2, :], obstacle.obstacle_corners[1, :]-obstacle.obstacle_corners[3, :]]))
                self.obstacles[i].obstacleW = np.mean(w_arr)
                self.obstacles[i].width = self.obstacles[i].obstacleW
                # get the center coordinates
                x_coord = np.mean([obstacle.obstacle_corners[0, 0], obstacle.obstacle_corners[1, 0],
                                obstacle.obstacle_corners[2, 0], obstacle.obstacle_corners[3, 0]])
                y_coord = np.mean([obstacle.obstacle_corners[0, 1], obstacle.obstacle_corners[1, 1],
                                obstacle.obstacle_corners[2, 1], obstacle.obstacle_corners[3, 1]])
                self.obstacles[i].obstacle_xy_cm = np.array([x_coord, y_coord])
                self.obstacles[i].x = x_coord
                self.obstacles[i].y = y_coord
                self.obstacles[i] = Obstacle(x_coord, y_coord, np.mean(w_arr), np.mean(l_arr))
                # print(obstacle.obstacle_xy_cm)
            return True
        else:
            return False


def get_track_piece(coords):
    """
    Function to get the track piece corresponding to an input coordinate
    Get track piece based on closest calibration point and coords relative to the point (forward, behind, ...)
    """
    #   3,    2
    # 4 p35 C p20  1
    #   5,    0
    #     PIECE 2
    # PIECE 3 C PIECE 1
    #     PIECE 0
    # d50 = 56.5 cm
    # d02 = 36 cm
    # d14 = 92.5 cm
    if coords[1] < 0 and coords[1] > -56.5 and coords[0] > 0:
        return 0  # piece 0
    elif coords[1] > 0:
        return 1
    elif coords[1] < 0 and coords[1] > -56.5:
        return 2
    else:
        return 3


def coord_solver(coords):
    """
    Transform coords from (x, y) to (s, n)
    """
    r = 18  # inner radius in cm
    straight_len = 56.5  # length of a straight section
    x = coords[0]
    y = coords[1]
    piece = get_track_piece(coords)
    if piece == 0:  # car.piece == 0:
        # print('straight')
        # have offset values to avoid negative
        s = y+straight_len + (straight_len + 2*np.pi*r)
        n = x
    elif piece == 1:
        # print("turn")
        n = np.sqrt((x + r)**2 + y**2) - r
        arr1 = np.array([x+r, y])
        arr2 = np.array([r, 0])
        theta = np.arccos(np.dot(arr1, arr2) / (r*np.linalg.norm(arr1)))
        s = r * theta
    elif piece == 2:
        # print('straight')
        s = abs(y) + np.pi*r
        n = -(x + 2*r)
    elif piece == 3:
        # print('turn')
        n = np.sqrt((x + r)**2 + (y + straight_len)**2) - r
        arr1 = np.array([x+r, y+straight_len])
        arr2 = np.array([-r, 0])
        theta = np.arccos(np.dot(arr1, arr2) /
                          (np.linalg.norm(arr2)*np.linalg.norm(arr1)))
        s = r * theta + straight_len + np.pi*r
    return np.array([s, n])  # Return Numpy Array


class SysID():
    # Store instances of A and B models for the car in SysID Objects
    def __init__(self):
        # system output data arrays
        # INITIALIZE X AND Y ARRAY WITH ZEROS TO CALC VELOCITY
        self.x_arr = np.array([0])
        self.y_arr = np.array([0])
        self.s_arr_raw = np.array([0])  # MODDED S
        self.s_arr = np.array([0])  # UNMODDED S
        self.n_arr = np.array([0])
        # OUPUT velocity PROJECTED onto centerline (s_dot) - this will give time step in upper right A term
        self.s_dot_arr = np.array([])
        # Output velocity normal to centerline (n_dot)
        self.n_dot_arr = np.array([])
        # array of angle headings (arctan(n_dot/s_dot))
        self.phi_arr = np.array([])
        self.time_step = 0.25  # DESIRED TIME STEP
        # FULL STATE VECTOR
        self.state_vec = []  # x = [s, n, v]^T
        # system input data arrays
        self.speed_arr_input = np.array([])  # INPUT speed array in anki units
        self.offset_arr = np.array([])  # input offset in anki units
        # FULL INPUT VECTOR
        self.input_vec = []  # u = [offset, input_speed]^T
        # regression matrices for data
        self.A_matrix = []
        self.B_matrix = []
        self.error = 0
        # data collection parameters
        self.speed_max = 800  # maximum speed of car for data collection
        self.speed_min = 300
        self.offset_min = -20*2  # set offset range of car for data collection
        self.offset_max = 20*2
        self.total_steps = 1  # set number of iterations of nested for loop

        # add lap parameter
        self.lap = 0  # initialize lap as 0

        # SAVED DATA NAMES
        self.savedStates = "stateVecData2.npy"
        self.savedInputs = "inputVecData2.npy"
        self.savedSmoothed = "smoothedStateVec2.npy"
        self.savedExpSmoothed = "expSmoothedStateVec2.npy"
        self.savedA = "savedAMatrix2.npy"  # save A and B for AnkiCar class
        self.savedB = "savedBmatrix2.npy"

    def sysID_datacollect(self, pointFromPixel, last_image_service):
        # collect data for sys id of A and B matrices
        # initialize overdrive object
        car = Overdrive("C9:08:8F:21:77:E3")  # overdrive object and address
        car.changeSpeed(self.speed_min, 1000)
        # define number of iterations over inputs
        step = 0
        increment = 10  # increment for speed and offset in each loop
        print((self.speed_max-self.speed_min)/increment *
              (self.offset_max-self.offset_min)/increment)
        break_flag = False  # flag to break out of inner loops
        getCarPos = GetCarState()
        while step < self.total_steps and break_flag == False:
            print('Steps to go', abs(step-self.total_steps))
            for speed in np.arange(self.speed_min, self.speed_max, increment):
                if break_flag == True:
                    break
                for offset in np.arange(self.offset_min, self.offset_max, increment):
                    time_start = datetime.datetime.now()  # start recording for time discretization
                    print('Speed to go', abs(speed-self.speed_max))
                    print('Offset to go', abs(offset-self.offset_max))
                    car.changeSpeed(speed, 1000)
                    car.changeLane(1000, 1000, offset)
                    try:
                        # GET CAR STATE
                        #getCarPos = GetCarState()
                        ros_img_msg = last_image_service().image_data
                        getCarPos.mask_car(
                            ros_img_msg, landmarks=pointFromPixel.points)
                        car_pos_xy = pointFromPixel.pixel_to_track_cartesian(
                            np.array([getCarPos.blue_y, getCarPos.blue_x]))
                        # transform to sn coordinates
                        car_pos_sn_raw = coord_solver(car_pos_xy)
                        print("XY coordinates: ", car_pos_xy)
                        print("Modded SN coordinates: ", car_pos_sn_raw)

                        # Calculate velocity value
                        #vel_solver = lambda xy_curr, xy_prev: np.linalg.norm(xy_curr-xy_prev)/self.time_step
                        # Take care of jump of s back to 0 from 226.097 at the end of every lap
                        def s_dot_solver(s_curr, s_prev):
                            if s_curr-s_prev < -100:  # define -100 as a threshold value
                                return (s_curr+226.097-s_prev)/self.time_step
                            return (s_curr-s_prev)/self.time_step

                        def n_dot_solver(n_curr, n_prev): return (
                            n_curr-n_prev)/self.time_step
                        # get values for velocity output arrays

                        def s_solver(s_curr, s_prev):
                            # takes in two modded s values, returns unmodded
                            if s_curr-s_prev < -100:  # define -100 as a threshold value
                                self.lap += 1
                            print("Lap: ", self.lap, "S unmodded: ",
                                  self.lap*226.097 + s_curr)
                            return self.lap*226.097 + s_curr

                        self.s_dot_arr = np.append(self.s_dot_arr, s_dot_solver(
                            car_pos_sn_raw[0], self.s_arr_raw[-1]))
                        self.n_dot_arr = np.append(self.n_dot_arr, n_dot_solver(
                            car_pos_sn_raw[1], self.n_arr[-1]))
                        # get values for car angle
                        self.phi_arr = np.append(self.phi_arr, np.arctan2(
                            self.n_dot_arr[-1], self.s_dot_arr[-1]))
                        # enter values into input array
                        self.speed_arr_input = np.append(
                            self.speed_arr_input, speed)
                        self.offset_arr = np.append(self.offset_arr, offset)
                        # get values for s n output arrays
                        unmodded_s = s_solver(
                            car_pos_sn_raw[0], self.s_arr_raw[-1])
                        self.s_arr = np.append(self.s_arr, unmodded_s)
                        self.s_arr_raw = np.append(
                            self.s_arr_raw, car_pos_sn_raw[0])
                        self.n_arr = np.append(self.n_arr, car_pos_sn_raw[1])
                        # get values for x y output arrays
                        self.x_arr = np.append(self.x_arr, car_pos_xy[0])
                        self.y_arr = np.append(self.y_arr, car_pos_xy[1])

                    except KeyboardInterrupt:
                        print('Keyboard Interrupt, exiting')
                        break_flag = True  # Change break flag value to break out of outer loops
                        break
                    # Catch if anything went wrong with the Image Service
                    except rospy.ServiceException as e:
                        print("image_process: Service call failed: %s" % e)

                    # np.save(self.savedStates, self.state_vec) #save every loop
                    #np.save(self.savedInputs, self.input_vec)

                    # FORCE DATA RECORDING TO HAPPEN AT EVEN INTERVALS
                    time_end = datetime.datetime.now()
                    t_delta = time_end - time_start
                    # get time delay in seconds of the whole loop
                    t_delta = t_delta.seconds + \
                        float(t_delta.microseconds)/(10**6)
                    print(t_delta)
                    time.sleep(self.time_step - t_delta)
            step += 1  # INCREMENT STEP VALUE

        num_dropped = 4  # number of data points to slice off the top of arrays
        # get rid of placeholder value from velocity calculation
        self.x_arr = self.x_arr[1:]
        self.y_arr = self.y_arr[1:]
        self.s_arr = self.s_arr[1:]
        self.n_arr = self.n_arr[1:]
        self.s_arr_raw = self.s_arr_raw[1:]
        # Horizontal stack data ararys to get recorded state and input vectors
        self.state_vec = np.vstack(
            (self.s_arr, self.n_arr, self.s_dot_arr, self.n_dot_arr, self.phi_arr))[:, num_dropped:]
        self.input_vec = np.vstack((self.offset_arr, self.speed_arr_input))[
            :, num_dropped:]

        # perform smoothing on each row
        #smoothed_data = np.copy(self.state_vec)
        # for row_index in range(self.state_vec.shape[0]):
        #    smoothed_row = wiener(smoothed_data[row_index, :], (3))  #2nd argument is the size of the filter
        #    smoothed_data[row_index, :] = smoothed_row
        np.save(self.savedStates, self.state_vec)
        np.save(self.savedInputs, self.input_vec)
        #np.save(self.savedSmoothed, smoothed_data)
        self.exp_smoothing()  # call and save smoothed data
        print("Final state vector array: ", self.state_vec)
        car.changeSpeed(0, 1000)  # Stop car after data collection

    def exp_smoothing(self):
        alpha = 0.5
        raw_data = np.load(self.savedStates)
        smoothed_data_exp = np.copy(raw_data)

        #smoothed_data = np.load("smoothedStateVec.npy")
        # gives a time array corresponding to each point
        time_array = np.arange(0, raw_data.shape[1])*self.time_step

        for row_index in [2, 3]:
            # smoothed_row = wiener(smoothed_data[col_index, :], (3))  #2nd argument is the size of the filter
            for t_step in range(raw_data.shape[1]):
                if t_step == 0:
                    smoothed_data_exp[row_index,
                                      t_step] = raw_data[row_index, t_step]
                else:
                    smoothed_data_exp[row_index, t_step] = (
                        1-alpha)*smoothed_data_exp[row_index, t_step-1] + (alpha)*raw_data[row_index, t_step]

        smoothed_data_exp[4, :] = np.arctan2(
            smoothed_data_exp[3, :], smoothed_data_exp[2, :])
        for row_index in range(5):
            plt.plot(time_array, raw_data[row_index, :])
            #plt.plot(time_array, smoothed_data[row_index, :])
            plt.plot(time_array, smoothed_data_exp[row_index, :])
            plt.title("Data smoothed vs unsmoothed")
            plt.show()
        np.save(self.savedExpSmoothed, smoothed_data_exp)

    def plot_smoothing(self, state_index):
        """
        Visualize effects of smoothing
        """
        smoothed_data = np.load("expSmoothedStateVec2.npy")
        state_data = np.load(self.savedStates)
        print(state_index)
        # gives a time array corresponding to each point
        time_array = np.arange(0, state_data.shape[1])*self.time_step
        print(time_array.shape)
        print(state_data.shape[1])
        plt.plot(time_array, smoothed_data[state_index, :])
        plt.plot(time_array, state_data[state_index, :])
        plt.title("Velocity smoothed vs unsmoothed")
        plt.show()

    def regression_law(self):
        print("Regression Beginning...")
        lam = 0  # regularization
        x = np.load(self.savedExpSmoothed).T
        # x = self.state_vec.T # n,5
        state_dim = 5  # number of states (5 max) to use
        X = x[:-1, :state_dim]
        Y = x[1:, :state_dim]
        u = np.load(self.savedInputs).T
        # u = self.input_vec.T # n,2
        U = u[:-1, :]

        print("X shape: ", X.shape)
        print("U shape: ", U.shape)
        A = np.hstack([X, U])  # n,7
        # A = np.array([[1, 0, 0.25, 0, 0],
        #               [0, 1, 0, 0.25, 0],
        #               [0, 0, 1, 0, 0],
        #               [0, 0, 0, 1, 0],
        #               [0, 0, 0, 0, 1]])
        # self.A = A
        coeff = np.linalg.inv(np.dot(A.T, A) + lam *
                              np.eye(state_dim+2)).dot(A.T).dot(Y)  # (7,5)
        self.A = coeff[:state_dim, :].T  # (5,5)
        self.B = coeff[state_dim:, :].T  # (5,2)
        # self.B = (np.dot(np.linalg.pinv(U), Y-np.dot(X, self.A.T))).T #MAKE SURE TO TRANSPOSE
        print("Lawrence")
        print("A: ", self.A)
        print("B: ", self.B)
        print("B Shape: ", self.B.shape)

        # error
        #error_matrix = np.dot(A, coeff)-Y
        #error_mean = np.mean(error_matrix, axis=0)
        #error_max = np.max(error_matrix, axis=0)
        #error_min = np.min(error_matrix, axis=0)
        #error = np.vstack((error_mean, error_min, error_max))

        #print("error: ", error)
        # save data
        # np.save(self.savedA, self.A)
        # np.save(self.savetupledB, self.B)

        # plt.plot(np.arange(A.shape[0]), np.dot(A, coeff)[:, 0], label='predicted')
        # plt.plot(np.arange(A.shape[0]), Y[:, 0], label='actual')
        # plt.legend()
        # plt.title("Actual vs Predicted s")
        # plt.show()

        # plt.plot(np.arange(A.shape[0]), np.dot(A, coeff)[:, 1], label='predicted')
        # plt.plot(np.arange(A.shape[0]), Y[:, 1], label='actual')
        # plt.legend()
        # plt.title("Actual vs Predicted Offset")
        # plt.show()

        # plt.plot(np.arange(A.shape[0]), np.dot(A, coeff)[:, 2], label='predicted')
        # plt.plot(np.arange(A.shape[0]), Y[:, 2], label='actual')
        # plt.legend()
        # plt.title("Actual vs Predicted n")
        # plt.show()



if __name__ == '__main__':

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
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point0)
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point1)
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point2)
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point3)
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point4)
            # pointFromPixel.pixel_to_track_cartesian(pointFromPixel.point5)
            # np_image = pointFromPixel.ros_to_np_img(ros_img_msg)
            # Display the CV Image
            # cv2.imshow("CV Image", np_image)
            # cv2.waitKey(0)

        except KeyboardInterrupt:
            print('Keyboard Interrupt, exiting')
            break

        # Catch if anything went wrong with the Image Service
        except rospy.ServiceException as e:
            print("image_process: Service call failed: %s" % e)

    if flag:
        while not rospy.is_shutdown():
            try:
                # getCarPos = GetCarState()
                # ros_img_msg = last_image_service().image_data
                # getCarPos.mask_red_obstacle(
                #    ros_img_msg, pointFromPixel.list_of_detected_points, pointFromPixel)
                # getCarPos.get_lw_xy(
                #     ros_img_msg, pointFromPixel.list_of_detected_points, pointFromPixel)  # get obstacle
                # print([getCarPos.blue_x, getCarPos.blue_y])
                # print(pointFromPixel.pixel_to_track_cartesian(np.array([getCarPos.blue_y, getCarPos.blue_x])))
                trial_SysID = SysID()
                # trial_SysID.sysID_datacollect(pointFromPixel, last_image_service)
                trial_SysID.regression_law()
                # trial_SysID.plot_smoothing(0)
                # trial_SysID.plot_smoothing(1)
                # trial_SysID.plot_smoothing(2)
                # trial_SysID.plot_smoothing(3)
                # trial_SysID.plot_smoothing(4)
            except KeyboardInterrupt:
                print('Keyboard Interrupt, exiting')
                break

            # Catch if anything went wrong with the Image Service
            except rospy.ServiceException as e:
                print("image_process: Service call failed: %s" % e)
