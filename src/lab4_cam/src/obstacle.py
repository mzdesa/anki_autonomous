#!/usr/bin/env python
import numpy as np 
#from mapping_track_to_model import coord_solver

class Obstacle:
    def __init__(self, x, y, width, length):
        self.x = x
        self.y  = y
        self.buffer = 2
        self.width = width
        self.length = length
        sn = coord_solver([self.x, self.y])
        self.delta = sn[1] #test
        #self.s = self.convert_colinear2xy()
        self.s = sn[0] #test

        #Bounding box center
        self.red_obstacle_bb_center_x = None
        self.red_obstacle_bb_center_y = None

        # 4x2 array containing the XY global fr corners of an obstacle
        self.obstacle_corners = np.array([])
        self.obstacle_corners_pix = np.array([])
        # a (2,) numpy array (x, y) of obstacle center in cm
        self.obstacle_xy_cm = np.array([])
        self.obstacleL = None  # obstacle length
        self.obstacleW = None  # obstacle width
    
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