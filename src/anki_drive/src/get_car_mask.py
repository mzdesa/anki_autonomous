#!/usr/bin/env python

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

from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Twist
from lab4_cam.msg import CarPosition

bridge = CvBridge()
# depth_cam_info = None
cam_model = None
# X = None
# Y = None
world_coords = None

class PointFromPixel():
  def mask_blue_car(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,(100,0,70), (240,240,240))
    bbox = get_loop_box(mask)
    mask2 = np.zeros((img.shape[0], img.shape[1]))
    mask2[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1.0
    mask2 = np.stack((mask2, mask2, mask2), axis=2)
    mask2 = mask2.astype(np.uint8)
    car_mask = np.multiply(img, mask2)
    y = int(bbox[1] + bbox[3]/2)
    x = int(bbox[0] + bbox[2]/2)
    return x, y

  def ros_to_np_img(ros_img_msg):
    return np.array(bridge.imgmsg_to_cv2(ros_img_msg,'bgr8'))

  def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):  
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = pyrealsense2.distortion.none  
    _intrinsics.coeffs = [i for i in cameraInfo.D]  
    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)  #result[0]: right, result[1]: down, result[2]: forward
    return result[2], -result[0], -result[1]

  def get_loop_box(mask, largest=1):
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
    
    if largest == 1:
      return bboxes[0]
    return bboxes[:largest]

  # def get_depth_cam_info(message):
  #   global depth_cam_info
  #   depth_cam_info = message

  def get_cam_model(message):
    global cam_model
    cam_model = image_geometry.PinholeCameraModel().fromCameraInfo(message)

  def xyzCallback(data):
    gen = pc2.read_points(data, field_names=("z"), skip_nans=False, uvs=[(X, Y)]) #may also need to have field names x and y
    nextGen = next(gen)
    vector = np.array(cam_model.projectPixelTo3dRay((X, Y)))
    ray_z = [el/vector[2] for el in vector]
    a = vector[0]
    b = vector[1]
    # points = nextGen[:, 0:2]
    # tree= cKDTree(points)

    global world_coords
    world_coords = ray_z
    print(world_coords)


if __name__ == '__main__':
  # Waits for the image service to become available
  rospy.wait_for_service('last_color_image')
  # rospy.wait_for_service('last_depth')
  rospy.init_node('car_mask', anonymous=True)

  pub = rospy.Publisher('/car_position', CarPosition, queue_size=10)

  last_image_service = rospy.ServiceProxy('last_color_image', ImageSrv)
  # last_depth_service = rospy.ServiceProxy('last_depth', ImageSrv)

  pointFromPixel = PointFromPixel()

  rospy.Subscriber("/camera/depth/camera_info", Image, pointFromPixel.get_cam_model)
  rospy.Subscriber("/camera/depth/color/points", PointCloud2, pointFromPixel.xyzCallback)

  global X 
  global Y

  while not rospy.is_shutdown():
    try:
      ros_img_msg = last_image_service().image_data
      np_image = pointFromPixel.ros_to_np_img(ros_img_msg)

      # ros_depth_msg = last_depth_service().image_data

      # Display the CV Image
      #cv2.imshow("CV Image", np_image)
      X, Y = mask_blue_car(np_image)
      position = CarPosition()
      position.x = X 
      position.y = Y 
      # world_x, world_y, world_z = convert_depth_to_phys_coord_using_realsense(x, y, ros_depth_msg, depth_cam_info)
      pub.publish(position)

    except KeyboardInterrupt:
      print('Keyboard Interrupt, exiting')
      break

    # Catch if anything went wrong with the Image Service
    except rospy.ServiceException as e:
      print("image_process: Service call failed: %s"%e)