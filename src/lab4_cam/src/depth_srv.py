#!/usr/bin/env python
import rospy

from sensor_msgs.msg import Image, PointCloud2
from lab4_cam.srv import ImageSrv, ImageSrvResponse


class ImgService:
  #Callback for when an image is received
  def imgReceived(self, message):
    #Save the image in the instance variable
    self.lastImage = message

    #Print an alert to the console
    #print(rospy.get_name() + ":Image received!")

  #When another node calls the service, return the last image
  def getLastImage(self, request):
    #Print an alert to the console
    #print("Image requested!")

    #Return the last image
    return ImageSrvResponse(self.lastImage)

  def __init__(self):
    #Create an instance variable to store the last image received
    self.lastImage = None;

    #Initialize the node
    rospy.init_node('depth_listener')

    #Subscribe to the image topic
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.imgReceived)
    # rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.imgReceived)

    #Create the service
    rospy.Service('/last_depth_image', ImageSrv, self.getLastImage)

  def run(self):
    rospy.loginfo('Running depth image server ...')
    rospy.spin()

#Python's syntax for a main() method
if __name__ == '__main__':
  node = ImgService()
  node.run()
