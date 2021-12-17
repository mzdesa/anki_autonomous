#!/usr/bin/env python
#The line above tells Linux that this file is a Python script,
#and that the OS should use the Python interpreter in /usr/bin/env
#to run it. Don't forget to use "chmod +x [filename]" to make
#this script executable.

#Import the rospy package. For an import to work, it must be specified
#in both the package manifest AND the Python file in which it is used.
import rospy
import tf2_ros
import sys

from geometry_msgs.msg import Twist

#Define the method which contains the main functionality of the node.
def controller():
  
  #Create a publisher and a tf buffer, which is primed with a tf listener
  pub = rospy.Publisher('/track2cam_transform', Twist, queue_size=10)
  tfBuffer = tf2_ros.Buffer()
  tfListener = tf2_ros.TransformListener(tfBuffer)
  t = Twist()

  # Create a timer object that will sleep long enough to result in
  # a 10Hz publishing rate
  r = rospy.Rate(10) # 10hz

  K1 = 0.3
  K2 = 1
  # Loop until the node is killed with Ctrl-C
  while not rospy.is_shutdown():
    try:
      trans = tfBuffer.lookup_transform("ar_marker_0", "camera_color_frame", rospy.Time())

      # Process trans to get your state error
      #vel = K1 * (trans.transform.translation.x)
      #theta = K2 * (trans.transform.translation.y)
      
      #control_command = Twist() # Generate this

      # Generate a control command to send to the robot
      #control_command.linear.x = vel
      #control_command.angular.z = theta

      #################################### end your code ###############
      rospy.loginfo(trans)
      #pub.publish(control_command)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      print("Error with connection")
      pass
    # Use our rate object to sleep until it is time to publish again
    r.sleep()

      
# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':
  # Check if the node has received a signal to shut down
  # If not, run the talker method

  #Run this program as a new node in the ROS computation graph 
  #called /turtlebot_controller.
  rospy.init_node('camConnection', anonymous=True)

  try:
    controller()
  except rospy.ROSInterruptException:
    pass