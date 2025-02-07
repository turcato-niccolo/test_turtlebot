import rospy
#import torch

from nav_msgs.msg import Odometry
import tf.transformations
from geometry_msgs.msg import Twist, Vector3
from std_srvs.srv import Empty
from vicon.msg import Subject

import time
import tf

import numpy as np

#import ExpD3
#import OurDDPG, TD3, SAC
#import utils

pub = rospy.Publisher('/turtlebot_14/cmd_wheels', Vector3, queue_size=1)
#reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

def reset():
    global start_time
    start_time = None
    #reset_simulation()
    time.sleep(0.2)


def callback(msg):
    
    quaternion = (
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]

    x = np.zeros((6,))
    x[0] = msg.position.x #- 3.0 # this is to move the reference frame
    x[1] = msg.position.y
    x[2] = yaw
    #x[3] = msg.twist.twist.linear.x
    #x[4] = msg.twist.twist.linear.y
    #x[5] = msg.twist.twist.angular.z

    #vel = Twist()
    #vel.linear.x = 0.5
    #vel.angular.z = 0.2
    '''v = 0.5
    w = 0.2
    r = 0.128
    d = 0.3

    w_r = (v + w * d/2) / r
    w_l = (v - w * d/2) / r'''

    # (destra, sinistra, null)
    vel = Vector3(0., 0., 0) # (r l null)

    print(vel)
    
    pub.publish(vel)
    time.sleep(0.01)

    print("x:", x[0], " y:", x[1], "theta:", x[2])


if __name__ == "__main__":
    print("START")

    reset()
    rospy.init_node('odometry', anonymous=True) #make node
    rospy.Subscriber('/vicon/turtlebot_14', Subject, callback, queue_size=1)

    print("FINISH")

    rospy.spin()
