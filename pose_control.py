#!/usr/bin/env python

import rospy
import tf
import random
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from math import atan2, sqrt

class PoseController:
    def __init__(self):
        rospy.init_node('pose_controller', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.Subscriber('/odom', Odometry, self.callback)
        
        self.rate = rospy.Rate(100)
        
        # Random initial position inside a 2x2 meter area
        self.init_x = -1
        self.init_y = 1
        
        # Fixed goal position inside the same area
        self.goal_x = -1
        self.goal_y = -1
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.rotation_flag = True
        self.come_flag = False
        self.stop_flag = False
        
        self.start_time = None
        
    def odom(self, msg):
        self.x = msg.pose.pose.position.x + self.init_x
        self.y = msg.pose.pose.position.y + self.init_y
        
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        _, _, self.theta = tf.transformations.euler_from_quaternion(quaternion)
        
    def rotate_to_goal(self):
        vel_msg = Twist()
        angle_to_goal = atan2(self.goal_y - self.y, self.goal_x - self.x)
        
        if abs(angle_to_goal - self.theta) > 0.05:
            angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = angular_speed
            self.cmd_vel_pub.publish(vel_msg)
        else:
            vel_msg.angular.z = 0.0
            self.come_flag = True
            self.rotation_flag = False

            if self.stop_flag:
                self.cmd_vel_pub.publish(vel_msg)
                elapsed_time = time.time() - self.start_time
                print(f"Goal reached in {elapsed_time:.2f} seconds!")

                print("EXITING...")
                rospy.signal_shutdown("EXITING. GOODBYE!")
        
        self.cmd_vel_pub.publish(vel_msg)
    
    def move_to_goal(self):
        vel_msg = Twist()
        
        # Compute distance and angle to the goal
        distance = sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
        angle_to_goal = atan2(self.goal_y - self.y, self.goal_x - self.x)
            
        # Proportional controller with increased speed
        linear_speed = min(0.5 * distance, 0.5)  # Increased max speed
        angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)  # Increased max angular speed
            
        if distance < 0.05:  # Stop condition
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(vel_msg)

            self.rotation_flag = True
            self.come_flag = False
            self.stop_flag = True
            self.goal_x = self.init_x
            self.goal_y = self.init_y
            
        vel_msg.linear.x = linear_speed
        vel_msg.angular.z = angular_speed
            
        self.cmd_vel_pub.publish(vel_msg)
    
    def callback(self, msg):

        self.odom(msg)

        if self.rotation_flag:
            self.rotate_to_goal()
        
        if self.come_flag:
            self.move_to_goal()
        
        self.rate.sleep()


if __name__ == '__main__':
        controller = PoseController()
        controller.reset_simulation()
        controller.start_time = time.time()

        rospy.spin()
