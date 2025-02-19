#!/usr/bin/env python

import rospy
import tf
import random
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3
from std_srvs.srv import Empty
from math import atan2, sqrt

class PoseController:
    def init(self):
        rospy.init_node('pose_controller', anonymous=True)
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_14/cmd_wheels', Vector3, queue_size=1)
        #self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.Subscriber('/turtlebot_14/odom', Odometry, self.odom, queue_size=1)
        
        self.rate = rospy.Rate(100)
        
        # Random initial position inside a 2x2 meter area
        self.init_x = 0
        self.init_y = 0
        
        # Fixed goal position inside the same area
        self.goal_x = -1
        self.goal_y = 0
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.rotation_flag = True
        self.come_flag = False
        self.stop_flag = False
        
        self.start_time = None
    
    def yaw_from_quaternion(self, q):
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        return atan2(siny_cosp, cosy_cosp)
    
    def odom(self, msg):
        self.x = msg.pose.pose.position.x + self.init_x
        self.y = msg.pose.pose.position.y + self.init_y
        
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        self.theta = self.yaw_from_quaternion(quaternion)
    
    def publish_velocity(self, action):
        """Publish velocity commands to the robot"""
        v = action[0]
        w = action[1]
        
        d = 0.173
        r = 0.0325

        w_r = (v + w * d/2) / r
        w_l = (v - w * d/2) / r
        vel_msg = Vector3(w_r, w_l, 0)

        self.cmd_vel_pub.publish(vel_msg)

    def rotate_to_goal(self):
        angle_to_goal = atan2(self.goal_y - self.y, self.goal_x - self.x)
        
        if abs(angle_to_goal - self.theta) > 0.05:
            angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)
            self.publish_velocity([0.0, angular_speed])
        else:
            self.come_flag = True
            self.rotation_flag = False

            if self.stop_flag:
                self.publish_velocity([0.0, 0.0])
                elapsed_time = time.time() - self.start_time
                print(f"Goal reached at ({self.goal_x}, {self.goal_y}) in {elapsed_time:.2f} seconds!")

                print("EXITING...")
                rospy.signal_shutdown("EXITING. GOODBYE!")
    
    def move_to_goal(self):
        # Compute distance and angle to the goal
        distance = sqrt((self.goal_x - self.x)**2 + (self.goal_y - self.y)**2)
        angle_to_goal = atan2(self.goal_y - self.y, self.goal_x - self.x)
            
        # Proportional controller with increased speed
        linear_speed = min(0.5 * distance, 0.5)  # Increased max speed
        angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)  # Increased max angular speed
            
        if distance < 0.05:  # Stop condition
            self.publish_velocity([0.0, 0.0])

            self.rotation_flag = True
            self.come_flag = False
            self.stop_flag = True
            self.goal_x = 1
            self.goal_y = 0
            
        self.publish_velocity([linear_speed, angular_speed])
    
    def callback(self, msg):

        self.odom(msg)

        if self.rotation_flag:
            self.rotate_to_goal()
        
        if self.come_flag:
            self.move_to_goal()
        
        #self.rate.sleep()


if __name__ == 'main':
        controller = PoseController()
        #controller.reset_simulation()
        controller.start_time = time.time()

        rospy.spin()