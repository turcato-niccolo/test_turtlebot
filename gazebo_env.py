import rospy
import numpy as np
import time

from geometry_msgs.msg import Twist, Pose, Vector3
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry

class GazeboEnv:

    def __init__(self):

        self.state = None

        self.MAX_VEL = [0.5, np.pi/4]

        # Environment parameters
        self.GOAL = np.array([1.0, 0.0])
        self.OBSTACLE = np.array([0.0, 0.0])
        self.WALL_DIST = 1.0
        self.GOAL_DIST = 0.15
        self.OBST_W = 0.5
        self.OBST_D = 0.2
        self.HOME = np.array([-1, 0.0])

        # Reward parameters
        self.DISTANCE_PENALTY = 0.5
        self.GOAL_REWARD = 1000
        self.OBSTACLE_PENALTY = 100
        self.MOVEMENT_PENALTY = 1
        self.GAUSSIAN_REWARD_SCALE = 2

        self.TIME_DELTA = 1/6 # For TD3

        self._initialize_ros()
    
    def _initialize_ros(self):
        """Initialize ROS nodes, publishers, and services"""

                         # Initialize velocity publisher
            
            # Initialize odometry subscriber
            
        
        # Initialize ROS node and publishers
        rospy.init_node('environment', anonymous=True)                                      # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_14/cmd_wheels', Vector3, queue_size=1)
        
        # Initialize odometry subscriber
        rospy.Subscriber('/vicon/turtlebot_14', Vector3, self.callback, queue_size=1)

        self.reset_simulation()
        print("ENV INIT...")

    def callback(self, msg):
        """Extract state information from odometry message"""
        # Robot position
        x = msg.pose.pose.position.x + self.HOME[0]
        y = msg.pose.pose.position.y + self.HOME[1]
        
        # Get orientation
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        # Compute yaw angle from quaternion
        x,y,z,w = quaternion
        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Homogeneous tranformation
        H = np.attay([0, 1, 0],
                     [-1, 0, -1],
                     [0, 0, 1])
        vec_hom = np.append([x, y], 1)
        trans_vec = H @ vec_hom
        x, y = trans_vec[0], trans_vec[1]
        
        # Robot velocities
        linear_vel = msg.twist.twist.linear.x
        vel_x = linear_vel * np.cos(yaw)
        vel_y = linear_vel * np.sin(yaw)
        angular_vel = msg.twist.twist.angular.z
        
        self.state = np.array([x, y, yaw, vel_x, vel_y, angular_vel])

    def publish_action(self, action):

        v = action[0] * self.MAX_VEL[0]
        w = action[1] * self.MAX_VEL[1]

        d = 0.173
        r = 0.0325

        w_r = (v + w*d/2) / r
        w_l = (v - w*d/2) / r
        vel_msg = Vector3(w_r, w_l, 0)

        self.cmd_vel_pub.publish(vel_msg)
    
    def step(self):
        state = self.state
        reward, done, target = self.get_reward()

        return state, reward, done, target

    def reset(self):

        # Publish velocity commands to stop the robot
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.cmd_vel_pub.publish(vel_msg)
        
        state = self.state

        r = np.sqrt(np.random.uniform(0,1))*0.1
        theta = np.random.uniform(0,2*np.pi)
        self.HOME = np.array([-1 + r * np.cos(theta), 0 + r * np.sin(theta)])

        return state, self.HOME

    def publish_markers(self):
        pass

    def change_position(self):
        pass

    def get_state(self):
        return self.state

    def get_reward(self):

        p = self.state[:2]
        dist_to_goal = np.linalg.norm(p - self.GOAL)
        
        # Initialize reward and termination flag
        reward = 0
        done = False
        target = False
        
        # Distance-based reward components
        reward -= self.DISTANCE_PENALTY * dist_to_goal ** 2
        reward += self.GAUSSIAN_REWARD_SCALE * np.exp(-dist_to_goal**2)
        
        bound_x = self.WALL_DIST + 0.2
        bound_y = bound_x

        # Check boundary
        if np.abs(p[0]) >= bound_x or np.abs(p[1]) >= bound_y:
            done = True

        # Check collision with obstacle
        if np.abs(p[0]) <= self.OBST_D / 2 and np.abs(p[1]) <= self.OBST_W / 2:
            reward -= 10
            done = True
        
        # Check goal achievement
        if dist_to_goal <= self.GOAL_DIST:
            reward += self.GOAL_REWARD
            done = True
            target = True
        
        return reward, done, target