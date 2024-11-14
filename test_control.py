import rospy

import torch

from nav_msgs.msg import Odometry
import tf.transformations
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import time
import tf

import numpy as np

import ExpD3
import OurDDPG, TD3, SAC
import utils


''' Implementare la reward function - Implementare lo spawn del robot random - Implementare contatore dei tempi ~ 10 h di simulazione in base alla frequenza '''
# Initialize publisher for cmd_vel
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

# Define state dimensions and action space
state_dim = 6
action_dim = 2

max_vel = [2.0, np.pi/2]
max_time = 10
start_time = None

old_state = None
old_action = None
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=10**5)
# policy = TD3.TD3(state_dim, action_dim, max_action=1)
policy = ExpD3.DDPG(state_dim, action_dim, max_action=1)
#policy = OurDDPG.DDPG(state_dim, action_dim, max_action=1, tau=0.1)

GOAL = [3, 0]
OBSTACLE = [0, 0]
WALL_dist = 5.0
GOAL_dist = 0.5
OBST_dist = 1.0


def reset():
    global start_time
    start_time = None
    reset_simulation()
    time.sleep(0.2)


def callback(msg):
    global pub, old_state, old_action, replay_buffer, max_vel, start_time
    done = False
    reward = 0.0
    if start_time is None:
        start_time = time.time()
    if time.time() - start_time > max_time:
        done = True
        # reset()

    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]

    x = np.zeros((6,))
    x[0] = msg.pose.pose.position.x - 1.0 # this is to move the reference frame
    x[1] = msg.pose.pose.position.y
    x[2] = yaw
    x[3] = msg.twist.twist.linear.x
    x[4] = msg.twist.twist.linear.y
    x[5] = msg.twist.twist.angular.z

    if replay_buffer.size > 10 ** 3:
        action = (
                policy.select_action(np.array(x))
                + np.random.normal(0, 0.3, size=action_dim)
        ).clip(-1, 1)
    else:
        action = (np.random.normal(0, 1, size=action_dim)).clip(-1, 1)

    vel = Twist()
    vel.linear.x = action[0] * max_vel[0]
    vel.angular.z = action[1] * max_vel[1]

    pub.publish(vel)
    
    time.sleep(0.1)

    next_state = x

    done_bool = float(done)

    if old_state is not None:
        # Store data in replay buffer
        replay_buffer.add(old_state, old_action, next_state, reward, done_bool)
        if replay_buffer.size > 10**3:
            policy.train(replay_buffer, batch_size=256)
            print('train')

    old_state = next_state if not done else None
    old_action = action if not done else None




if __name__ == "__main__":
    reset()
    rospy.init_node('oodometry', anonymous=True) #make node
    rospy.Subscriber('/odom', Odometry, callback, queue_size=1)

    rospy.spin()
