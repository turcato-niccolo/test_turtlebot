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

pub = rospy.Publisher('/wcias_controller/cmd_vel', Twist, queue_size=1)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

state_dim = 6
action_dim = 2

max_vel = [2.0, np.pi/2]
max_time = 10
start_time = None

old_state = None
old_action = None
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=10**5)
# policy = TD3.TD3(state_dim, action_dim, max_action=1)
# policy = ExpD3.DDPG(state_dim, action_dim, max_action=1)
policy = OurDDPG.DDPG(state_dim, action_dim, max_action=1, tau=0.1)

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
        reset()

    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]

    x = np.zeros((6,))
    x[0] = msg.pose.pose.position.x - 3.0 # this is to move the reference frame
    x[1] = msg.pose.pose.position.y
    x[2] = yaw
    x[3] = msg.twist.twist.linear.x
    x[4] = msg.twist.twist.linear.y
    x[5] = msg.twist.twist.angular.z

    if np.abs(x[0]) > WALL_dist or np.abs(x[1]) > WALL_dist:
        done = True
        reset()
        reward += -10

    if np.sqrt((x[0]-OBSTACLE[0])**2 + (x[1]-OBSTACLE[1])**2) < OBST_dist:
        done = True
        reset()
        reward += -10

    if np.sqrt((x[0]-GOAL[0])**2 + (x[1]-GOAL[1])**2) < GOAL_dist:
        done = True
        reset()
        reward += +1000

    if replay_buffer.size > 10 ** 3:
        action = (
                policy.select_action(np.array(x))
                + np.random.normal(0, 0.1, size=action_dim)
        ).clip(-1, 1)
    else:
        action = (np.random.normal(0, 1, size=action_dim)).clip(-1, 1)

    vel = Twist()
    vel.linear.x = action[0] * max_vel[0]
    vel.angular.z = action[1] * max_vel[1]

    if not done:
        pub.publish(vel)

    next_state = x
    if old_state is not None:
        reward += +1 if np.sqrt((next_state[0]-GOAL[0])**2 + (next_state[1]-GOAL[1])**2) < np.sqrt((old_state[0]-GOAL[0])**2 + (old_state[1]-GOAL[1])**2) else -1
        # reward += -np.sqrt((next_state[0]-GOAL[0])**2 + (next_state[1]-GOAL[1])**2)
        reward += 5 * np.exp(-((next_state[0]-GOAL[0])**2 + (next_state[1]-GOAL[1])**2)/4)

    # reward = -np.sqrt((x[0]-1)**2 + (x[1]-4)**2)
    print('Reward', reward, 'state', x, 'action', action, done)

    done_bool = float(done)

    if old_state is not None:
        # Store data in replay buffer
        replay_buffer.add(old_state, old_action, next_state, reward, done_bool)
        if replay_buffer.size > 10**3:
            policy.train(replay_buffer)
            print('train')

    old_state = next_state if not done else None
    old_action = action if not done else None




if __name__ == "__main__":
    reset()
    rospy.init_node('oodometry', anonymous=True) #make node
    rospy.Subscriber('/wcias_controller/odom', Odometry, callback, queue_size=1)

    rospy.spin()
