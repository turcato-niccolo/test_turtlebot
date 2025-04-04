from train import GazeboEnv
import tf.transformations
import numpy as np
import rospy
import os

from config import parse_args

class CustomGazeboEnv(GazeboEnv):

    def trajectory(self):
        """Define the trajectory of the robot"""
        a, b = 1, 1
        x_t = a * self.t
        y_t = b * np.sin(x_t)
        return x_t, y_t

    def odom(self):
        """Extract state information from odometry message"""
        # Robot position
        x = self.msg.pose.pose.position.x #+ self.HOME[0]
        y = self.msg.pose.pose.position.y #+ self.HOME[1]
        self.x, self.y = x, y
        # Get orientation
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.theta = yaw
        
        # Robot velocities
        linear_vel = self.msg.twist.twist.linear.x
        vel_x = linear_vel * np.cos(yaw)
        vel_y = linear_vel * np.sin(yaw)
        angular_vel = self.msg.twist.twist.angular.z
        
        self.state = np.array([x, y, yaw, vel_x, vel_y, angular_vel])
    
    def get_reward(self):
        "Get reward"
        p = np.array(self.state[:2])  # Current position
        dist_to_goal = np.linalg.norm(p - self.GOAL)
        reward = -self.DISTANCE_PENALTY * dist_to_goal ** 2 + self.GAUSSIAN_REWARD_SCALE * np.exp(-dist_to_goal**2)
        terminated = False

        # Movement reward/penalty
        if self.old_state is not None:
            prev_dist_to_goal = np.linalg.norm(np.array(self.old_state[:2]) - self.GOAL)
            reward += self.MOVEMENT_PENALTY if dist_to_goal < prev_dist_to_goal else -self.MOVEMENT_PENALTY

        # Check if agent is out of bounds
        if np.any(np.abs(p) >= self.WALL_DIST + 0.2):
            #print("DANGER ZONE")
            return reward, True, False  # Terminate immediately if out of bounds

        # Check collision with obstacle
        if np.abs(p[0]) <= self.OBST_D / 2 and np.abs(p[1]) <= self.OBST_W / 2:
            #print("OBSTACLE")
            return reward - 10, True, False  # Immediate termination on collision

        # Check goal achievement
        if dist_to_goal <= self.GOAL_DIST:
            #print("WIN")
            return reward + self.GOAL_REWARD, True, True  # Immediate termination on reaching goal

        return reward, terminated, False



def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    os.makedirs("./runs/replay_buffers", exist_ok=True)
    os.makedirs(f"./runs/results/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/trajectories/{args.policy}/seed{args.seed}", exist_ok=True)
    
    CustomGazeboEnv(args, kwargs)
    rospy.spin()

if __name__ == "__main__":
    main()
