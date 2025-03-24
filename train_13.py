from geometry_msgs.msg import Twist, Pose, Vector3
from nav_msgs.msg import Odometry
from train import GazeboEnv
import tf.transformations
import numpy as np
import rospy
import os

from config import parse_args

class RealEnv(GazeboEnv):

    '''def _initialize_ros(self):
        """Initialize ROS node and publishers/subscribers"""
        # Initialize ROS node and publishers
        rospy.init_node('robot_trainer', anonymous=True)                                    # Initialize ROS node
        self.cmd_vel_pub = rospy.Publisher('/turtlebot_13/cmd_wheels', Vector3, queue_size=1)                 # Initialize velocity publisher
        
        # Initialize odometry subscriber
        rospy.Subscriber('/turtlebot_13/odom', Odometry, self.callback, queue_size=1)                    # Initialize odometry subscriber
        rospy.loginfo("ROS initialization completed")

        print("ENV INIT...")'''
    
    def _init_parameters(self, args):
        super()._init_parameters(args)

        self.train_flag = False
        self.evaluate_flag = False
        self.come_flag = True

    def yaw_from_quaternion(self, q):
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        return np.arctan2(siny_cosp, cosy_cosp)
    
    def homogeneous_transfomration(self, vector):
        H = np.array([[0, 1, 0],
                      [-1, 0, -1],
                      [0, 0, 1]])


        vec_hom = np.append(vector, 1)
        transformed_vec = H @ vec_hom

        return transformed_vec[0], transformed_vec[1]

    def odom(self):
        """Extract state information from odometry message"""
        # Robot position
        x = self.msg.pose.pose.position.x
        y = self.msg.pose.pose.position.y
        self.x, self.y = x, y
        # Get orientation
        quaternion = (
            self.msg.pose.pose.orientation.x,
            self.msg.pose.pose.orientation.y,
            self.msg.pose.pose.orientation.z,
            self.msg.pose.pose.orientation.w
        )
        x , y = self.homogeneous_transfomration([x, y])
        yaw = self.yaw_from_quaternion(quaternion) + 2.8381249
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        self.theta = yaw
        
        # Robot velocities
        linear_vel = self.msg.twist.twist.linear.x
        vel_x = linear_vel * np.cos(yaw)
        vel_y = linear_vel * np.sin(yaw)
        angular_vel = self.msg.twist.twist.angular.z
        
        #self.state = np.array([x, y, yaw, vel_x, vel_y, angular_vel])

        # Compute distance to goal
        dx = self.GOAL[0] - x
        dy = self.GOAL[1] - y
        distance = np.linalg.norm([dx, dy])
        
        # Compute the angle from the robot to the goal
        goal_angle = np.arctan2(dy, dx)
        
        # Compute the relative heading error (normalize to [-pi, pi])
        e_theta_g = (yaw - goal_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Compute speed in the direction toward the goal (projection of velocity onto goal direction)
        v_g = vel_x * np.cos(goal_angle) + vel_y * np.sin(goal_angle)
        
        # Compute lateral (sideways) velocity (component perpendicular to the goal direction)
        v_perp = -vel_x * np.sin(goal_angle) + vel_y * np.cos(goal_angle)

        # Compute distance to obstacle (assuming obstacle is at the origin)
        d_obs = np.linalg.norm([x, y])
        
        # Create the processed state vector
        self.state = np.array([distance, e_theta_g, v_g, v_perp, angular_vel, d_obs])

    def publish_velocity(self, action):
        """Publish velocity commands to the robot"""
        v = action[0] * self.MAX_VEL[0]
        w = action[1] * self.MAX_VEL[1]
        
        d = 0.173
        r = 0.0325

        w_r = (v + w * d/2) / r
        w_l = (v - w * d/2) / r
        vel_msg = Vector3(w_r, w_l, 0)

        self.cmd_vel_pub.publish(vel_msg)

    def reset(self):
        """Reset the environment"""
        self.publish_velocity([0, 0])
        rospy.sleep(0.5)
        # Change initila position
        r = np.sqrt(np.random.uniform(0,1))*0.1
        theta = np.random.uniform(0,2*np.pi)
        self.HOME = np.array([-1 + r * np.cos(theta), 0 + r * np.sin(theta)])
    
    def train(self):
        """Train function"""
        if self.count == 0:
            self.episode_time = rospy.get_time()

        if self.timestep > 1e3:
            action = self.policy.select_action(self.state)
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)
                        ).clip(-self.max_action, self.max_action)
        else:
            action = np.random.uniform(-self.max_action, self.max_action,size=self.action_dim)

        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        if self.timestep > 1e3:
            self.policy.train(self.replay_buffer, batch_size=self.batch_size)
            rospy.sleep(self.TIME_DELTA)

        reward, done, target = self.get_reward()
        self.episode_reward += reward

        '''elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True'''
        
        if self.count > self.max_count:
            done = True

        if self.old_state is not None:
            self.replay_buffer.add(self.old_state, self.old_action, self.state, reward, float(done))

        # Update state and action
        self.old_state = None if done else self.state
        self.old_action = None if done else action
        self.episode_timesteps += 1
        self.timestep += 1
        self.count += 1

        if done:
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Episode: {self.episode_num} - Reward: {self.episode_reward:.1f} - Steps: {self.episode_timesteps} - Target: {target} - Expl Noise: {self.expl_noise:.3f} - Time: {self.episode_time:.1f} sec")
            self.reset()
            if self.expl_noise > 0.1:
                self.expl_noise = self.expl_noise - ((0.3 - 0.1) / 300)

            self.train_flag = False
            self.evaluate_flag = False
            self.come_flag = True
            
            self.episode_reward = 0
            self.episode_timesteps = 0
            self.count = 0
            self.episode_num += 1

            if ((self.episode_num - 1) % self.eval_freq) == 0:
                print("-" * 80)
                print(f"VALIDATING - EPOCH {self.epoch + 1}")
                print("-" * 80)

    def evaluate(self):
        self.trajectory.append([self.x, self.y])

        if self.count == 0:
            self.episode_time = rospy.get_time()

        action = self.policy.select_action(self.state) if self.expl_noise != 0 else self.policy.select_action(self.state, True)
        a_in = [(action[0] + 1 ) / 2, action[1]]
        self.publish_velocity(a_in)

        reward, done, target = self.get_reward()
        self.avrg_reward += reward

        '''elapsed_time = rospy.get_time() - self.episode_time
        if elapsed_time > self.max_time:
            done = True'''
        
        if self.count > self.max_count:
            done = True

        self.count += 1
        self.old_state = None if done else self.state
        self.old_action = None if done else action

        if done:
            self.suc += int(target)      # Increment suc if target is True (1)
            self.col += int(not target)  # Increment col if target is False (0)
            self.episode_time = rospy.get_time() - self.episode_time
            print(f"Evaluation: {self.e + 1} - Average Reward: {self.avrg_reward / (self.e + 1):.1f} - Target: {target} - Time: {self.episode_time:.1f} sec")
            
            self.all_trajectories.append(np.array(self.trajectory))
            self.trajectory = []
            self.e += 1
            self.count = 0
            self.reset()

            self.train_flag = False
            self.evaluate_flag = False
            self.come_flag = True

            if self.e >= self.eval_ep:
                self.avrg_reward /= self.eval_ep
                avrg_col = self.col / self.eval_ep
                avrg_suc = self.suc / self.eval_ep

                print("-" * 50)
                print(f"Average Reward: {self.avrg_reward:.2f} - Collisions: {avrg_col*100} % - Successes: {avrg_suc*100} % - TIME UP: {(1-avrg_col-avrg_suc)*100:.0f} %")
                print("-" * 50)

                self.evaluations_reward.append(self.avrg_reward)
                self.evaluations_suc.append(avrg_suc)
                np.savez(f"./runs/trajectories/{self.args.policy}/seed{self.args.seed}/{self.epoch}_trajectories.npz", **{f"traj{idx}": traj for idx, traj in enumerate(self.all_trajectories)})
                np.save(f"./runs/results/{self.args.policy}/evaluations_reward_seed{self.args.seed}", self.evaluations_reward)
                np.save(f"./runs/results/{self.args.policy}/evaluations_suc_seed{self.args.seed}", self.evaluations_suc)
                self.policy.save(f"./runs/models/{self.args.policy}/seed{self.args.seed}/{self.epoch}")


                self.all_trajectories = []
                self.avrg_reward = 0
                self.suc = 0
                self.col = 0
                self. e = 0
                self.epoch +=1

    def come(self):
        """Come home function"""
        if self.rotation_flag:
            angle_to_goal = np.arctan2(self.HOME[1] - self.y, self.HOME[0] - self.x)
            
            if abs(angle_to_goal - self.theta) > 0.05:
                angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)
                self.publish_velocity([0, angular_speed])
            else:
                self.publish_velocity([0,0])
                self.move_flag = True
                self.rotation_flag = False

        elif self.move_flag:
            # Compute distance and angle to the goal
            distance = np.sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)
            angle_to_goal = np.arctan2(self.HOME[1] - self.y, self.HOME[0] - self.x)

            # Proportional controller with increased speed
            linear_speed = min(0.5 * distance, 0.5)  # Increased max speed
            angular_speed = min(2.0 * (angle_to_goal - self.theta), 2.0)  # Increased max angular speed
                
            if distance < 0.05:  # Stop condition
                self.publish_velocity([0, 0])
                self.move_flag = False
                self.rotation_flag = True

                self.train_flag = True
                self.evaluate_flag = False
                self.come_flag = False

                if ((self.episode_num - 1) % self.eval_freq) == 0:
                    self.train_flag = False
                    self.evaluate_flag = True
                    self.come_flag = False

                    if self.e >= self.eval_ep:
                        self.train_flag = True
                        self.evaluate_flag = False
                        self.come_flag = False
            else:
                self.publish_velocity([linear_speed, angular_speed])
        else:
            pass


def main():
    print("\nRUNNING MAIN...")
    args, kwargs = parse_args()
    
    os.makedirs("./runs/replay_buffers", exist_ok=True)
    os.makedirs(f"./runs/results/{args.policy}", exist_ok=True)
    os.makedirs(f"./runs/models/{args.policy}/seed{args.seed}", exist_ok=True)
    os.makedirs(f"./runs/trajectories/{args.policy}/seed{args.seed}", exist_ok=True)
    
    RealEnv(args, kwargs)
    rospy.spin()

if __name__ == "__main__":
    main()
