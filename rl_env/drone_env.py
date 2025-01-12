import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import gymnasium as gym
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gymnasium.spaces import Box
import numpy as np
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty as Emptysrv
from gazebo_msgs.msg import ContactsState
import threading
import time


class SecurityDroneEnv(gym.Env):
    def __init__(self):
        super(SecurityDroneEnv, self).__init__()
        rclpy.init(args=None)
        self.node = Node('security_drone_env_node')
        self.collision_count = 0
        self.collision_msg = None
        self.x_start = 0.0
        self.y_start = 0.0
        self.odometry_msg = None
        self.image = None
        self.bridge = CvBridge()
        self.isFlying = False
        self.height = 360
        self.width = 640
        self.channels = 3
        self.terminated = False
        self.truncated = False
        self.episode_number = 0
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = Box(self.action_low, self.action_high, dtype=np.float32)
        self.observation_space = Box(0, 255, shape=(self.height, self.width, self.channels), dtype=np.uint8)
        self.current_step = 0
        self.ep_length = 1000
        self.spin_thread = threading.Thread(target=self.spin_node, daemon=True) #experimental
        self.spin_thread.start() #experimental
        # Subscribers
        self.camera_sub = self.node.create_subscription(Image, '/simple_drone/front/image_raw', self.camera_callback, 1024)
        self.odometry_sub = self.node.create_subscription(Odometry, '/simple_drone/odom', self.odometry_callback, 1024)
        self.collision_sub = self.node.create_subscription(ContactsState, '/simple_drone/bumper_states', self.collision_callback, 1024)

        # Publishers
        self.land_pub = self.node.create_publisher(Empty, '/simple_drone/land', 10)
        self.takeoff_pub = self.node.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/simple_drone/cmd_vel', 10)

        # Clients
        self.reset_world_client = self.node.create_client(Emptysrv, "reset_world")

    def camera_callback(self, msg: Image):
        self.image = msg

    def odometry_callback(self, msg: Odometry):
        self.odometry_msg = msg

    def collision_callback(self, msg: ContactsState):
        self.collision_msg = msg
    
    def spin_node(self): #experimental
        """Spin the node continuously in a separate thread."""
        rclpy.spin(self.node)

    def detect_collision(self):
        if self.collision_msg is None:
            return 0
        if not self.isFlying:
            return 0
        else:
            if len(self.collision_msg.states) > 0:
                self.collision_count += 1
                return self.collision_count
            else:
                return 0
    
    def distance_from_start(self):
        if self.odometry_msg is None:
            return
        x = self.odometry_msg.pose.pose.position.x
        y = self.odometry_msg.pose.pose.position.y
        distance = np.sqrt((x - self.x_start)**2 + (y - self.y_start)**2)
        self.node.get_logger().info(f"Distance from start: {distance}")
        return distance
    
    def get_reward(self):
        reward = 0
        collision_reward = 0
        distance_reward = self.distance_from_start() * 10
        collisions = self.detect_collision()
        if collisions > 0:
            collision_reward -= 100
            self.terminated = True
        reward = distance_reward + collision_reward
        self.node.get_logger().info(f"Reward: {reward}")
        return reward

    def reset_simulation(self):
        request = Emptysrv.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("service not available, waiting again...")
        self.reset_world_client.call_async(request)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def takeoff(self):
        self.isFlying = True
        self.takeoff_pub.publish(Empty())
        self.node.get_logger().info("Drone taking off")
        time.sleep(3)

    def land(self): 
        self.isFlying = False
        self.land_pub.publish(Empty())
        self.node.get_logger().info("Drone landing")

    def take_action(self, action):
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)
        self.node.get_logger().info("Action taken")

    def get_observation(self):
        if self.image is None:
            self.node.get_logger().info("image not available")
            return np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        convert_img = self.bridge.imgmsg_to_cv2(self.image, 'bgr8')
        image_obs = cv2.normalize(convert_img, None, 0, 255, cv2.NORM_MINMAX)
        self.node.get_logger().info("image available")
        return image_obs

    def reset(self,seed=None, options=None):
        self.land()
        rclpy.spin_once(self.node, timeout_sec=0.1)
        self.episode_number += 1
        self.node.get_logger().info(f"Episode: {self.episode_number}")
        self.reset_simulation()
        self.terminated = False
        self.truncated = False
        self.current_step = 0
        self.collision_count = 0
        self.collision_msg = None
        self.image = None
        if self.odometry_msg is not None:
            self.x_start = self.odometry_msg.pose.pose.position.x
            self.y_start = self.odometry_msg.pose.pose.position.y
        else:
            self.x_start = 0.0
            self.y_start = 0.0
        self.takeoff()
        observation = self.get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        self.take_action(action)
        rclpy.spin_once(self.node, timeout_sec=0.1)
        observation = self.get_observation()
        reward = self.get_reward()
        self.current_step += 1
        if self.current_step >= self.ep_length:
            self.truncated = True
        done = self.terminated
        truncated = self.truncated
        info = {}
        self.node.get_logger().info(f"Step: {self.current_step}")
        return observation, reward, done, truncated, info

        
    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        self.spin_thread.join() #experimental
