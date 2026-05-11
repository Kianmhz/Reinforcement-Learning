"""ROS 2 ↔ Gym bridge.

Owns one rclpy node that:
  - subscribes to /scan (sensor_msgs/LaserScan) and /odom (nav_msgs/Odometry)
  - publishes to /cmd_vel (geometry_msgs/Twist)
  - calls /reset_simulation to reset Gazebo between episodes

The node spins in a background thread so the Gym env can call its methods
synchronously without blocking ROS callbacks.

Phase 1 uses only lidar + odom. Phase 3 will add an /image_raw subscription.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class RosBridge(Node):
    def __init__(self):
        super().__init__("turtlebot_rl_bridge")

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self._reset_sim = self.create_client(Empty, "/reset_simulation")

        self._lock = threading.Lock()
        self._latest_scan: LaserScan | None = None
        self._latest_pose: Pose2D | None = None

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._latest_scan = msg

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._latest_pose = Pose2D(p.x, p.y, yaw)

    def publish_cmd(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        self.publish_cmd(0.0, 0.0)

    def _wait_for(self, getter, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                value = getter()
            if value is not None:
                return value
            time.sleep(0.01)
        return None

    def get_scan(self, timeout: float = 2.0) -> LaserScan | None:
        return self._wait_for(lambda: self._latest_scan, timeout)

    def get_pose(self, timeout: float = 2.0) -> Pose2D | None:
        return self._wait_for(lambda: self._latest_pose, timeout)

    def reset_simulation(self, timeout: float = 5.0) -> bool:
        if not self._reset_sim.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("/reset_simulation service unavailable")
            return False
        future = self._reset_sim.call_async(Empty.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        with self._lock:
            self._latest_scan = None
            self._latest_pose = None
        return future.done()


def start_bridge() -> tuple[RosBridge, SingleThreadedExecutor, threading.Thread]:
    """Start rclpy and spin a RosBridge node in a daemon thread."""
    if not rclpy.ok():
        rclpy.init()
    node = RosBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return node, executor, thread


def shutdown_bridge(
    node: RosBridge, executor: SingleThreadedExecutor, thread: threading.Thread
) -> None:
    executor.shutdown()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    thread.join(timeout=2.0)
