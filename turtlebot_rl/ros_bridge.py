"""ROS 2 ↔ Gym bridge.

Owns one rclpy node that:
  - subscribes to /scan (sensor_msgs/LaserScan) and /odom (nav_msgs/Odometry)
  - publishes to /cmd_vel (geometry_msgs/Twist)
  - calls /reset_simulation to reset Gazebo between episodes

Phase 1 uses only lidar + odom. Phase 3 will add an /image_raw subscription.
"""
raise NotImplementedError("ros_bridge.py will be implemented in Phase 1.")
