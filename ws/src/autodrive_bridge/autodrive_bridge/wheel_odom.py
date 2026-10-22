"""Sensor-derived forward velocity for the RoboRacer AutoDRIVE simulator.

Adapted from the user's previous synchronized encoder odometry. No simulator
position/velocity or differential-drive yaw is used.
"""
import math

import message_filters
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


class WheelOdometry(Node):
    def __init__(self):
        super().__init__('autodrive_wheel_odometry')
        self.declare_parameter('meters_per_encoder_unit', 0.0589022388631801)
        self.scale = float(self.get_parameter('meters_per_encoder_unit').value)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid encoder calibration')

        self.previous = None
        self.publisher = self.create_publisher(Odometry, '/wheel/odom', 10)
        ns = '/autodrive/roboracer_1'
        self.left = message_filters.Subscriber(self, JointState, ns + '/left_encoder')
        self.right = message_filters.Subscriber(self, JointState, ns + '/right_encoder')
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.left, self.right], queue_size=20, slop=0.02
        )
        self.synchronizer.registerCallback(self.update)
        self.get_logger().info(
            f'Waiting for synchronized wheel encoders; scale={self.scale:.8f} m/unit.'
        )

    def update(self, left, right):
        if not left.position or not right.position:
            return
        le, re = float(left.position[0]), float(right.position[0])
        if not (math.isfinite(le) and math.isfinite(re)):
            return

        left_ns = left.header.stamp.sec * 10**9 + left.header.stamp.nanosec
        right_ns = right.header.stamp.sec * 10**9 + right.header.stamp.nanosec
        stamp_ns = max(left_ns, right_ns)
        current = (stamp_ns, le, re)
        if self.previous is None:
            self.previous = current
            self.get_logger().info('Encoder baseline received.')
            return

        prev_ns, prev_le, prev_re = self.previous
        dt = (stamp_ns - prev_ns) / 1e9
        if dt <= 0:
            return
        self.previous = current
        if dt > 1.0:
            self.get_logger().warning('Encoder gap: rebasing.')
            return

        dl, dr = (le - prev_le) * self.scale, (re - prev_re) * self.scale
        if max(abs(dl), abs(dr)) > 25.0 * dt + 0.05:
            self.get_logger().warning('Encoder jump: sample skipped.')
            return

        v = (dl + dr) / (2.0 * dt)
        if not math.isfinite(v):
            return

        msg = Odometry()
        msg.header.stamp.sec = stamp_ns // 10**9
        msg.header.stamp.nanosec = stamp_ns % 10**9
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = v
        # Only linear.x is measured. EKF must fuse *only* this field.
        for i in range(6):
            msg.pose.covariance[7 * i] = 1e6
            msg.twist.covariance[7 * i] = 1e6
        msg.twist.covariance[0] = 0.04  # Provisional variance (m/s)^2.
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
