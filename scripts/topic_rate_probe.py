#!/usr/bin/env python3
"""
Measure the real delivery rate and latency of a large ROS2 topic.

The ros2-mcp measurement tools and ``ros2 topic hz`` under-report
multi-megabyte messages on the Jetson (disparity showed 0.6 Hz there while
this probe measured 30 Hz). Use this probe before treating a rate on a
large topic as a defect.

Usage::

    python3 topic_rate_probe.py disparity [reliable|best_effort] [seconds]
    python3 topic_rate_probe.py points [reliable|best_effort] [seconds]
"""

import sys
import time

import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from stereo_msgs.msg import DisparityImage

TOPICS = {
    'disparity': ('/stereo_camera/disparity', DisparityImage),
    'points': ('/stereo_camera/points', PointCloud2),
}


def main():
    """Subscribe for a fixed window and print one RESULT line."""
    kind = sys.argv[1] if len(sys.argv) > 1 else 'disparity'
    reliability = sys.argv[2] if len(sys.argv) > 2 else 'reliable'
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    topic, msg_type = TOPICS[kind]

    rclpy.init()
    node = rclpy.create_node('topic_rate_probe')
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=(ReliabilityPolicy.RELIABLE if reliability == 'reliable'
                     else ReliabilityPolicy.BEST_EFFORT),
    )
    arrivals, latencies_ms, sizes = [], [], []

    def on_msg(msg):
        now = node.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        arrivals.append(now)
        latencies_ms.append((now - stamp) / 1e6)
        sizes.append(len(msg.data) if hasattr(msg, 'data') else 0)

    node.create_subscription(msg_type, topic, on_msg, qos)
    start = time.time()
    while time.time() - start < duration:
        rclpy.spin_once(node, timeout_sec=0.1)

    count = len(arrivals)
    span_s = (arrivals[-1] - arrivals[0]) / 1e9 if count > 1 else 0.0
    rate = (count - 1) / span_s if span_s > 0 else 0.0
    latencies_ms.sort()
    p50 = latencies_ms[count // 2] if count else float('nan')
    size_kb = sorted(sizes)[count // 2] / 1024 if count else 0.0
    print(f'RESULT topic={topic} qos={reliability} msgs={count} '
          f'hz={rate:.2f} lat_p50_ms={p50:.0f} msg_kb={size_kb:.0f}', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
