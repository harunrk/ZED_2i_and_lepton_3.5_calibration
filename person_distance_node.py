# This script performs topic subscription to ZED object detection data,
# calculates distances to detected persons, and publishes these distances in ROS2.

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float32MultiArray
from zed_msgs.msg import ObjectsStamped

class PersonDistancesNode(Node):
    def __init__(self):
        super().__init__('person_distances_node')
        self.sub = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.objects_callback,
            10
        )
        self.pub = self.create_publisher(Float32MultiArray, '/person_distances', 10)
        self.get_logger().info('Topic listening')
        self.latest_distances = [] 
        self.latest_positions = []
        self.timer = self.create_timer(3.0, self.publish_distances)

    def objects_callback(self, msg: ObjectsStamped):
        distances = []
        positions = []
        for obj in msg.objects:
            if obj.label.lower() == "person":
                x, y, z = obj.position[0], obj.position[1], obj.position[2]
                dist = math.sqrt(x*x + y*y + z*z)
                distances.append(dist)
                positions.append((x, y, z))

        # en yakından uzağa sırala
        sorted_data = sorted(zip(distances, positions), key=lambda dp: dp[0])
        self.latest_distances = [d for d, _ in sorted_data]
        self.latest_positions = [p for _, p in sorted_data]

    def publish_distances(self):
        if self.latest_distances:
            out = Float32MultiArray()
            out.data = self.latest_distances
            self.pub.publish(out)

            # Log: mesafe + pozisyon
            log_lines = []
            for d, (x, y, z) in zip(self.latest_distances, self.latest_positions):
                log_lines.append(f"Dist={d:.2f}m | Pos=({x:.2f}, {y:.2f}, {z:.2f})")
            self.get_logger().info("People (3s interval):\n" + "\n".join(log_lines))

def main(args=None):
    rclpy.init(args=args)
    node = PersonDistancesNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
