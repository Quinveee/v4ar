#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time

class LedTest(Node):
    def __init__(self):
        super().__init__('led_test_node')
        self.led_pub = self.create_publisher(Float32MultiArray, '/ugv/led_ctrl', 10)
        self.get_logger().info("LED test node started")

    def turn_leds(self, on=True):
        msg = Float32MultiArray()
        msg.data = [9.0, 0.0] if on else [0.0, 0.0]
        self.led_pub.publish(msg)
        self.get_logger().info(f"LEDs {'ON' if on else 'OFF'}")

def main(args=None):
    rclpy.init(args=args)
    node = LedTest()

    node.turn_leds(True)   # Turn LEDs ON
    time.sleep(2)          # Wait 2 seconds
    node.turn_leds(False)  # Turn LEDs OFF

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
