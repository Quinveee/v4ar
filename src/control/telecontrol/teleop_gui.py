#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import tkinter as tk
from functools import partial


class RoverTeleop(Node):
    def __init__(self):
        super().__init__('rover_teleop_gui')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Rover Teleop GUI initialized. Publishing to /cmd_vel")
        self.speed_linear = 0.2   # m/s
        self.speed_angular = 0.6  # rad/s

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.publisher.publish(msg)
        self.get_logger().info(f"Published cmd_vel: linear={linear_x:.2f}, angular={angular_z:.2f}")

    def stop(self):
        self.publish_cmd(0.0, 0.0)


def create_gui(node: RoverTeleop):
    window = tk.Tk()
    window.title("UGV Rover Teleop Control")
    window.geometry("300x300")
    window.configure(bg="#2b2b2b")

    btn_cfg = dict(width=10, height=2, fg="white", font=("Arial", 12, "bold"))

    def on_forward(): node.publish_cmd(node.speed_linear, 0.0)
    def on_backward(): node.publish_cmd(-node.speed_linear, 0.0)
    def on_left(): node.publish_cmd(0.0, node.speed_angular)
    def on_right(): node.publish_cmd(0.0, -node.speed_angular)
    def on_stop(): node.stop()
    def on_exit():
        node.stop()
        window.destroy()
        rclpy.shutdown()

    # Layout
    tk.Button(window, text="Forward", command=on_forward, **btn_cfg).pack(pady=10)
    frame = tk.Frame(window, bg="#2b2b2b")
    frame.pack(pady=5)
    tk.Button(frame, text="Left", command=on_left, **btn_cfg).grid(row=0, column=0, padx=5)
    tk.Button(frame, text="Stop", command=on_stop, bg="#f44336", **btn_cfg).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Right", command=on_right, **btn_cfg).grid(row=0, column=2, padx=5)
    tk.Button(window, text="Backward", command=on_backward, **btn_cfg).pack(pady=10)
    tk.Button(window, text="Exit", command=on_exit, bg="#757575", **btn_cfg).pack(pady=10)

    return window


def main(args=None):
    rclpy.init(args=args)
    node = RoverTeleop()
    window = create_gui(node)
    try:
        window.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
