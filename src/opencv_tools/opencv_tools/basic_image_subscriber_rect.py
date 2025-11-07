# Basic ROS 2 program to subscribe to real-time streaming 
# video from compressed rectified camera topic
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
   
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import CompressedImage # Changed to CompressedImage
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np # Added for compressed image handling
  
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):  # Fixed: double underscores
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')  # Fixed: double underscores
       
    # Create the subscriber for compressed rectified image
    self.subscription = self.create_subscription(
      CompressedImage,  # Changed message type
      '/image_rect/compressed',  # Changed topic
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
       
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
    
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    # self.get_logger().info('Receiving video frame')
  
    # Convert compressed ROS Image message to OpenCV image
    np_arr = np.frombuffer(data.data, np.uint8)
    current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if current_frame is not None:
      # Display image
      cv2.imshow("camera", current_frame)
      cv2.waitKey(1)
    else:
      self.get_logger().error('Failed to decode compressed image')
   
def main(args=None):
   
  # Initialize the rclpy library
  rclpy.init(args=args)
   
  # Create the node
  image_subscriber = ImageSubscriber()
   
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
   
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
   
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()