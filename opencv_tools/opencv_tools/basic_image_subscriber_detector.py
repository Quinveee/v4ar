# Basic ROS 2 program to subscribe to real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
   
# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
  
class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
       
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
       
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

  def detect_lines_canny(self, image):
    """
    TASK 2: Basic Canny edge detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    return edges

  import cv2

  def detect_edges_dog(image, sigma1=1, sigma2=2):
      """Simple Difference-of-Gaussians edge detector."""
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Blur with two different Gaussian sigmas
      blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
      blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

      # Difference of Gaussians
      dog = blur1 - blur2

      # Normalize for display
      dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

      # Optional: threshold to get binary edges
      _, edges = cv2.threshold(dog, 20, 255, cv2.THRESH_BINARY)

      return edges
    
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving video frame')
  
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)

    current_frame = self.detect_lines_canny(current_frame)
    # current_frame = self.detect_edges_dog(current_frame)
     
    # Display image
    cv2.imshow("camera", current_frame)
     
    cv2.waitKey(1)

   
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