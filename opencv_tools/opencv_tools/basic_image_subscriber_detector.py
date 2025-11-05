# Basic ROS 2 program to subscribe to real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# Import the necessary libraries
from email.mime import image
import rclpy  # Python library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library


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
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def detect_lines_canny(self, image):
        """
        TASK 2: Basic Canny edge detection
        """
        # Convert to grayscale
        original = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges,
                                rho=1,              # distance resolution in pixels
                                theta=np.pi/180,    # angle resolution in radians
                                threshold=80,       # min number of votes
                                minLineLength=50,   # shortest line to detect
                                maxLineGap=10)      # max allowed gap between line segments

        # Draw lines on a copy of the original image
        line_img = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return cv2.addWeighted(original, 0.8, line_img, 1, 0)

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
        dog = cv2.normalize(dog, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

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
