from abc import ABC, abstractmethod
from line_msgs.msg import DetectedLines, DetectedLine

class BaseLineSelector(ABC):
    """
    Abstract base class for line selectors.
    Subclasses must implement the `select_line` method.
    """
    @abstractmethod
    def select_line(self, lines: DetectedLines) -> DetectedLine:
        """
        Selects which line to follow from a list of detected lines.
        Returns the selected line as a DetectedLine message.
        """
        pass
