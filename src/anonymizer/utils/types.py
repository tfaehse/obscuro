"""
Data types for the anonymizer package.
"""


class Detection:
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        object_class: int,
        mask: object | None = None,
    ):
        """
        Detection with relative coordinates (0-1).

        :param x1, y1, x2, y2: Bounding box coordinates in relative format (0-1)
        :param confidence: Detection confidence score
        :param object_class: Class index
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.object_class = object_class
        self.mask = mask

    @property
    def box(self):
        """Return box coordinates as tuple (relative coordinates 0-1)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self):
        """Return box in xywh format (relative coordinates 0-1)."""
        return (
            self.x1,
            self.y1,
            self.x2 - self.x1,
            self.y2 - self.y1,
        )

    def __eq__(self, other):
        """Check equality between detections."""
        if not isinstance(other, Detection):
            return False
        return (
            self.x1 == other.x1
            and self.y1 == other.y1
            and self.x2 == other.x2
            and self.y2 == other.y2
            and self.confidence == other.confidence
            and self.object_class == other.object_class
        )

    def __repr__(self):
        """String representation of detection."""
        return f"Detection(x1={self.x1:.2f}, y1={self.y1:.2f}, x2={self.x2:.2f}, y2={self.y2:.2f}, confidence={self.confidence:.2f}, object_class={self.object_class})"


class Track(Detection):
    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        object_class: int,
        track_id: int,
    ):
        super().__init__(x1, y1, x2, y2, confidence, object_class)
        self.track_id = track_id
