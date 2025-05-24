from enum import Enum


class EventTypeDesc:
    """Represents the description of an event type.

    Attributes:
        name (str): The name of the event type.
        order (int): The order of the event type in a path.
    """
    name: str
    order: int

    def __init__(self, name: str, order: int):
        """Initializes the EventTypeDesc with a name and order."""
        self.name = name
        self.order = order


class EventType(Enum):
    """Enum representing different types of events."""
    RAW = EventTypeDesc("raw", 2)
    PATH_START = EventTypeDesc("path_start", 0)
    PATH_END = EventTypeDesc("path_end", 5)
    SESSION_START = EventTypeDesc("session_start", 1)
    SESSION_END = EventTypeDesc("session_end", 3)
