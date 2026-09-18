from abc import ABC, abstractmethod

class abstractPin(ABC):
    def __init__(self, pin):
        self.pin = pin

    def set(self, value):
        raise NotImplementedError("Subclasses should implement this!")

    def get(self):
        raise NotImplementedError("Subclasses should implement this!")