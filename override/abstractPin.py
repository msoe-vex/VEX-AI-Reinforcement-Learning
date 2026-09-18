from abc import ABC, abstractmethod

class abstractPin(ABC):
    @abstractmethod
    def __init__(self, cup1 = None, cup2 = None, top = True):
        pass

