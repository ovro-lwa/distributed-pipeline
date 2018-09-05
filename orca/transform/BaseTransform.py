from abc import ABC, abstractmethod

class AbstractTransform(ABC):
    def __init__(self, use_scratch, persist):
        self.use_scratch = use_scratch
        self.persist = persist

    @classmethod
    @abstractmethod
    def get_function(cls):
        pass
