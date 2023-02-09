from abc import ABC, abstractmethod


class Orchestrator(ABC):
    """
    Builds data preprocessing and model training pipeline.
    """
    @abstractmethod
    def run(self) -> float:
        """
        Main method, runs the pipeline of data and model. Should be implemented in all class childs.
        """
        pass

