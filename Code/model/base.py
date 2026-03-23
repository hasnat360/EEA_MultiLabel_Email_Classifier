from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def train(self, data) -> None:
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> None:
        ...

    @abstractmethod
    def print_results(self, data) -> None:
        ...

    @abstractmethod
    def data_transform(self) -> None:
        ...
