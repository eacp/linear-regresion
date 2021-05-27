from dataclasses import dataclass
from typing import List
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler
import pickle

@dataclass
class SavedModel:
    # The params
    params: List[float]

    # The scaling that we used
    scaler: MinMaxScaler

    def save(self, path: str):
        """
        Method to picke and save to a file
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)
    
    @property
    def size(self):
        return len(self.params)

    def predict(self, xs: npt.ArrayLike) -> float:
        # It is a hypothesis, so you just do the same as in the h func

        # you have to transform everything fisrt

        scaled = self.scaler.transform(xs.reshape(1, -1))
        # For some reason it returns a list of one number
        return np.dot(self.params, np.transpose(scaled))[0]