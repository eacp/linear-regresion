import pickle # Load saved models
from export import SavedModel
import numpy as np

# Load the model
path = input("Write the path to your .model file: ")

with open(path, "rb") as file:
    model: SavedModel = pickle.load(file)

# Get the Xs from the user
xs = np.zeros(model.size)

for i in range(model.size):
    xs[i] = float(input("Write the sample #%i:" % i))

pred = model.predict(xs)

print("The prediction is", pred)