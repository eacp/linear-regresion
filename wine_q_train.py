"""
Eduardo Castillo
A01702948

Intelligent Systems (Group 1)
Intelligent Systems Technologies (G1)

Linear Regression implementation
for Hand In

"""

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler
from typing import List
from export import SavedModel

err = []

# Hypothesis function, which is also the dot product
def h(params: npt.ArrayLike, sample: npt.ArrayLike) -> float:
	"""
    Evaluate the hypothesys funtion,
    Which is actually a dot product
	"""
	
	return np.dot(params, sample)


# For plotting purposes
__errors__ = []

def register_mse(params: List[float] , samples,ys: List[float]):
	# This one is for plotting the errors
	global __errors__
	acc = 0
#	print("transposed samples") 
#	print(samples)
	for i in range(len(samples)):
		hyp = h(params,samples[i])
		error=hyp-ys[i]
		acc=+error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param=acc/len(samples)
	__errors__.append(mean_error_param)

def GD(params: npt.ArrayLike, samples, y: npt.ArrayLike, alfa: float = 0.1) -> npt.ArrayLike:

	new_params = np.array(params)

	for j in range(len(params)):
		acc =0; 
		for i in range(len(samples)):
			error = h(params,samples[i]) - y[i]

			#Sumatory part of the Gradient Descent formula for linear Regression.
			# Just like the formula

			acc += error*samples[i][j]  

		# Continuation of the formula
		new_params[j] = params[j] - alfa*(1/len(samples))*acc  
		
	return new_params

# Load data from csv

ds_path = input("Where is the CSV file?")
headers = input("Does the CSV has headers? [y/n]") == "y"
delim = input("What is the delimiter of the CSV? (default ',')") or ","

print("Loading the dataset from", ds_path)

csv_ds = np.genfromtxt(
    ds_path,
    delimiter=delim,
	skip_header= 1 if headers else 0
)

print("Loaded dataset")

print(csv_ds)

epochs = int(input("Max number of epochs (default 100): ") or 100)

print("Using %i epochs" % epochs)

# Extract data and discard the rest
ys = csv_ds[:,-1] # this is magic if you ask me, and i question its legality

print("This are the values of the dependant variable (y)", ys)

# Remove the y column, we already have this info on a separate array
xs = np.delete(csv_ds, -1, 1)

print("This are the UNSCALED samples")
print(xs)

# Now scale the values
scaler = MinMaxScaler()
scaler.fit(xs)
scaled = scaler.transform(xs)

print("Scaled values")
print(scaled)

alfa = float(input("Choose an alfa value (default 0.1)") or 0.1)

# This is like in class now

ep = 0

params = np.zeros(11)

while ep < epochs:  #  run gradient descent until local minima is reached
	oldparams = list(params)

	params=GD(params, scaled,ys,alfa)	

	# register the MSE for later analysis and visualisation
	register_mse(params, xs, ys)  
	# print (params)

	ep += 1

# Show the computed params
print("This are the resulting params from the training: ", params)

import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)
plt.plot(__errors__)
plt.show()

# proceed to export the model
model = SavedModel(params, scaler)

if input("Want to save the model? y/n") == "y":
	export_name = input("Please name the export file") + ".model"

	model.save(export_name)

wants_to_test = input("Would you like to test? y/n") == "y"

if not wants_to_test:
	exit()

test_path = input("Write the path to the test csv")

test_csv = np.genfromtxt(
    test_path,
    delimiter=delim,
	skip_header= 1 if headers else 0
)

print(test_csv)

mse = 0.0

total = 0

for row in test_csv:
	print("Testing", row)

	sample = row[:-1] # All but last
	y = row[-1]

	p = model.predict(sample)

	print("Expected: %i Got: %i" % (y, p))

	mse += (y-p)**2

	total += 1

mse /= total

print("MSE:", mse)
