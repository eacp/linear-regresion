# Linear Regresion implementation with real life example
Implementation of the Linear Regresion AI technique using Python. Testing by using real data set from University of Califofrnia Irvine about wine.

## Linear Regresion

Linear regression is a supervised machine learning algorithm that tries to fit the data you give it into a linear model. A linear model is a 
model of N parameters and an optional bias that are used in conjunction with N x values. All those values are used to compute a variable, wich is the prediction. 
Linear regresion tries to discover (or learn) the aforemention parameters by using gradient descend to minimize a cost funtion. In this case, we are using the 
Mean Square Error.

### Gradient Descend

Gradient descend is a technique to minimize the cost function. It is achieved by updating the parameters until a reach a certain number of rounds (epochs) is finished , we get an acceptable error.

## Objectives

This project has two objectives

- Implement Linear regresion in the source code by using Gradient Descend and Mean Square Error
- Make the implementation reusable
- Apply the implentation to 2 real datasets: red wine and white wine.

## Dataset

The dataset is composed of 1600 records corresponing to chemical properties of Red Wine and 1600 records of White Wine. Each record contains 11 parameters and
a wine quality:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

The dataset also contains a label for wine quality, which goes from 1 to 8. The quality was evalutaed for multiple wine samples by independant wine testers and experts.
This linear regresion attempts to build a model to predict the wine quality based on its chemical properties, and explores if we can find correlations between 
individual independant variables (the parameters) and the dependant variable (the quality).

The dataset was compiled by [The University of California Irvine Center for Machine Learning and Intelligent Systems](https://archive.ics.uci.edu/ml/datasets/wine+quality)
in colaboration with [Paulo Cortez, University of Minho, Guimarães, Portugal](http://www3.dsi.uminho.pt/pcortez/wine/)

## Training and testing

Both Red Wine and White Wine have been split in a training and testing subset. Validation is perfomred with a subset of the training.

### Testing

Execute the `train.py` script. It will prompt you for the training dataset and will ask you if you want to save the model to the disk.

## Model presisntance
The parameters and the scalaning information are serialized and stored using the [Pickle Package](https://docs.python.org/3/library/pickle.html). Models can be loaded
by the `predict.py` script.
