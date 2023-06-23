# ANN_Deep_Leaning_Model
ANN Deep Learning Model For Classification
This code represents a churn prediction model using an Artificial Neural Network (ANN). The model aims to predict customer churn based on various features from a dataset called "Churn_Modelling.csv". The dataset is preprocessed, and the ANN is trained and evaluated using the data.
To run this code, you need to have the following libraries installed:

NumPy
Matplotlib
Pandas
Scikit-learn
TensorFlow
You can install these libraries using pip
pip install numpy matplotlib pandas scikit-learn tensorflow

#Import the necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load the dataset
#!
Churn Prediction Model
This code represents a churn prediction model using an Artificial Neural Network (ANN). The model aims to predict customer churn based on various features from a dataset called "Churn_Modelling.csv". The dataset is preprocessed, and the ANN is trained and evaluated using the data.

Getting Started
To run this code, you need to have the following libraries installed:

NumPy
Matplotlib
Pandas
Scikit-learn
TensorFlow
You can install these libraries using pip:

Copy code
pip install numpy matplotlib pandas scikit-learn tensorflow
Dataset
The dataset used for training and testing the model is stored in a CSV file called "Churn_Modelling.csv". It should be located in the same directory as this script. The dataset contains customer information, including various features and a target variable indicating whether a customer has churned or not.

Instructions
Import the necessary libraries:
python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
Load the dataset:
python
Copy code
dataset = pd.read_csv('Churn_Modelling.csv')
Preprocess the dataset:
