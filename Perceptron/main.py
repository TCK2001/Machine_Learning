from model import Perceptron
from Visualization_Point import visualization_base
from Visualization_training_error import visualization_error
from Visualization_boundary import visualization_region

import numpy as np    
import pandas as pd

# load iris data
data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(data, header = None, encoding = 'utf-8') 

# Select the input values and labels
X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

user_input = input("""
                   Select the number to visualization the result.
                   1. Only visualize the points.
                   2. Visualize the training errors.
                   3. Visualization with the boundary lines.
""")

# call model
ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

if user_input == "1":
    visualization_base(X, y)
elif user_input == "2":
    visualization_error(X, y, ppn)
elif user_input == "3":
    visualization_region(X, y, classifier = ppn)
    
