#  main.py
```python
data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(data, header = None, encoding = 'utf-8')
```
`data` : Load the iris data.  
`df` : Read the iris data using pandas.

---

```python
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
```
|   | 0   | 1   | 2   | 3   | 4           |
|---|-----|-----|-----|-----|-------------|
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
...
   
You can see in the table above that the index 4 represents the iris label.
if the iris label is Iris-setosa set it to -1 otherwise, set it to 1.

---

```python
X = df.iloc[0:100, [0, 2]].values
```
+ Index 0 : Sepal.Length <----- selected
+ Index 1 : Sepal.Width 
+ Index 2 : Petal.Length <----- selected
+ Index 3 : Petal.Width

## Model Readme & Code
* Readme   
[model.py](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/model_readme.md)   
[Visualization_Point.py](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/visualization_point_readme.md)   
[Visualization_boundary.py](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/visualization_boundary_readme.md)
* Code
[model](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/model.py)
[visual : point](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/visualization_point_readme.md)
[visual : training error](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/Visualization_training_error.py)
[visual : boundary](https://github.com/TCK2001/Machine_Learning/blob/main/Perceptron/Visualization_boundary.py)

