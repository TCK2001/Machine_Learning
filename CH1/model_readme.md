# model.py
`class Perceptron()` : In Python 2.x you have to write `object` in the class definition (e.g., class Perceptron(object)), but in Python 3.x, it's not needed.

---
## Explanation of the Function.

```python
def __init__(self, eta = 0.01, n_iter = 50, random_state = 42):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
```
`eta` : learning_rate.   
`n_iter` : iteration、epoch.  
`random_state` : Fixing the output result.

---

```python
def fit(self, X, y):
    rgen = np.random.RandomState(self.random_state)
    self.w = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
    self.errors = []
        ...
```
`rgen` : Fixing the numpy output result.   
`w` : Generate the random values with a mean of `0` and a standard deviation of `0.01`.   
`size = 1 + X.shape[1]` : weight count and bias.  
`errors` : Save each error value for visualization.

---

```python
for _ in range(self.n_iter):
    errors = 0
    for input, target in zip(X, y):
        update = self.eta * (target - self.predict(input))
        self.w[1:] += update * input
        self.w[0] += update
        errors += int(update != 0.0)
    self.errors.append(errors)
return self
```
+ <b> w[0] : bias   
+ w[1:] : weight </b>
     
<img src = "https://git.io/JtIbL" width = "550px" height = "300px" ></img> 
<img src = "https://git.io/JtIbO" width = "630px" height = "250px" ></img> 

---  
## Perceptron training method
$$
\Delta w_j = \eta\left(y^{(i)} - \hat{y}^{(i)}\right)x_j^{(i)}
$$

$$
w_j := w_j + \Delta w_j
$$

---

### If the model predict correct  
$$
y^{(i)} = -1, \quad \hat{y}^{(i)} = -1, \quad \Delta w_j = \eta(-1 - (-1))x_j^{(i)} = 0
$$ 

$$
y^{(i)} = 1, \quad \hat{y}^{(i)} = 1, \quad \Delta w_j = \eta(1 - 1)x_j^{(i)} = 0
$$

---

### If the model predict incorrect  
$$
y^{(i)} = 1, \; \hat{y}^{(i)} = -1, \qquad \Delta w_j = \eta(1 - (-1))x_j^{(i)} = \eta(2)x_j^{(i)}
$$

$$
y^{(i)} = -1, \; \hat{y}^{(i)} = 1, \qquad \Delta w_j = \eta(-1 - 1)x_j^{(i)} = \eta(-2)x_j^{(i)}
$$

---


```python
def net_input(self, X):
    return np.dot(X, self.w[1:]) + self.w[0]
```
Multiply the input X by the weight and then add the bias to predict the result.

---

```python
def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, -1)
```
Classifier the result as 1 if it is greater than 0, otherwirse classifiy it as -1.