# Logistic Regression
This method turns data into a probability between 0 and 1 to figure out which category it belongs to. Then, using that probability, it decides the category.

It uses something called Sigmoid to make sure the probability is between 0 and 1. If the probability is above a certain point (called a threshold), the answer will be 1, and if it’s lower, the answer will be 0.

This method is mostly used to solve problems where there are only two possible answers, like yes or no. 

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This is called the logistic function or sigmoid.

Now, let’s learn how this function came about. First, we need to understand something called the Logit function and Odds.   

$$
\text{odds} = \frac{P}{1 - P} = \frac{P(y=1 | x)}{1 - P(y=1 | x)}
$$

The formula means the chance (P) of something happening. For example, in the medical field, P is the chance of having a disease, and 1 - P is the chance of not having the disease.

The formula helps us figure out how much higher the chance of having the disease is compared to not having it.


## Method
![linear regression](images/linear_regression.png)   
In the picture above, the x-values can be anywhere from -infinity to infinity. The y-values also go from -infinity to infinity.

But in logistic regression, while x can still be from -infinity to infinity, the y-value is a probability, so it’s between 0 and 1. We need to change it so the y-value can also go from -infinity to infinity.


$$y = ax + b$$
Here, we change the output y to a probability p. But since y is a probability, we can't just use it like that without adjusting it.  

$$
\text{Odds} = \frac{P}{1 - P} = ax + b
$$
 
Let’s use something called Odds. The Odds have values between 0 and infinity.   

$$
\log_e(\text{Odds}) = \log_e\left(\frac{P}{1 - P}\right) = ax + b
$$

Now, we take the log of this value to change the range from -infinity to infinity.


## how logs work:

If P is `small`, the log gives a `negative` number,   
If P is `0.5`, the log gives `0`,   
If P is `close to 1`, the log gives a `big` number.     

---

Now, we apply "e" (exponential) to both sides of the formula we found earlier.

$$
e^{\log_a\left(\frac{P}{1 - P}\right)} = e^{ax + b}
$$

$$
\frac{P}{1 - P} = e^{ax + b}
$$
 
---   
Next, we take the reciprocal of the formula (flip it upside down).

$$
\frac{1 - P}{P} = \frac{1}{e^{ax + b}}
$$

$$
\frac{1}{P} - 1 = \frac{1}{e^{ax + b}}
$$

---  

To get rid of the -1, we add 1 to both sides of the formula.

$$
\frac{1}{P} = \frac{1}{e^{ax + b}} + 1
$$

$$
\frac{1}{P} = \frac{1 + e^{ax + b}}{e^{ax + b}}
$$
 
---   

Finally, by taking the reciprocal (flipping it) one more time, we get the same result as the sigmoid, or in other words, the logistic function.

$$
P = \frac{e^{ax + b}}{1 + e^{ax + b}}
$$

Wait, they definitely look different, so why are they the same? Let’s look at the formula below to understand.     

$$
f(x) = \frac{1}{1 + e^{-x}} = \frac{e^{x}}{1 + e^{x}}
$$

If we multiply both sides by $e^{x}$, we will get the same value.

---
# Sigmoid
<img src="images/sigmoid.png" alt="Sigmoid" width="500"/>

---

## README
+ [Entropy, Cross entropy, KL divergence](https://github.com/TCK2001/Machine_Learning/blob/main/Logistic_Regression/entropy_all.md)
+ [Logistic Regression + Cross entropy + Gradient descent](https://github.com/TCK2001/Machine_Learning/blob/main/Logistic_Regression/cross_entropy_code.py)


