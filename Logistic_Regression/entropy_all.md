# Entropy
Before understanding cross-entropy, it's important to know what entropy is.
Entropy, simply put, represents uncertainty. If the entropy is high, it means that it’s harder to predict the data.  

$$
H(x) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

### Let's look at a simple example: rolling a die vs. flipping a coin.   
Try to predict the probability of getting heads or tails when flipping a coin!   

$$ 
H(x) = - \left( \frac{1}{2} \log \frac{1}{2} + \frac{1}{2} \log \frac{1}{2} \right)
$$

The result of the calculation is approximately 0.693. Here, the $log$ refers to the natural logarithm $\log_{exponential}$.   

---

Now, let's calculate the probability for each of the 6 sides of a die when rolled!
$$ H(x) = - \left( \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} + \frac{1}{6} \log \frac{1}{6} \right)
$$

The result of the calculation is approximately 1.79.

As we can see, the entropy value for the die is higher than for the coin.
This means that predicting the result of a die roll is much harder than predicting the result of a coin flip.  

❗❗ In conclusion, in $log x$, the smaller the value of $x$, the smaller the $log x$ value. However, since a minus sign is applied at the end, the final result becomes larger.

---

# Cross entropy
The main purpose of cross-entropy is to predict the distribution $𝑞$ In other words, it checks how similar the predicted distribution 
$p$ is to the actual distribution $q$   .

$$
H_p(q) = -\sum_{i=1}^{n} q(x_i) \log p(x_i)
$$

For example, let’s say there are `red`, `green`, and `blue` balls in a box in the ratio of `0.8/0.1/0.1`.   
However, User `A's` intuition suggests the ratio might be `0.2/0.2/0.6`,   
while User `B's` intuition suggests the ratio might be `0.7/0.1/0.2`.   
The cross-entropy results for these predictions are as follows:

$Entropy :$
$$
H(q) = -[0.8 \log(0.8) + 0.1 \log(0.1) + 0.1 \log(0.1)] = 0.63
$$

## A user
$Cross-entropy :$
$$
H_p(q) = -[0.8 \log(0.2) + 0.1 \log(0.2) + 0.1 \log(0.6)] = 1.50
$$

## B user
$Cross-entropy :$
$$
H_p(q) = -[0.8 \log(0.7) + 0.1 \log(0.1) + 0.1 \log(0.2)] = 0.68
$$

As we can see, User B's prediction is very close to the correct answer.

# Kullback-Leibler Divergence
Now that we understand `entropy` and `cross-entropy`, it's important to know about `KL-divergence` as well !   
This concept is very similar to cross-entropy; it is used to measure the difference between two different distributions.

$$
D_{KL}(p \parallel q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

Here, $q(x)$ is always fixed, and $q(x)$ is always greater than or equal to $p(x)$. This is because our main objective is to make the distribution $p(x)$ as close as possible to $q(x)$. In other words, the closer the distribution $p(x)$ is to $q(x)$, the closer the KL-divergence value will be to 0.


### Reference
[Entropy, Cross-entropy. KL-divergence](https://velog.io/@rcchun/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%81%AC%EB%A1%9C%EC%8A%A4-%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BCcross-entropy)
