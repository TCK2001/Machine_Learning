import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert class labels to -1 and 1

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Optimization via gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # Only update the regularization term when condition is satisfied (no hinge loss)
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Update weights and bias if hinge loss occurs
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        # Return the predicted class (-1 or 1) using the sign of w^T x + b
        # np.sign => if value > 0 return 1 else return -1
        return np.sign(np.dot(X, self.w) + self.b)

    # Function to visualize decision boundary and support vectors
    def visualize(self, X, y):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x - b + offset) / w[1]

        fig, ax = plt.subplots()

        # Plot the data points
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='bwr')

        # Plot the decision boundary (hyperplane)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        # Decision boundary
        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)
        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

        # # Parallel lines (d = 1 for support vectors)
        # x1_1 = get_hyperplane_value(x0_1, self.w, self.b, -1)
        # x1_2 = get_hyperplane_value(x0_2, self.w, self.b, -1)
        # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

        # x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 1)
        # x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 1)
        # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

        # Set axis labels and title
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("SVM Decision Boundary Visualization")
        plt.show()

# Generate data (2D data)
X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [4, 2], [4, 3]])
y_train = np.array([1, 1, 1, -1, -1, -1, -1])

# Train the SVM model
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Visualize the decision boundary
svm.visualize(X_train, y_train)
