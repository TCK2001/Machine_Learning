import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate data (2D data)
X_train = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [4, 2], [4, 3]])
y_train = np.array([1, 1, 1, -1, -1, -1, -1])

# Initialize the SVC model
svc = SVC(kernel='linear')  # We are using a linear kernel to match the previous implementation

# Train the SVM model
svc.fit(X_train, y_train)

# Function to visualize decision boundary
def visualize_svc(svc, X, y):
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x - b + offset) / w[1]

    fig, ax = plt.subplots()

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='bwr')

    # Get the coefficients and intercept from the trained model
    w = svc.coef_[0]
    b = svc.intercept_[0]

    # Plot the decision boundary (hyperplane)
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # Decision boundary
    x1_1 = get_hyperplane_value(x0_1, w, b, 0)
    x1_2 = get_hyperplane_value(x0_2, w, b, 0)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

    # # Parallel lines (d = 1 for support vectors)
    # x1_1 = get_hyperplane_value(x0_1, w, b, -1)
    # x1_2 = get_hyperplane_value(x0_2, w, b, -1)
    # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

    # x1_1 = get_hyperplane_value(x0_1, w, b, 1)
    # x1_2 = get_hyperplane_value(x0_2, w, b, 1)
    # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k--')

    # Set axis labels and title
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("SVM Decision Boundary Visualization with sklearn SVC")
    plt.legend()
    plt.show()

# Visualize the decision boundary
visualize_svc(svc, X_train, y_train)
