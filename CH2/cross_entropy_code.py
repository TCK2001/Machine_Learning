import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate and prepare data
# Generate a binary classification dataset using sklearn
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally standardize the data (mean=0, variance=1)
# Standardization is not always necessary, but it can help speed up training for models like logistic regression.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 2. Define a simple logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single linear layer
    
    def forward(self, x):
        # Apply sigmoid function to the linear output for binary classification
        return torch.sigmoid(self.linear(x))

# Initialize the model
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# 3. Implement binary cross-entropy loss manually
def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-7  # Small value to avoid log(0)
    loss = - (y_true * torch.log(y_pred + epsilon) + (1 - y_true) * torch.log(1 - y_pred + epsilon))
    return loss.mean()

# 4. Perform a single step of gradient descent manually
learning_rate = 0.1

def gradient_descent_step(model, X, y):
    # Forward pass: calculate predictions
    y_pred = model(X)
    
    # Compute the loss using binary cross-entropy
    loss = binary_cross_entropy(y_pred, y)
    
    # Zero the gradients before backpropagation
    model.zero_grad()
    
    # Backpropagate to compute the gradients
    loss.backward()
    
    # Manually update the model's parameters (weights) using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    
    return loss

# 5. Train the model using gradient descent
num_epochs = 1000
for epoch in range(num_epochs):
    loss = gradient_descent_step(model, X_train, y_train)
    
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Evaluate the model on the test set
with torch.no_grad():
    # Get predictions for the test set
    y_pred = model(X_test)
    
    # Convert probabilities to binary class labels (0 or 1)
    y_pred_cls = y_pred.round()
    
    # Calculate the accuracy
    correct = (y_pred_cls == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
