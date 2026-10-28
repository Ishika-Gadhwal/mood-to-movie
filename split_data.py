
from sklearn.model_selection import train_test_split
import numpy as np

# Load the prepared data
data = np.load("prepared_data.npz")

X = data["X"]
y = data["y"]

# Keep 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training ratings:", y_train.shape)
print("Testing ratings:", y_test.shape)
