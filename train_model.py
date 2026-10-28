
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Load prepared data
data = np.load("prepared_data.npz")
X = data["X"]
y = data["y"]

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(21,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])

# 4. Configure training
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

# 5. Train the model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    verbose=1,
)

# 6. Evaluate on data not used for training
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f} stars")


# Compare against a simple average-rating baseline
baseline_prediction = np.full_like(y_test, np.mean(y_train))
baseline_mae = np.mean(np.abs(y_test - baseline_prediction))

print(f"Baseline MAE: {baseline_mae:.4f} stars")
print(f"Neural network MAE: {test_mae:.4f} stars")
print(f"MAE improvement: {baseline_mae - test_mae:.4f} stars")


# 7. Save the trained model
model.save("models/mood_movie_model.keras")
print("Model saved successfully!")
