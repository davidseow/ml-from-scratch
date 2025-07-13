# 1. Import the tools we need
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create a function to plot model predictions
def plot_predictions(X_train, y_train, model, title, x_label="Input", y_label="Output"):
    plt.figure(figsize=(10, 6))

    # Plot training data points
    plt.scatter(X_train, y_train, color='blue', label='Training Data')

    # Generate points for prediction line
    X_line = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
    y_pred = model.predict(X_line)

    # Plot prediction line
    plt.plot(X_line, y_pred, color='red', label='Model Predictions')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

print("--- Starting our Number Predictor ---")

# 2. Prepare our "training data"
# This is what we show the model to teach it patterns.
# We'll use single numbers to predict the next single number.

# Example 1: Simple counting sequence
# If the input is 1, the output is 2
# If the input is 2, the output is 3
# If the input is 3, the output is 4
# ... and so on.

# Our inputs (X) - notice the double square brackets!
# This is how scikit-learn likes its input data: a list of lists.
# Each inner list is one 'example' for the model.
X_train_counting = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Our outputs (y) - the correct answer for each input
y_train_counting = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Example 2: Multiples of 5 sequence
X_train_multiples = np.array([[5], [10], [15], [20], [25], [30], [35], [40], [45], [50]])
y_train_multiples = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])


# 3. Create our model
# We're using a simple Linear Regression model
model = LinearRegression()

print("\n--- Training Model for Counting Sequence ---")
# 4. Train the model (this is where it "learns")
# We tell the model: "Here are your inputs (X) and here are their correct answers (y)."
model.fit(X_train_counting, y_train_counting)
plot_predictions(X_train_counting, y_train_counting, model, "Counting Sequence Model")

# 5. Make a prediction for the counting sequence
print(f"Predicting next number after 11 (Counting): {model.predict(np.array([[11]]))[0]:.0f}")
print(f"Predicting next number after 100 (Counting): {model.predict(np.array([[100]]))[0]:.0f}")
print(f"Predicting next number after 0 (Counting): {model.predict(np.array([[0]]))[0]:.0f}")

print("attempt to break the model")
print(f"Predicting next number after 1.1 (Counting): {model.predict(np.array([[1.1]]))[0]:.0f}")
print(f"Predicting next number after -1 (Counting): {model.predict(np.array([[-1]]))[0]:.0f}")
print(f"Predicting next number after 11*2 (Counting): {model.predict(np.array([[11*2]]))[0]:.0f}")
print(f"Predicting next number after 999 (Counting): {model.predict(np.array([[999]]))[0]:.0f}")



# Now, let's train a new model for the multiples sequence
# For simplicity, we'll create a new model instance for each type of pattern.
# In more complex scenarios, you might use a single model with more varied data.
model_multiples = LinearRegression()
print("\n--- Training Model for Multiples of 5 Sequence ---")
model_multiples.fit(X_train_multiples, y_train_multiples)
plot_predictions(X_train_multiples, y_train_multiples, model_multiples, "Multiples of 5 Sequence Model")

# 6. Make predictions for the multiples sequence
print(f"Predicting next number after 55 (Multiples of 5): {model_multiples.predict(np.array([[55]]))[0]:.0f}")
print(f"Predicting next number after 100 (Multiples of 5): {model_multiples.predict(np.array([[100]]))[0]:.0f}")
print(f"Predicting next number after 7.5 (Multiples of 5): {model_multiples.predict(np.array([[7.5]]))[0]:.0f}")

print("attempt to break the model")
print(f"Predicting next number after 0.5 (Multiples of 5): {model_multiples.predict(np.array([[0.5]]))[0]:.0f}")


print("\n--- Predictions Complete! ---")