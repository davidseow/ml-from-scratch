# 1. Import the tools we need
import numpy as np
from sklearn.linear_model import LinearRegression

print("--- Starting our Advanced Number Predictor ---")

# 2. Prepare our "training data"
# For each sequence, we'll extract features:
# 1. The last known number in the sequence.
# 2. The common difference (or step) found in the sequence.

# Function to calculate the common difference from a list of numbers
def calculate_common_difference(sequence):
    if len(sequence) < 2:
        return 0 # Or handle as an error, for simplicity let's return 0

    # Calculate differences between consecutive numbers
    diffs = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

    # If all differences are close enough, assume it's an arithmetic progression
    # We use np.isclose for floating point comparisons
    if all(np.isclose(d, diffs[0]) for d in diffs):
        return diffs[0]
    else:
        # If differences vary, this simple model might struggle.
        # For now, let's just use the last difference for prediction
        # or a more sophisticated approach for non-linear patterns.
        # For this simple linear model, we assume a constant difference.
        # Let's return the average difference as a simple strategy for now
        return np.mean(diffs)


# Our training examples (input_sequence, expected_output)
training_examples = [
    ([0.5, 0.9], 1.3),
    ([-0.1, 0], 0.1),
    ([0, -1], -2),
    ([-0.5, -0.25, 0], 0.25),
    ([0.5, 1, 1.5], 2)
]

# Now, let's convert these examples into X (features) and y (target)
X_train = [] # This will store [last_number_in_sequence, common_difference]
y_train = [] # This will store the next number

for sequence, next_number in training_examples:
    last_num = sequence[-1] # Get the last number in the sequence
    common_diff = calculate_common_difference(sequence) # Calculate the common difference
    X_train.append([last_num, common_diff])
    y_train.append(next_number)

# Convert lists to NumPy arrays, which scikit-learn prefers
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training Features (X_train):\n{X_train}")
print(f"Target Values (y_train):\n{y_train}")

# 3. Create our model
# We're still using Linear Regression, but now it learns from two features.
model = LinearRegression()

print("\n--- Training the Model ---")
# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
print("\n--- Making Predictions ---")

# Function to prepare a new input sequence for prediction
def prepare_input_for_prediction(sequence):
    last_num = sequence[-1]
    common_diff = calculate_common_difference(sequence)
    # Return as a 2D array, as model expects list of lists (or 2D array)
    return np.array([[last_num, common_diff]])

# Let's test with the original examples and some new ones
test_sequences = [
    ([0.5, 0.9]),
    ([0.8, 1.0, 1.2]),
    ([-0.1, 0]),
    ([0, -1]),
    ([-0.5, -0.25, 0]),
    ([0.5, 1, 1.5]),
    # New test cases
    ([10, 20, 30]),      # Expect 40 (difference 10)
    ([1, 1.5, 2]),       # Expect 2.5 (difference 0.5)
    ([5, 3]),            # Expect 1 (difference -2)
    ([100]),             # Edge case: single number. Common diff will be 0.
    ([1, 2, 4])          # This has varying differences (1 then 2).
                         # Our simple model will use the average difference,
                         # so it might not be perfect for non-linear patterns.
]

for seq in test_sequences:
    input_features = prepare_input_for_prediction(seq)
    predicted_next = model.predict(input_features)[0]
    # We'll round to 2 decimal places for cleaner output if floats are involved
    print(f"Input: {seq} -> Predicted: {predicted_next:.2f}")

print("\n--- Predictions Complete! ---")