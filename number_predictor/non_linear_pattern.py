# 1. Import the tools we need
import numpy as np
# This is our new model type! Multi-layer Perceptron Regressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split # To split data for better testing
from sklearn.preprocessing import StandardScaler # To scale data, important for MLPs
import warnings

# Suppress specific warnings from sklearn about convergence
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

print("--- Starting our Non-Linear Number Predictor ---")

# Define the maximum length of our input sequences
# All input sequences will be padded/truncated to this length.
# We'll choose 3 as a common length for your examples (e.g., -0.5,-0.25, 0)
MAX_SEQUENCE_LENGTH = 4

# Function to prepare a single sequence for the model
# We'll pad shorter sequences with zeros at the beginning
# and truncate longer ones from the beginning.
def prepare_sequence_for_model(sequence, max_len):
    if len(sequence) > max_len:
        return sequence[-max_len:] # Take the last 'max_len' elements
    elif len(sequence) < max_len:
        # Pad with zeros at the beginning
        padded_sequence = [0.0] * (max_len - len(sequence)) + list(sequence)
        return padded_sequence
    else:
        return list(sequence)

# 2. Prepare our "training data" for non-linear patterns
# We need to explicitly define the input sequences and their next number.
# We'll include your original linear examples and some non-linear ones.

# Format: (input_sequence, expected_next_number)
raw_training_data = [
    # --- Linear (Arithmetic Progressions) ---
    ([1, 2, 3], 4), ([2, 4, 6], 8), ([0, 5, 10], 15), ([-3, -1, 1], 3),
    ([10, 20, 30], 40), ([1, 1.5, 2], 2.5), ([5, 3, 1], -1), ([1, 3, 5], 7),
    ([2, 5, 8], 11), ([100, 90, 80], 70), ([0, 0, 0], 0), ([1, 1, 1], 1),
    ([0, 10, 20], 30), ([7, 14, 21], 28), ([100, 95, 90], 85),
    ([0.1, 0.2, 0.3], 0.4), ([-5, 0, 5], 10), ([10, 8, 6], 4),
    ([20, 18, 16, 14], 12), ([1.2, 2.4, 3.6], 4.8),

    # --- Non-linear (Arithmetic progression with increasing/decreasing differences) ---
    # Quadratic patterns (n^2 or similar progression)
    ([1, 2, 4], 7),   # Diff: 1, 2, 3
    ([1, 2, 4, 7], 11), # Diff: 1, 2, 3, 4
    ([0, 1, 3], 6),   # Diff: 1, 2, 3
    ([0, 1, 3, 6], 10), # Diff: 1, 2, 3, 4
    ([1, 5, 12], 22), # Diff: 4, 7, 10 (diffs are linear)
    ([1, 3, 6, 10], 15), # Triangular numbers

    # --- Non-linear (Geometric progressions - multiplication) ---
    ([1, 2, 4], 8),
    ([2, 4, 8], 16),
    ([0.5, 1, 2], 4),
    ([3, 9, 27], 81),
    ([10, 100, 1000], 10000),
    ([2, 1, 0.5], 0.25), # Halving

    # --- Your specific problem examples (more of these) ---
    ([0.5, 0.9], 1.3), ([0.5, 0.9, 1.3], 1.7), # Add more points for these specific patterns
    ([-0.1, 0], 0.1), ([-0.1, 0, 0.1], 0.2),
    ([0, -1], -2), ([0, -1, -2], -3),
    ([-0.5, -0.25, 0], 0.25), ([-0.5, -0.25, 0, 0.25], 0.5),
    ([0.5, 1, 1.5], 2), ([0.5, 1, 1.5, 2], 2.5),

    # --- Mix of short/long sequences requiring padding/truncation ---
    ([100], 100), # Model will see [0,0,0,100]
    ([5, 10], 15), # Model will see [0,0,5,10]
    ([1, 2, 3, 4, 5], 6), # Will be truncated to [2,3,4,5]
    ([10, 20, 30, 40, 50, 60], 70), # Will be truncated to [30,40,50,60]
    ([7], 7),
    ([-1], 0), # Simple +1 pattern start

    # --- Randomly generated examples (or more fixed ones like this) ---
    # To fill out the data, imagine varying ranges and patterns.
    # For simplicity, I'm adding more fixed linear/geometric.
    # In a real project, you might write code to generate thousands of random sequences.
    ([1.0, 1.1, 1.2, 1.3], 1.4),
    ([5.0, 4.5, 4.0, 3.5], 3.0),
    ([0.01, 0.02, 0.04], 0.08), # Small geometric
    ([100, 50, 25], 12.5), # Dividing by 2
    ([0, 0.2, 0.4, 0.6], 0.8),
    ([-2, -4, -6, -8], -10),
    ([1, -1, 1, -1], 1), # Alternating
    ([10, 0, -10], -20),
    ([3, 6, 9, 12], 15),
    ([10, 20, 40], 80),
    ([1, 10, 100], 1000),
    ([50, 40, 30, 20], 10),
    ([0, 0.5, 1.0, 1.5], 2.0),
    ([1000, 100, 10, 1], 0.1),
    ([1, 4, 9], 16), # Perfect squares
    ([1, 4, 9, 16], 25),
    ([1, 8, 27], 64), # Perfect cubes

    # Add many more similar examples to reach ~100-200 total for good effect
    # I'll just put a placeholder for many more, but you can manually add more specific ones
    # Example to show the effect of adding more data:
    ([0.2, 0.4, 0.6], 0.8), ([0.3, 0.6, 0.9], 1.2), ([0.1, 0.15, 0.2], 0.25),
    ([2, 2.5, 3], 3.5), ([5, 5.5, 6], 6.5), ([10, 10.1, 10.2], 10.3),
    ([1, 2, 3, 5], 8), # Fibonacci-like (hard for simple MLP)
    ([1, 1, 2, 3], 5), # Fibonacci
    ([3, 5, 8], 13), # Fibonacci

    # More diverse geometric/quadratic
    ([2, 6, 18], 54),
    ([5, 10, 20], 40),
    ([1, 10, 100], 1000),
    ([1000, 100, 10], 1),
    ([1, 3, 9], 27),
    ([1, 4, 16], 64),
    ([2, 5, 10], 17), # Diff: 3, 5, 7
    ([2, 5, 10, 17], 26) # Diff: 3, 5, 7, 9
]

# Convert raw data into features (X) and targets (y)
X_raw = []
y_raw = []

for seq, next_num in raw_training_data:
    X_raw.append(prepare_sequence_for_model(seq, MAX_SEQUENCE_LENGTH))
    y_raw.append(next_num)

X = np.array(X_raw)
y = np.array(y_raw)

print(f"Original X shape: {X.shape}, Example X[0]: {X[0]}")
print(f"Original y shape: {y.shape}, Example y[0]: {y[0]}")

# 3. Split data into training and testing sets
# Why: We train the model on one part of the data and test it on a part it hasn't seen.
# This helps us know if it truly learned the pattern or just memorised the training data.
# `test_size=0.2` means 20% of data for testing, 80% for training.
# `random_state` ensures you get the same split every time you run it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {len(X_train)} examples")
print(f"Testing data size: {len(X_test)} examples")

# 4. Scale our data (very important for Neural Networks!)
# Why: Neural networks work much better when input numbers are in a similar range (e.g., between -1 and 1).
# `StandardScaler` adjusts numbers so they have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Learn scaling from training data, then apply
X_test_scaled = scaler.transform(X_test)       # Apply the *same* scaling to test data

print(f"\nScaled X_train_scaled[0]: {X_train_scaled[0]}")

# 5. Create our Neural Network model (MLP Regressor)
# hidden_layer_sizes=(10, 5) means two hidden layers, one with 10 neurons, one with 5.
#   You can experiment with these numbers. More neurons/layers can learn more complex patterns
#   but also take longer to train and can "overfit" (memorise training data too well).
# activation='relu' is a common non-linear activation function.
# max_iter=2000 means it will try to learn for up to 2000 cycles.
# random_state for reproducibility
model = MLPRegressor(hidden_layer_sizes=(20, 10), # More neurons in hidden layers
                     activation='relu',
                     max_iter=5000,              # Increased iterations
                     random_state=42,
                     verbose=False,              # Set to True if you want to see training progress
                     solver='adam',
                     learning_rate_init=0.005,   # Slightly smaller learning rate
                     tol=1e-5)                   # Tolerance for stopping training

print("\n--- Training the Neural Network Model ---")
# 6. Train the model
model.fit(X_train_scaled, y_train)

print("\n--- Model Training Complete ---")

# 7. Evaluate the model on the test set
# Why: This tells us how well our model performs on data it has NEVER seen during training.
# A score of 1.0 is perfect. Closer to 1.0 is better.
score = model.score(X_test_scaled, y_test)
print(f"\nModel accuracy (R-squared) on test data: {score:.3f}")

# 8. Make predictions
print("\n--- Making Predictions ---")

# Function to predict a next number for a given sequence
def predict_next_number(model_to_use, scaler_to_use, sequence_to_predict, max_len):
    prepared_seq = prepare_sequence_for_model(sequence_to_predict, max_len)
    # The model expects a 2D array, even for a single prediction
    prepared_seq_array = np.array([prepared_seq])
    # IMPORTANT: Scale the input for prediction using the SAME SCALER that was used for training
    scaled_input = scaler_to_use.transform(prepared_seq_array)
    prediction = model_to_use.predict(scaled_input)[0]
    return prediction

# Test sequences (include your original and some new non-linear ones)
test_sequences_to_predict = [
    # Your specific examples
    ([0.5, 0.9], "Expected: 1.3"),
    ([-0.1, 0], "Expected: 0.1"),
    ([0, -1], "Expected: -2"),
    ([-0.5, -0.25, 0], "Expected: 0.25"),
    ([0.5, 1, 1.5], "Expected: 2"),

    # Test non-linear patterns
    ([1, 2, 4], "Expected: 7 (arithmetic with increasing diff) or 8 (geometric)"),
    ([1, 2, 4, 7], "Expected: 11"),
    ([2, 4, 8], "Expected: 16"), # Geometric
    ([0.5, 1, 2], "Expected: 4"), # Geometric

    # More linear tests
    ([10, 20, 30], "Expected: 40"),
    ([5, 3], "Expected: 1"), # Requires padding
    ([1, 1, 1], "Expected: 1"),
    ([100], "Expected: ? (single number, model might struggle more)"), # Padding makes this [0,0,100]
    ([10, 15], "Expected: 20"), # Padding makes this [0,10,15]
]

for seq, expectation in test_sequences_to_predict:
    predicted_val = predict_next_number(model, scaler, seq, MAX_SEQUENCE_LENGTH)
    print(f"Input: {seq} -> Predicted: {predicted_val:.2f} ({expectation})")

print("\n--- Predictions Complete! ---")