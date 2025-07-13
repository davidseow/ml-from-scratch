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
MAX_SEQUENCE_LENGTH = 3

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
    # Original linear-like examples
    ([0.5, 0.9], 1.3),
    ([-0.1, 0], 0.1),
    ([0, -1], -2),
    ([-0.5, -0.25, 0], 0.25),
    ([0.5, 1, 1.5], 2),

    # Non-linear examples (Arithmetic progression with increasing difference)
    ([1, 2, 4], 7),   # Diff: 1, 2, 3
    ([1, 2, 4, 7], 11), # Diff: 1, 2, 3, 4

    # Non-linear examples (Geometric progression - doubling)
    ([1, 2, 4], 8),
    ([2, 4, 8], 16),
    ([0.5, 1, 2], 4),

    # More linear-like to give the model more data
    ([10, 20, 30], 40),
    ([1, 1.5, 2], 2.5),
    ([5, 3, 1], -1), # Note this sequence implies -2 difference.
    ([1, 3, 5], 7),
    ([2, 5, 8], 11),
    ([100, 90, 80], 70),
    ([100, 30, -5], -22.5),
    ([0, 0, 0], 0), # A steady sequence
    ([1, 1, 1], 1), # Another steady sequence
    ([0, 10, 20], 30)
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
model = MLPRegressor(hidden_layer_sizes=(10, 5), activation='relu',
                     max_iter=2000, random_state=42, verbose=False,
                     solver='adam', learning_rate_init=0.01)

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