import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import random # Needed for data generation

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

print("--- Starting our Improved Scikit-learn Predictor (with more features) ---")

# Define the maximum length of our input sequences
MAX_SEQUENCE_LENGTH = 4 # Keep this consistent for padding

# --- NEW: Feature Engineering Function ---
def create_features_from_sequence(sequence, max_len):
    # 1. Pad/Truncate the sequence to max_len
    processed_seq = [0.0] * max(0, max_len - len(sequence)) + list(sequence[-max_len:])

    # 2. Extract basic features (last number, sum, average, min, max)
    last_num = processed_seq[-1]
    seq_sum = sum(processed_seq)
    seq_mean = np.mean(processed_seq)
    seq_min = min(processed_seq)
    seq_max = max(processed_seq)

    # 3. Calculate differences/steps (crucial for linear patterns)
    diffs = [processed_seq[i] - processed_seq[i-1] for i in range(1, len(processed_seq))]
    avg_diff = np.mean(diffs) if diffs else 0.0
    # If there are enough numbers, consider a "second order" difference (for quadratic patterns)
    second_diffs = [diffs[i] - diffs[i-1] for i in range(1, len(diffs))]
    avg_second_diff = np.mean(second_diffs) if second_diffs else 0.0

    # 4. Indicators for geometric patterns (e.g., ratios)
    # Avoid division by zero
    ratios = [processed_seq[i] / processed_seq[i-1] for i in range(1, len(processed_seq)) if processed_seq[i-1] != 0]
    avg_ratio = np.mean(ratios) if ratios else 1.0 # Default to 1 if no ratios

    # Combine all features into a single list
    # The order matters and must be consistent!
    features = processed_seq + [last_num, seq_sum, seq_mean, seq_min, seq_max,
                                avg_diff, avg_second_diff, avg_ratio]
    return features

# --- Data Generation (Expanded and Diverse) ---
def generate_training_data(num_examples=10000, sequence_length=MAX_SEQUENCE_LENGTH):
    sequences = []
    targets = []

    for _ in range(num_examples):
        pattern_type = random.choice(['linear', 'geometric', 'quadratic_diff', 'alternating_sign', 'constant'])

        if pattern_type == 'linear':
            start = random.uniform(-20, 20)
            diff = random.uniform(-10, 10)
            seq = [start + i * diff for i in range(sequence_length)]
            next_num = start + sequence_length * diff
        elif pattern_type == 'geometric':
            start = random.uniform(0.1, 10) if random.random() < 0.5 else random.uniform(-10, -0.1)
            # Ensure ratio doesn't make numbers too huge/small quickly
            ratio = random.uniform(0.5, 2.0)
            if abs(ratio - 1.0) < 0.05: ratio = random.choice([0.5, 1.5]) # Avoid ratio too close to 1
            seq = [start * (ratio ** i) for i in range(sequence_length)]
            next_num = start * (ratio ** sequence_length)
        elif pattern_type == 'quadratic_diff': # e.g., 1,2,4,7,11 (diffs 1,2,3,4)
            start = random.uniform(-10, 10)
            first_diff = random.uniform(-5, 5)
            second_diff_inc = random.uniform(0.1, 2.0) # How much the difference increases
            seq = [start]
            current_diff = first_diff
            for _ in range(sequence_length - 1):
                next_val = seq[-1] + current_diff
                seq.append(next_val)
                current_diff += second_diff_inc
            next_num = seq[-1] + current_diff
        elif pattern_type == 'alternating_sign':
            start = random.uniform(1, 10)
            seq = [start * (-1)**i for i in range(sequence_length)]
            next_num = start * (-1)**sequence_length
        elif pattern_type == 'constant':
            val = random.uniform(-10, 10)
            seq = [val] * sequence_length
            next_num = val

        # Randomly shorten sequences to test padding
        actual_len = random.randint(1, sequence_length)
        sequences.append(seq[:actual_len]) # Store actual sequence for feature creation
        targets.append(next_num)

    return sequences, targets

# Generate the raw data (sequences can be shorter than MAX_SEQUENCE_LENGTH)
X_raw_sequences, y_raw = generate_training_data(num_examples=20000, sequence_length=MAX_SEQUENCE_LENGTH) # More examples!

# Convert raw sequences into rich features
X = []
for seq in X_raw_sequences:
    X.append(create_features_from_sequence(seq, MAX_SEQUENCE_LENGTH))

X = np.array(X)
y = np.array(y_raw)

print(f"Total number of training examples: {len(X)}")
print(f"Input feature shape (X): {X.shape}, Example X[0]: {X[0]}")
print(f"Target value shape (y): {y.shape}, Example y[0]: {y[0]}")

# Split data (same as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42) # Slightly larger test set

print(f"\nTraining data size: {len(X_train)} examples")
print(f"Testing data size: {len(X_test)} examples")

# Scale our data (StandardScaler) - CRITICAL
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the output (target) as well - can help MLP converge better
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()


print(f"\nScaled X_train_scaled[0]: {X_train_scaled[0]}")
print(f"Scaled y_train_scaled[0]: {y_train_scaled[0]}")


# --- MLPRegressor Model and Training (Tuned Parameters) ---
model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), # More layers/neurons
                     activation='relu',
                     solver='adam',
                     learning_rate_init=0.001, # Finer learning rate
                     max_iter=5000,            # More iterations
                     random_state=42,
                     tol=1e-6,                 # Stricter tolerance
                     verbose=True,             # See training progress
                     n_iter_no_change=200)     # Stop if no improvement for 200 iterations

print("\n--- Training the Neural Network Model ---")
model.fit(X_train_scaled, y_train_scaled) # Train with scaled targets

print("\n--- Model Training Complete ---")

# --- Evaluation ---
from sklearn.metrics import r2_score, mean_absolute_error

y_pred_scaled = model.predict(X_test_scaled)
# Inverse transform predictions and true values to original scale for evaluation
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred) # Mean Absolute Error - easier to interpret than MSE
print(f"\nR-squared on Test Data: {r2:.4f}")
print(f"Mean Absolute Error on Test Data: {mae:.4f}") # Average difference between prediction and true value

# --- Make Predictions for Specific Sequences ---
print("\n--- Making Predictions for Specific Sequences ---")

def predict_next_number_mlp(model, scaler_X, scaler_y, sequence, max_len):
    # Create features from the input sequence
    features = create_features_from_sequence(sequence, max_len)

    # Scale the features using the SAME scaler used during training
    scaled_features = scaler_X.transform(np.array(features).reshape(1, -1))

    # Make prediction with the scaled features
    predicted_scaled = model.predict(scaled_features)

    # Inverse transform the prediction back to original scale
    predicted_original = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()[0]
    return predicted_original

# Test sequences (your original list, and some new ones)
test_sequences_to_predict = [
    # Your specific examples
    ([0.5, 0.9], "Expected: 1.3"),
    ([-0.1, 0], "Expected: 0.1"),
    ([0, -1], "Expected: -2"),
    ([-0.5, -0.25, 0], "Expected: 0.25"),
    ([0.5, 1, 1.5], "Expected: 2"),
    # Longer examples that match or exceed SEQUENCE_LENGTH
    ([0.5, 0.9, 1.3, 1.7], "Expected: 2.1"),
    ([-0.5, -0.25, 0, 0.25], "Expected: 0.5"),
    ([1, 2, 4, 7], "Expected: 11"),   # Quadratic diff (1,2,3,4)
    ([1, 2, 4, 8], "Expected: 16"),   # Geometric (doubling)
    ([2, 4, 8, 16], "Expected: 32"),  # Geometric
    ([1, 1, 2, 3], "Expected: 5"),    # Fibonacci (will be harder due to feature engineering)
    ([3, 5, 8, 13], "Expected: 21"),  # Fibonacci

    # More linear tests
    ([10, 20, 30], "Expected: 40"),
    ([5, 3], "Expected: 1"),          # Short sequence, will be padded
    ([1, 1, 1], "Expected: 1"),
    ([100], "Expected: 100 (single number, padded)"),
    ([10, 15], "Expected: 20"),       # Short sequence, will be padded
    ([1, 2, 3, 4], "Expected: 5"),
    ([100, 99, 98], "Expected: 97"),
    ([10, 20, 40], "Expected: 80"),
    ([2, 5, 10, 17], "Expected: 26"), # Arithmetic with increasing diff (3,5,7,9)
    ([1, 4, 9, 16], "Expected: 25"), # Perfect squares
    ([10, 0, -10], "Expected: -20")
]

for seq, expectation in test_sequences_to_predict:
    predicted_val = predict_next_number_mlp(model, scaler, scaler_y, seq, MAX_SEQUENCE_LENGTH)
    print(f"Input: {seq} -> Predicted: {predicted_val:.2f} ({expectation})")

print("\n--- Predictions Complete! ---")