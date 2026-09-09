import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# 1. Define the Single Neuron Architecture
# ---------------------------------------------------------
class SingleNeuronClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 inputs (Temperature, Wind Speed) -> 1 output
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # Returns raw scores (logits)
        return self.linear(x)

# ---------------------------------------------------------
# 2. Create Training Data
# ---------------------------------------------------------
# FIXME: Add more training data
# Features: [Temperature (°F), Wind Speed (mph)]
X_raw = torch.tensor([
    [75.0,  3.0],  # Warm, calm -> YES
], dtype=torch.float32)

# Target Labels: 1.0 (YES - Play), 0.0 (NO - Don't Play)
y_train = torch.tensor([
    [1.0],
], dtype=torch.float32)

# Normalize inputs for stable gradient descent
mean = X_raw.mean(dim=0)
std = X_raw.std(dim=0)
X_train = (X_raw - mean) / std

# ---------------------------------------------------------
# 3. Model Setup & Training Loop
# ---------------------------------------------------------
model = SingleNeuronClassifier()

# BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy for numerical stability.
# We will explain loss function later in the article.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    
    logits = model(X_train)
    loss = criterion(logits, y_train)
    
    loss.backward()
    optimizer.step()

print(f"Training completed. Final Loss: {loss.item():.4f}\n")

# Inspect learned parameters
learned_weights = model.linear.weight.data[0]
learned_bias = model.linear.bias.data[0]
print(f"Learned Weight for Temp: {learned_weights[0]:.3f}")
print(f"Learned Weight for Wind: {learned_weights[1]:.3f}")
print(f"Learned Bias:           {learned_bias:.3f}\n")

# ---------------------------------------------------------
# 4. Predict on Unseen Weather Conditions
# ---------------------------------------------------------
# New test days: [Temperature (°F), Wind Speed (mph)]
test_days_raw = torch.tensor([
    [78.0, 4.0],   # Great conditions
    [48.0, 18.0],  # Bad conditions
    [62.0, 12.0],  # Marginal conditions
], dtype=torch.float32)

# Apply same normalization
test_days = (test_days_raw - mean) / std

model.eval()
with torch.no_grad():
    test_logits = model(test_days)
    # Sigmoid converts logits to probabilities in [0, 1]
    probabilities = torch.sigmoid(test_logits)
    # Binary Step Threshold: >= 0.5 -> YES (1), < 0.5 -> NO (0)
    decisions = (probabilities >= 0.5).int()

# ---------------------------------------------------------
# 5. Display Predictions
# ---------------------------------------------------------
print("--- Prediction Results ---")
for i in range(len(test_days_raw)):
    temp = test_days_raw[i][0].item()
    wind = test_days_raw[i][1].item()
    prob = probabilities[i].item() * 100
    decision_str = "YES (Play)" if decisions[i].item() == 1 else "NO (Stay Inside)"
    
    print(f"Day {i+1}: {temp:.0f}°F, {wind:.0f} mph wind -> Confidence: {prob:5.1f}% | Decision: {decision_str}")
