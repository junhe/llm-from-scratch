import torch
import torch.nn.functional as F

# Add 3 numbers, with at least one -inf.
weight = torch.tensor([2., 1., float('-inf')])
print("Input:", weight)

probs = F.softmax(weight, dim=0)
print("Softmax probabilities:", probs)
print("Sum of probabilities:", probs.sum().item())
