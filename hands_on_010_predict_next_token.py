import torch

# A tiny single-character vocabulary
chars = ['a', 'b', 'c', 'd']
probs = torch.tensor([0.10, 0.20, 0.30, 0.40])

# Sample the next token index according to the probabilities
idx_next = torch.multinomial(probs, num_samples=1)
print('---')
print(f"Sampled token index: {idx_next.item()}")
print(f"Sampled character  : '{chars[idx_next.item()]}'")

