import torch

# 3x8x4 concrete token embeddings
tok_emb = torch.tensor([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
    [[41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64], [65, 66, 67, 68], [69, 70, 71, 72]],
    [[81, 82, 83, 84], [85, 86, 87, 88], [89, 90, 91, 92], [93, 94, 95, 96], [97, 98, 99, 100], [101, 102, 103, 104], [105, 106, 107, 108], [109, 110, 111, 112]]
], dtype=torch.float32)

# 8x4 concrete position embeddings
pos_emb = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5, 1.6],
    [1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4],
    [2.5, 2.6, 2.7, 2.8],
    [2.9, 3.0, 3.1, 3.2]
], dtype=torch.float32)

# FIXME:
# Add the two tensors. PyTorch broadcasts pos_emb from (8, 4) to (3, 8, 4).
x = ...

print("tok_emb shape:", tok_emb.shape)
print("pos_emb shape:", pos_emb.shape)
print("x = tok_emb + pos_emb shape:", x.shape)
print(x)
