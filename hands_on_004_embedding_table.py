import torch.nn as nn

# Create an embedding table. Each token index maps to a trainable vector.
# FIXME: define the embedding table that stores 5 embeddings, the dimension of
# each embedding is 8.
embedding = nn.Embedding(...)

# Look up the embedding for a single token index
token_id = torch.tensor([2])
single_embedding = embedding(token_id)
print("Single token embedding shape:", single_embedding.shape)
print("Single token embedding:\n", single_embedding)
