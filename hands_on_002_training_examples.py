import torch

text = """First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge."""

block_size = 8
batch_size = 4

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)

ix = torch.randint(len(data) - block_size, (batch_size,))
print("ix:", ix)
x = torch.stack([data[i:i+block_size] for i in ix])
# FIXME: implement y
y = ...
print('x:\n', x)
print('y:\n', y)
print('decoded x:\n', [decode(x[i].tolist()) for i in range(batch_size)])
print('decoded y:\n', [decode(y[i].tolist()) for i in range(batch_size)])