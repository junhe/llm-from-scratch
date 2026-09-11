import torch
import torch.nn.functional as F

# 4 classes; the correct class is index 2.
# class logit
# 0     2
# 1     1
# 2     5   <- correct class with high logit
# 3     3
logits_good = torch.tensor([[2., 1., 5., 3.]])
# class logit
# 0     2
# 1     1
# 2     0.5 <- correct class with low logit
# 3     3
logits_bad = torch.tensor([[2., 1., 0.5, 3.]])
# FIXME
# Specify target class index to be 2
target = torch.tensor(...)

loss_good = F.cross_entropy(logits_good, target)
loss_bad = F.cross_entropy(logits_bad, target)

print("Good prediction loss:", loss_good.item())
print("Bad prediction loss:", loss_bad.item())
print("Probs (good):", F.softmax(logits_good, dim=-1))
print("Probs (bad):", F.softmax(logits_bad, dim=-1))
