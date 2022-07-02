import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
inp = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
out = loss(inp, target)
out.backward()
print(out)