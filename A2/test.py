import numpy as np
import torch
from PIL import Image, ImageFilter
from deap import base
from torch import nn

from A2.util import Tuple

# Example of target with class indices
loss = nn.BCEWithLogitsLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input, target)
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(input, target)
output = loss(input, target)
print(output)
output.backward()
