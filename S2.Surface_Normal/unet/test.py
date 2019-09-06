import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from collections import OrderedDict as odict

input   = torch.rand(1,1,64,64)
pool1   = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)
pool2   = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool1 = nn.MaxUnpool2d(2, stride=2)
unpool2 = nn.MaxUnpool2d(2, stride=2, padding=1)

output1, indices1 = pool1(input)
output2, indices2 = pool2(output1)

print (output1.size())
output3 = unpool1(output2, indices2, output_size = output1.size())
output4 = unpool2(output3, indices1, output_size = input.size()  )
