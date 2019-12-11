# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:00:32 2019

@author: Yao
"""

import torch
import numpy as np

from torch.autograd import Variable

sample = np.array([ 1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,
        1.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,
        0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,
        0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,
        0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,
        0.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  1.,  1.,
        0.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  1.,  1.,
        1.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,
        1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,
        0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,
        1.,  1.,  1.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,
        1.,  0.,  1.,  0.,  1.])

print(np.mean(sample))


x = Variable(torch.from_numpy(sample)).type(torch.FloatTensor)
p = Variable(torch.rand(1), requires_grad=True)



learning_rate = 0.0002
for t in range(1000):
    NLL = -torch.sum(torch.log(x*p + (1-x)*(1-p)) )
    NLL.backward()

    if t % 100 == 0:
        print("loglik  =", NLL.data.numpy(), "p =", p.data.numpy(), "dL/dp = ", p.grad.data.numpy())
    
    p.data -= learning_rate * p.grad.data
    p.grad.data.zero_()