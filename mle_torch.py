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
        1.,  0.,  1.,  0.,  1.])#原始伯努利分布结果 p = 0.725

print(np.mean(sample))


x = Variable(torch.from_numpy(sample)).type(torch.FloatTensor) #数据转化为tensor
p = Variable(torch.rand(1), requires_grad=True)  #目标为估计正确的p



learning_rate = 0.0002  #学习率
for t in range(1000):
    NLL = -torch.sum(torch.log(x*p + (1-x)*(1-p)) )  #使用似然函数作为loss
    NLL.backward()  #根据loss函数反向传播

    if t % 100 == 0:
        print("loglik  =", NLL.data.numpy(), "p =", p.data.numpy(), "dL/dp = ", p.grad.data.numpy())  #每100次打印一次结果
    
    p.data -= learning_rate * p.grad.data  #根据梯度调整p 
    p.grad.data.zero_()  #清除累计的梯度