# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:41:19 2019

@author: Yao
"""

from scipy.stats import bernoulli

p = 1.0/2 #需要估计的目标
sample = bernoulli(p)
xs = sample.rvs(100) #生成100个样本
print(xs[:10]) #查看前10个生成样本

import sympy
import numpy as np
x, p, z = sympy.symbols('x p z', positive = True)
phi = p ** x * (1-p) ** (1-x)
L = np.prod([phi.subs(x,i)for i in xs])
print(L)

logL = sympy.expand_log(sympy.log(L))
print(logL)
sol, = sympy.solve(sympy.diff(logL, p),p) #求解
print(sol)

