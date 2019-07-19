#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:23:54 2019

@author: rong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# exp(b * t) - 1 = r

r = 3E3
t = 1 * 60 * 60
b = np.log(1 + r) / t

#b = 0.004
#a = 1

x = range(int(3600))
y = map(lambda t: np.exp(b * t) - 1, x)

plt.plot(list(x), list(y))