import math
import time
import numpy as np
import torch

n = 10000
a = torch.ones([n])
b = torch.ones([n])

print(a)
print(b)

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        self.tik = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

    
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')


import matplotlib.pyplot as plt

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
#plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
#         ylabel='p(x)', figsize=(4.5, 2.5),
#         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
y = [normal(x, mu, sigma) for mu, sigma in params]

# plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params])
settings = [{"color" : "blue", "linestyle" : "solid"},
            {"color" : "purple", "linestyle" : "dashed"},
            {"color" : "green", "linestyle" : "dashdot"}
            ]

for z in range(len(y)):
    settings[z]["label"] = "mean {}, std {}".format(params[z][0], params[z][1])
    plt.plot(x, y[z], **settings[z])
plt.legend()
plt.show()
