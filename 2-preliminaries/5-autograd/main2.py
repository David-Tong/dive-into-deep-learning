import torch

# 2
x = torch.arange(4.0)
x.requires_grad_(True) 
y = 2 * torch.dot(x, x)
y.backward()

try:
    y.backward()
except RuntimeError:
    pass

print(x.grad)

# 3 
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(3, 2), requires_grad=True)
# a = torch.randn(size=12, requires_grad=True)
d = f(a)
# make it work
# d.backward()
d.sum().backward()

print(a)
print(d)
print(a.grad)

# 5
import matplotlib.pylab as plt
import numpy as np
import torch

f , ax = plt.subplots(1)

x = np.linspace(-3 * np.pi, 3 * np.pi, 100)
x1= torch.tensor(x, requires_grad=True)
y1= torch.sin(x1)
y1.sum().backward()

ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, x1.grad, label="gradient of sin(x)")
ax.legend(loc="upper center", shadow=True)
plt.show()