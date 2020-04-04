---
layout: post
published: false
title: Pytorch Basics and Speed Comparisons with Numpy and GPU
date: '2020-01-04'
---

## Pytorch Basics and Speed Comparisons with Numpy and GPU

* Initialising, slicing, reshaping tensors
* Numpy and PyTorch interfacing
* Speed comparisons, Numpy -- PyTorch -- PyTorch on GPU
* Autodiff concepts and application
* Writing a basic learning loop using autograd


```python
import torch
import numpy as np
import matplotlib.pyplot as plt
```

## Initialise tensors


```python
x = torch.ones(3, 2)
print(x)
x = torch.zeros(3, 2)
print(x)
x = torch.rand(3, 2)
print(x)
```

    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[0.2365, 0.3903],
            [0.9967, 0.0030],
            [0.5356, 0.1902]])



```python
x = torch.empty(3, 2)
print(x)
y = torch.zeros_like(x)
print(y)
```

    tensor([[5.1069e-36, 0.0000e+00],
            [0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00]])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])



```python
x = torch.linspace(0, 1, steps=5)
print(x)
```

    tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])



```python
x = torch.tensor([[1, 2], 
                 [3, 4], 
                 [5, 6]])
print(x)
```

    tensor([[1, 2],
            [3, 4],
            [5, 6]])


## Slicing tensors


```python
print(x.size())
print(x[:, 1]) 
print(x[0, :]) 
```

    torch.Size([3, 2])
    tensor([2, 4, 6])
    tensor([1, 2])



```python
y = x[1, 1]
print(y)
print(y.item())
```

    tensor(4)
    4


## Reshaping tensors


```python
print(x)
y = x.view(2, 3)
print(y)
```

    tensor([[1, 2],
            [3, 4],
            [5, 6]])
    tensor([[1, 2, 3],
            [4, 5, 6]])



```python
y = x.view(6,-1) 
print(y)
```

    tensor([[1],
            [2],
            [3],
            [4],
            [5],
            [6]])


## Simple Tensor Operations


```python
x = torch.ones([3, 2])
y = torch.ones([3, 2])
z = x + y
print(z)
z = x - y
print(z)
z = x * y
print(z)
```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])



```python
z = y.add(x)
print(z)
print(y)
```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]])



```python
z = y.add_(x)
print(z)
print(y)
```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])
    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]])


## Numpy vs PyTorch


```python
x_np = x.numpy()
print(type(x), type(x_np))
print(x_np)
```

    <class 'torch.Tensor'> <class 'numpy.ndarray'>
    [[1. 1.]
     [1. 1.]
     [1. 1.]]


### Convert Numpy array to Torch tensor


```python
a = np.random.randn(5)
print(a)
a_pt = torch.from_numpy(a)
print(type(a), type(a_pt))
print(a_pt)
```

    [-0.22648217  2.33063636 -2.35281499  1.52592661  0.05184195]
    <class 'numpy.ndarray'> <class 'torch.Tensor'>
    tensor([-0.2265,  2.3306, -2.3528,  1.5259,  0.0518], dtype=torch.float64)



```python
np.add(a, 1, out=a)
print(a)
print(a_pt) 
```

    [ 0.77351783  3.33063636 -1.35281499  2.52592661  1.05184195]
    tensor([ 0.7735,  3.3306, -1.3528,  2.5259,  1.0518], dtype=torch.float64)


### Speed Comparison b/w Numpy and Torch on CPU


```python
%%time
for i in range(100):
  a = np.random.randn(100,100)
  b = np.random.randn(100,100)
  c = np.matmul(a, b)
```

    CPU times: user 161 ms, sys: 117 ms, total: 278 ms
    Wall time: 158 ms



```python
%%time
for i in range(100):
  a = torch.randn([100, 100])
  b = torch.randn([100, 100])
  c = torch.matmul(a, b)
```

    CPU times: user 32.2 ms, sys: 1.93 ms, total: 34.1 ms
    Wall time: 36.3 ms



```python
%%time
for i in range(10):
  a = np.random.randn(10000,10000)
  b = np.random.randn(10000,10000)
  c = a + b
```

    CPU times: user 1min 36s, sys: 1.35 s, total: 1min 37s
    Wall time: 1min 37s



```python
%%time
for i in range(10):
  a = torch.randn([10000, 10000])
  b = torch.randn([10000, 10000])
  c = a + b
```

    CPU times: user 16 s, sys: 9.8 ms, total: 16 s
    Wall time: 16 s


### We can see that Pytorch on CPU is around 4-5 times faster than Numpy

## CUDA support


```python
print(torch.cuda.device_count())
```

    1



```python
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
```

    <torch.cuda.device object at 0x7f7ed42ee390>
    Tesla K80


### We have Tesla K80 GPU running on this notebook. To know more about this GPU and its performance refer to this <a href='https://www.nvidia.com/en-gb/data-center/tesla-k80/'>link</a>.


```python
cuda0 = torch.device('cuda:0')
```


```python
a = torch.ones(3, 2, device=cuda0)
b = torch.ones(3, 2, device=cuda0)
c = a + b
print(c)
```

    tensor([[2., 2.],
            [2., 2.],
            [2., 2.]], device='cuda:0')



```python
print(a)
```

    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]], device='cuda:0')



```python
%%time
for i in range(10):
  a = torch.randn([10000, 10000], device=cuda0)
  b = torch.randn([10000, 10000], device=cuda0)
  c = a + b
```

    CPU times: user 3.19 ms, sys: 1.02 ms, total: 4.22 ms
    Wall time: 6.16 ms


### We see drastic improvement in the performance on GPU. It is almost 800 times faster that pytorch on CPU and around 5000 times faster than Numpy.

### Autodiff in Pytorch
Automatic differentiation (autodiff) refers to a general way of taking
a program which computes a value, and automatically constructing a
procedure for computing derivatives of that value.


```python
x = torch.ones([3, 2], requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.],
            [1., 1.]], requires_grad=True)


`requires_grad=True` is mandatory to tell Pytorch to do the book keeping of all the operations being performed with the data.


```python
y = x + 5
print(y)
```

    tensor([[6., 6.],
            [6., 6.],
            [6., 6.]], grad_fn=<AddBackward0>)


Here, `grad_fn=<AddBackward0>` is the way of book keeping in Pytorch


```python
z = y*y + 1
print(z)
```

    tensor([[37., 37.],
            [37., 37.],
            [37., 37.]], grad_fn=<AddBackward0>)



```python
t = torch.sum(z)
print(t)
```

    tensor(222., grad_fn=<SumBackward0>)



```python
t.backward()
```


```python
print(x.grad)
```

    tensor([[12., 12.],
            [12., 12.],
            [12., 12.]])


$t = \sum_i z_i, z_i = y_i^2 + 1, y_i = x_i + 5$

$\frac{\partial t}{\partial x_i} = \frac{\partial z_i}{\partial x_i} = \frac{\partial z_i}{\partial y_i} \frac{\partial y_i}{\partial x_i} = 2y_i \times 1$


At x = 1, y = 6, $\frac{\partial t}{\partial x_i} = 12$


```python
x = torch.ones([3, 2], requires_grad=True)
y = x + 5
r = 1/(1 + torch.exp(-y))
print(r)
s = torch.sum(r)
s.backward()
print(x.grad)
```

    tensor([[0.9975, 0.9975],
            [0.9975, 0.9975],
            [0.9975, 0.9975]], grad_fn=<MulBackward0>)
    tensor([[0.0025, 0.0025],
            [0.0025, 0.0025],
            [0.0025, 0.0025]])



```python
x = torch.ones([3, 2], requires_grad=True)
y = x + 5
r = 1/(1 + torch.exp(-y))
a = torch.ones([3, 2])
r.backward(a)
print(x.grad)
```

    tensor([[0.0025, 0.0025],
            [0.0025, 0.0025],
            [0.0025, 0.0025]])


$\frac{\partial{s}}{\partial{x}} = \frac{\partial{s}}{\partial{r}} \cdot \frac{\partial{r}}{\partial{x}}$

For the above code $a$ represents $\frac{\partial{s}}{\partial{r}}$ and then $x.grad$ gives directly $\frac{\partial{s}}{\partial{x}}$



## Autodiff example that looks like what we have been doing


```python
x = torch.randn([20, 1], requires_grad=True)
y = 3*x - 2
```


```python
w = torch.tensor([1.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)

y_hat = w*x + b

loss = torch.sum((y_hat - y)**2)
```


```python
print(loss)
```

    tensor(237.3524, grad_fn=<SumBackward0>)



```python
loss.backward()
```


```python
print(w.grad, b.grad)
```

    tensor([-63.6629]) tensor([115.7930])


## Do it in a loop


```python
learning_rate = 0.01

w = torch.tensor([1.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)

print(w.item(), b.item())

for i in range(10):
  
  x = torch.randn([20, 1])
  y = 3*x - 2
  
  y_hat = w*x + b
  loss = torch.sum((y_hat - y)**2)
  
  loss.backward()
  
  with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

  print(w.item(), b.item())
  
```

    1.0 1.0
    2.2244343757629395 -0.46036696434020996
    2.5868167877197266 -1.0856199264526367
    2.58056902885437 -1.3719744682312012
    2.7070467472076416 -1.6221551895141602
    2.808570146560669 -1.7485421895980835
    2.8510940074920654 -1.842361330986023
    2.863948106765747 -1.877304196357727
    2.9146628379821777 -1.9155502319335938
    2.9505720138549805 -1.9522515535354614
    2.9600718021392822 -1.9627878665924072


### We can see that w is approaching towards 3 and b towards -2.

## Do it for a large problem


```python
%%time
learning_rate = 0.001
N = 10000000
epochs = 200

w = torch.rand([N], requires_grad=True)
b = torch.ones([1], requires_grad=True)

# print(torch.mean(w).item(), b.item())

for i in range(epochs):
  
  x = torch.randn([N])
  y = torch.dot(3*torch.ones([N]), x) - 2
  
  y_hat = torch.dot(w, x) + b
  loss = torch.sum((y_hat - y)**2)
  
  loss.backward()
  
  with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

#   print(torch.mean(w).item(), b.item())
  
```

    CPU times: user 32.2 s, sys: 86.4 ms, total: 32.2 s
    Wall time: 32.3 s



```python
%%time
learning_rate = 0.001
N = 10000000
epochs = 200

w = torch.rand([N], requires_grad=True, device=cuda0)
b = torch.ones([1], requires_grad=True, device=cuda0)

# print(torch.mean(w).item(), b.item())

for i in range(epochs):
  
  x = torch.randn([N], device=cuda0)
  y = torch.dot(3*torch.ones([N], device=cuda0), x) - 2
  
  y_hat = torch.dot(w, x) + b
  loss = torch.sum((y_hat - y)**2)
  
  loss.backward()
  
  with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

  #print(torch.mean(w).item(), b.item())
  
```

    CPU times: user 837 ms, sys: 492 ms, total: 1.33 s
    Wall time: 1.34 s


At last again some speed comparison b/w Pytorch on CPU vs GPU. On CPU it took 32.3 secs and on GPU, only 1.34 secs.

References: I learnt these basics of Pytorch from https://www.guvi.in/
