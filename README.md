
<div align="center">
      <H1> important of activation function</H1>
<H2>A In most deep learning frameworks like TensorFlow or PyTorch, you can import activation functions directly from the libraries they provide. Below are examples of how to import and use common activation functions in both frameworks.
</H2>  
     </div>

<body>
<p align="center">
  <a href="mailto:arifmiahcse952@gmail.com"><img src="https://img.shields.io/badge/Email-arifmiah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/Arif-miad"><img src="https://img.shields.io/badge/GitHub-%40ArifMiah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/arif-miah-8751bb217/"><img src="https://img.shields.io/badge/LinkedIn-Arif%20Miah-blue?style=flat-square&logo=linkedin"></a>

 
  
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801998246254-green?style=flat-square&logo=whatsapp">
  
</p>
1. TensorFlow / Keras:
Keras, which is part of TensorFlow, has a variety of activation functions built in. You can import them from tensorflow.keras.layers or tensorflow.keras.activations.

Example Imports for Activation Functions in TensorFlow/Keras:
```python
# Importing individual activation functions
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, softmax, sigmoid

# Example usage in a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple feed-forward neural network model
model = Sequential([
    Dense(64, input_shape=(784,)),
    Activation('relu'),  # Using activation by name
    Dense(10),
    Activation('softmax')  # Using activation by name
])

# Alternatively, you can use activation functions directly in Dense layers
model = Sequential([
    Dense(64, activation=relu, input_shape=(784,)),  # Using relu
    Dense(10, activation=softmax)  # Using softmax
])

```
2. PyTorch:
In PyTorch, activation functions are available in torch.nn (as layers) or torch.nn.functional (as functions).

Example Imports for Activation Functions in PyTorch:

```python
# Importing activation functions from torch.nn.functional
import torch.nn.functional as F

# Example usage in a custom neural network model
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Using relu activation
        x = F.softmax(self.fc2(x), dim=1)  # Using softmax activation
        return x

# Creating an instance of the model
model = SimpleModel()

```
Commonly Used Activation Functions in PyTorch:
ReLU: F.relu(x) or nn.ReLU()
Sigmoid: F.sigmoid(x) or nn.Sigmoid()
Softmax: F.softmax(x, dim=1) or nn.Softmax(dim=1)
Tanh: F.tanh(x) or nn.Tanh()
These activations can be used either as part of layers or as standalone functions during the forward pass, depending on how you prefer to structure the model.
```python
# Linear Activation Functions
import numpy as np 
import matplotlib.pyplot as plt

# Define the linear activation fucntion
def linear_activation(x):
    return x 
# Generate a range of values
x = np.linspace(-10,10,400) 
y = linear_activation(x)
plt.figure(figsize=(4,3))
plt.plot(x,y,label="Linear Activation", color="blue")
plt.title("Linear Activation Function")
plt.xlabel("INput")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

```
```python
# Sinmoid Activation Function
def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.linspace(-10,10,600)
y = sigmoid(x)
plt.plot(x,y)
plt.title("sigmoid Activation function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
```

```python
def tanh_function(x):
    return np.tanh(x)
x = np.linspace(-20,20,500)
y = tanh_function(x)
plt.figure(figsize=(8,6))
plt.plot(x,y,label='tanh(x)', color='blue')
plt.title("Hyperblic Tangent Function (tanh)")
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
```
```python
def relu(x):
    return np.maximum(0,x)
x = np.linspace(-30,30,800)
y = relu(x)
plt.figure(figsize=(10,5))
plt.plot(x,y, label='ELU function', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel("ReLU(x)")
plt.grid()
    
plt.legend()
plt.show()
```

