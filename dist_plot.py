## Draw distribution plot in python

import torch
import matplotlib.pyplot as plt
import numpy as np

# Generate a random tensor from a normal distribution (mean=0, std=0.5) on CUDA
tensor = torch.randn(3000, device='cuda') * 0.5

# Apply the function f(x) = sqrt(x^2 + 1)
transformed_tensor = torch.sqrt(tensor**2 + 1)

# Move the tensor to CPU and convert it to a NumPy array
transformed_cpu = transformed_tensor.cpu().numpy()

# Plot the histogram using matplotlib
plt.hist(transformed_cpu, bins=30, alpha=0.7, color='blue')
plt.title(r'Histogram of $f(x) = \sqrt{x^2 + 1}$ Applied to Tensor (mean=0, std=0.5)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
