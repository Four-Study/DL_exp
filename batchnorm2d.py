import torch
import torch.nn as nn

# Set the random seed for reproducibility
torch.manual_seed(0)

# Define a Batch Normalization layer for 2D inputs (e.g., images)
# Assuming input has 3 channels (like an RGB image)
batch_norm2d = nn.BatchNorm2d(num_features=1)

# Create input data with batch size 1, 3 channels, height 4, and width 4
input_data_single = torch.randn(1, 1, 2, 2)

# Create input data with a larger batch size (e.g., 10) for comparison
input_data_large = torch.randn(2, 1, 2, 2)

# Function to apply batch normalization and print statistics
def apply_batch_norm2d(input_data, batch_norm_layer):
    batch_norm_layer.train()  # Set to training mode to use batch statistics
    output = batch_norm_layer(input_data)
    print("Input Data:")
    print(input_data)
    print("\nBatch Normalization Output:")
    print(output)
    print("\nRunning Mean:")
    print(batch_norm_layer.running_mean)
    print("Running Variance:")
    print(batch_norm_layer.running_var)
    print("\n")

# Apply batch normalization to batch size 1
print("Batch Size 1:")
apply_batch_norm2d(input_data_single, batch_norm2d)

# Reset the running statistics
batch_norm2d.reset_running_stats()

# Apply batch normalization to a larger batch size
print("Batch Size 2:")
apply_batch_norm2d(input_data_large, batch_norm2d)

####################################################
#################### Conclusion ####################
####################################################
# The batchnorm2d layer computes the mean and variance over 
# every element of the mini-batch. Then it normalizes every element in 
# the mini-batch by using those statistics. Notice that the running mean 
# and variance are not just from the current batch as there is a momentum
# that keeps the previous ones. 
