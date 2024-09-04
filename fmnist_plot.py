import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformation for the dataset
transform = transforms.Compose([transforms.ToTensor()])

# Load FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transform)

# Get the indices of images labeled as 'shoes' (label 7 in FashionMNIST)
shoe_indices = [i for i, label in enumerate(train_dataset.targets) if label == 7]
shoe_images = [train_dataset[i][0].squeeze(0).numpy() for i in shoe_indices[:9]]  # Get the first 10 shoe images

# Plot a few examples of shoe images
plt.figure(figsize=(5, 5))
for i, img in enumerate(shoe_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Sneaker {i+1}")

plt.tight_layout()
plt.show()
