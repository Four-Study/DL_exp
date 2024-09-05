import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def save_cifar10_images(num_images=5):
    # Define the transformation to convert the image data to a tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    # Load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=transform)
    
    # Create a DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_images,
                                              shuffle=True, num_workers=2)

    # Get one batch of images using an iterator
    dataiter = iter(trainloader)
    images, labels = next(dataiter)  # Use next() function properly here

    # Unnormalize and convert the tensor to a PIL image, then save
    unnormalize = transforms.Normalize((-1, -1, -1), (2, 2, 2))  # Undo the normalization
    for i in range(num_images):
        image = images[i]  # Get the image tensor
        image = unnormalize(image)  # Unnormalize the image
        img = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
        img.save(f'imgs/cifar10_image_{i}.png')  # Save the image

# Call the function
save_cifar10_images(num_images=10)
