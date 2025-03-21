import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader



# Set the size of the image (channels, height, width)
channels = 3  # RGB channels
height = 256    # height of the image
width = 256     # width of the image
batch_size = 10

# Generate a random tensor with values between 0 and 1, simulating an image
image_tensor_example = torch.rand((batch_size, channels, height, width))

#Target Tensor Example

# Print the shape of the tensor and the tensor itself
print("Tensor shape:", image_tensor_example.shape)
print("Image tensor:\n", image_tensor_example)

# Set the size of the image (channels, height, width)
 # RGB channels
batch_size = 10 
height = 1    # height of the image
width = 96     # width of the image

# Generate a random tensor with values between 0 and 1, simulating an image
target_tensor_example = torch.rand((batch_size, height, width))

# Print the shape of the tensor and the tensor itself
print("Tensor shape:", target_tensor_example.shape)
print("Image tensor:\n", target_tensor_example)


class WellDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

training_data_set = WellDataSet(image_tensor_example, target_tensor_example)

dataloader = DataLoader(training_data_set, batch_size=32, shuffle=True)



# Model Definition
model = models.mobilenet_v3_large(pretrained=True)

num_classes = 96

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


epochs = 20



