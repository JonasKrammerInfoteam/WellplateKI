import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader



# Set the size of the image (channels, height, width)
channels = 3  # RGB channels
height = 224    # height of the image
width = 224     # width of the image

# Generate a random tensor with values between 0 and 1, simulating an image
image_tensor_example = torch.rand((channels, height, width))

#Target Tensor Example



# Set the size of the image (channels, height, width)
 # RGB channels
height = 96    # height of the image
     # width of the image

# Generate a random tensor with values between 0 and 1, simulating an image
target_tensor_example = torch.rand((height))



class WellDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

     
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

training_data_set = WellDataSet(image_tensor_example, target_tensor_example)

dataloader = DataLoader(training_data_set, batch_size=32, shuffle=True)



# Model Definition
model = models.mobilenet_v3_large(pretrained=True)

num_classes = 96

model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


epochs = 20

for epoch in range(epochs):
    for input, targets in dataloader:
        optimizer.zero_grad()

        outputs = model(input)

        loss = loss_function(outputs.squeeze(), targets.float())

        loss.backward()

        optimizer.step()

print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')