import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

class WellDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

     
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_random_input_data():
    batch_size = 32

    channels = 3  
    height = 224    
    width = 224     
    random_image = torch.randn(channels, height, width)
    image_tensor_example = random_image.unsqueeze(0).repeat(batch_size,1,1,1)

    height = 96   
    random_target = torch.rand(height)
    target_tensor_example = random_target.unsqueeze(0).repeat(batch_size,1)

    return image_tensor_example, target_tensor_example

def create_model():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    num_classes = 96
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def train_model(epochs):
    for epoch in range(epochs):
        for input, targets in dataloader:
            optimizer.zero_grad()

            outputs = model(input)

            loss = loss_function(outputs.squeeze(), targets.float())

            loss.backward()

            optimizer.step()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


image_tensor_example, target_tensor_example = create_random_input_data()

training_data_set = WellDataSet(image_tensor_example, target_tensor_example)
dataloader = DataLoader(training_data_set, batch_size=32, shuffle=True)

model = create_model()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

train_model(epochs=20)

