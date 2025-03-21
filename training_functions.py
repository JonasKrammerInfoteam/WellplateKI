import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        return self.base_lrs

class WellDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

     
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

class MobileNetWithSigmoid(nn.Module):
    def __init__(self, num_classes=1):  # Set num_classes as 1 for binary classification
        super(MobileNetWithSigmoid, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        num_classes = 96
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
           
        # Add sigmoid activation after the final fully connected layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.mobilenet(x)
        x = self.sigmoid(x)
        return x

def create_random_input_data():
    batch_size = 32

    channels = 3  
    height = 224    
    width = 224     
    random_image = torch.rand(channels, height, width)
    image_tensor_example = random_image.unsqueeze(0).repeat(batch_size,1,1,1)

    height = 96   
    #random_target = torch.randint(1,99, (height,))
    random_target = torch.rand(height)
    target_tensor_example = random_target.unsqueeze(0).repeat(batch_size,1)

    return image_tensor_example, target_tensor_example

def create_model():
    model = MobileNetWithSigmoid()
    
    nn.sigmoid = nn.Sigmoid()
    return model

def train_model(epochs, loss_function, optimizer):
    warmup_steps = 10
    scheduler = WarmUpLR(optimizer, warmup_steps)

    for epoch in range(epochs):
        for input, targets in dataloader:
            optimizer.zero_grad()

            outputs = model(input)

            loss = loss_function(outputs.float(), targets.float())

            loss.backward()

            optimizer.step()

            scheduler.step()

           
            print(f"LR: {scheduler.get_lr()[0]}")

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


image_tensor_example, target_tensor_example = create_random_input_data()

training_data_set = WellDataSet(image_tensor_example, target_tensor_example)
dataloader = DataLoader(training_data_set, batch_size=32, shuffle=True)

model = create_model()


loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

train_model(20, loss_function, optimizer)

