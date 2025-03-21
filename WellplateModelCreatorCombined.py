import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torch.functional as F
import provide_images as datas


#640x480
class Embedding(nn.Module):
        def __init__(self):
            super(Embedding, self).__init__()
            self.Conv = torch.nn.Conv2d(3, 3, 3, 3, 1)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x = self.Conv.forward(x)
            x = self.relu.forward(x)
            return x

class EmbeddingMod(nn.Module):
        def __init__(self):
            super(EmbeddingMod, self).__init__()
            self.embed = Embedding()
            self.Expand = torch.nn.ConvTranspose2d(3, 3, 3, 3, 0)
    
        def forward(self, x):
            y = self.embed.forward(x)
            y = self.Expand.forward(y)
            return y[0:x.size(0),0:x.size(1),0:x.size(2)]

def EmbeddedTraining():

    # Example RGB data (e.g., 100 samples, each with 3 RGB values)
    # Shape: [batch_size, channels, height, width]
    rgb_values = torch.randint(0, 256, (100, 3, 640, 480), dtype=torch.float32)  # Random RGB values between 0 and 255

    # Normalize the RGB values to the range [0, 1]
    normalized_rgb_values = rgb_values / 255.0  # Divide by 255 to normalize

    # Example target data (e.g., 100 target values)


    # Flatten input to shape [100, 30] (3 channels * 1 height * 10 width)
    input_train = normalized_rgb_values#.view(100, -1)  # Flatten RGB data

    output_train = input_train

    from torch.utils.data import DataLoader, TensorDataset

    # Split the data into training and validation (in this case, same for simplicity)
    input_val = input_train  # Same as training for validation (for simplicity)
    output_val = output_train  # Same as training for validation (for simplicity)

    # Create TensorDataset and DataLoader for training and validation
    train_data = TensorDataset(input_train, output_train)
    val_data = TensorDataset(input_val, output_val)

    dataloader = datas.load_trainingsdata(".", 42, 64)

    # Initialize model, loss function, and optimizer
    embedding = EmbeddingMod()
    print(embedding.forward(torch.rand(3, 640, 480)).size())
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(embedding.parameters(), lr=0.001)

    epochs = 2
    batches = dataloader.get_batches_per_epoch()
    batchesV = dataloader.get_batches_per_epoch(False)
    # Training loop
    for epoch in range(epochs):
        # Training phase
        embedding.train()
        running_loss = 0.0
        for i in range(0, batches):
            optimizer.zero_grad()  # Zero the gradients
            targets, inputs = dataloader.next_batch()[1]
            
            # Explicitly calling model.forward
            outputs = embedding.forward(inputs)  # Forward pass using model.forward()
            
            loss = criterion.forward(outputs, inputs)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            running_loss += loss.item()

        # Validation phase
        embedding.eval()
        val_loss = 0.0
        with torch.no_grad():  # No need to track gradients during validation
            for i in range(batchesV):
                targets, inputs = dataloader.next_batch(False)[1]
                outputs = embedding.forward(inputs)  # Explicitly calling model.forward
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Print statistics
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/batches:.4f}, Validation Loss: {val_loss/batchesV:.4f}')

    # At the end of training, print out the entire train_data as tensors
    print("\nTrain Data (Tensors):")
    print("input_train Tensor:")
    print(input_train)  # Print the input RGB features tensor
    print("\noutput_train Tensor:")
    print(output_train)  # Print the target tensor
    return embedding.embed

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

def train_model(epochs, loss_function, optimizer, model, embedding, dataloader):
    warmup_steps = 10
    scheduler = WarmUpLR(optimizer, warmup_steps)
    batches = dataloader.get_batches_per_epoch()
    batchesV = dataloader.get_batches_per_epoch(False)

    for epoch in range(epochs):

        for i in range(0, batches):
            (batches, (targets, input)) = dataloader.next_batch()
            optimizer.zero_grad()

            input = embedding.forward(input)
            padding0 = 224 - input.size(0)
            padding1 = 224 - input.size(1)

            if padding0 > 0 and padding1 > 0:
                input = F.pad(input, (padding0, padding1), "constant", 0)

            outputs = model(input)

            loss = loss_function(outputs.squeeze(), targets.float())

            loss.backward()

            optimizer.step()

            scheduler.step()

           
            print(f"LR: {scheduler.get_lr()[0]}")

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
        embedding.eval()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # No need to track gradients during validation
            for i in range(batchesV):
                (batches, (targets, inputs)) = dataloader.next_batch(False)
                outputs = embedding.forward(inputs)  # Explicitly calling model.forward
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
    return


embedding = EmbeddedTraining()

#image_tensor_example, target_tensor_example = create_random_input_data()

dataloader = datas.load_trainingsdata(".", 42, 256)


model = create_model()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

train_model(50, loss_function, optimizer, model, embedding, dataloader)

(embedding, model)

embedding.save()
model.save()