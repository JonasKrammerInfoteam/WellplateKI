import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Load pre-trained MobileNetV3
#model = models.mobilenet_v3_large(pretrained=True)

# Set the model to evaluation mode (important when saving)
#model.eval()

# Define a simple model with a single linear layer
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # 10 input features, 1 output

    def forward(self, x):
        return self.fc(x)
    
rgb_tensor = torch.tensor([[[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Image 1 (RGB)
                            [[255, 255, 0], [0, 255, 255], [255, 0, 255]]],  # Image 2 (RGB)
                           [[[255, 127, 0], [0, 127, 255], [127, 255, 127]],  # Image 3 (RGB)
                            [[255, 255, 255], [127, 127, 127], [0, 0, 0]]]])  # Image 3 (RGB)

# Normalize the RGB values to the range [0, 1] by dividing by 255.0
normalized_rgb_tensor = rgb_tensor / 255.0

from torch.utils.data import DataLoader, TensorDataset

# Example training data (input features and target)
x_train = normalized_rgb_tensor  # 100 samples, 10 features each
y_train = torch.randn(100, 1)   # 100 target values

#x_val = torch.randn(20, 10)  # Validation data
#y_val = torch.randn(20, 1)
x_val = x_train #Same for validation
y_val = y_train

train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = SimpleLinearModel()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer

epochs = 100

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        
        # Explicitly calling model.forward
        outputs = model.forward(inputs)  # Forward pass using model.forward()
        
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # No need to track gradients during validation
        for inputs, targets in val_loader:
            outputs = model.forward(inputs)  # Explicitly calling model.forward
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Print statistics
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')



print("\nTrain Data (Tensors):")
print("x_train Tensor:")
print(x_train)  # Print the input features tensor
print("\ny_train Tensor:")
print(y_train)  # Print the target tensor