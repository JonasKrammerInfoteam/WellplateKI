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
        self.fc = nn.Linear(30, 1)  # 30 input features, (flattened RGB), 1 output

    def forward(self, x):
        return self.fc(x)
    
# Example RGB data (e.g., 100 samples, each with 3 RGB values)
# Shape: [batch_size, channels, height, width]
rgb_values = torch.randint(0, 256, (100, 3, 1, 10), dtype=torch.float32)  # Random RGB values between 0 and 255

# Normalize the RGB values to the range [0, 1]
normalized_rgb_values = rgb_values / 255.0  # Divide by 255 to normalize

# Example target data (e.g., 100 target values)
output_train = torch.randn(100, 1)  # 100 target values (randomly generated)

# Flatten input to shape [100, 30] (3 channels * 1 height * 10 width)
input_train = normalized_rgb_values.view(100, -1)  # Flatten RGB data

from torch.utils.data import DataLoader, TensorDataset

# Split the data into training and validation (in this case, same for simplicity)
input_val = input_train  # Same as training for validation (for simplicity)
output_val = output_train  # Same as training for validation (for simplicity)

# Create TensorDataset and DataLoader for training and validation
train_data = TensorDataset(input_train, output_train)
val_data = TensorDataset(input_val, output_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Initialize model, loss function, and optimizer
model = SimpleLinearModel()
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

# Training loop
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

# At the end of training, print out the entire train_data as tensors
print("\nTrain Data (Tensors):")
print("input_train Tensor:")
print(input_train)  # Print the input RGB features tensor
print("\noutput_train Tensor:")
print(output_train)  # Print the target tensor