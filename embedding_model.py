import torch
import torch.nn as nn
import torchvision
import torch.functional as F

model = torchvision.models.mobilenet_v3_large()

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
            x = self.embed.forward(x)
            x = self.Expand.forward(x)
            return x
        
embed = EmbeddingMod()
print(embed.forward(torch.rand(3, 640, 480)).size())