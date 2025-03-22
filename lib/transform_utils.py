import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from lib.metadata import WellPlate
import cv2


def image_to_tensor(image, normalize=True):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_transposed = img_rgb.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_transposed).contiguous()
    if normalize:
        tensor = tensor.float() / 255.0
    return tensor


def image_to_tensor(image, normalize=True):
    """
    Convert an image to a PyTorch tensor.
    
    Parameters:
    image_path (str): Path to the image file
    normalize (bool): Whether to normalize using ImageNet mean and std values
    
    Returns:
    torch.Tensor: PyTorch tensor representing the image
    """
    # Define the transformation pipeline
    transform_list = []
    
    # Convert to tensor (scales values to [0.0, 1.0])
    to_tensor = transforms.ToTensor()
    
    # Normalize with ImageNet mean and std if requested
    if normalize:
        # ImageNet mean and std values
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform_list.append(normalize)
    
    # Compose the transformations
    transform = transforms.Compose(transform_list)
    
    # Convert to tensor first
    tensor = to_tensor(image)
    
    # Apply any additional transformations
    if transform_list:
        tensor = transform(tensor)

    tensor.transpose(1, 2)
    #print(tensor.size())
    return tensor


def images_to_batch_tensor(images, normalize=True):
    """
    Convert multiple images to a batch tensor.
    
    Parameters:
    image_paths (list): List of image file paths
    normalize (bool): Whether to normalize using ImageNet mean and std values
    
    Returns:
    torch.Tensor: Batch of image tensors of shape [batch_size, channels, height, width]
    """
    tensors = [image_to_tensor(image, normalize) for image in images]
    return torch.stack(tensors)

def metadata_to_tensor(plate: WellPlate):
   """Convert well plate to tensor"""
   return torch.tensor([1 if well.is_filled else 0 for well in plate.wells])