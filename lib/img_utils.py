from PIL import Image
import cv2


def open_and_normalize_image(input_path, target_width, target_height):
    img = Image.open(input_path)
    orig_width, orig_height = img.size
    if orig_width == target_width and orig_height == target_height:
        return img.copy()

    orig_aspect = orig_width / orig_height
    target_aspect = target_width / target_height
    epsilon = 0.01

    if abs(orig_aspect - target_aspect) < epsilon:
        return img.resize((target_width, target_height), resample=Image.BILINEAR)
    elif orig_aspect > target_aspect:  # Original is wider
        # Calculate what portion of the image to keep
        crop_width = int(orig_height * target_aspect)
        left = (orig_width - crop_width) // 2
        
        # Resize directly from the cropped portion
        return img.resize((target_width, target_height), 
                          box=(left, 0, left + crop_width, orig_height),
                          resample=Image.BILINEAR)
    else:  # Original is taller
        # Calculate what portion of the image to keep
        crop_height = int(orig_width / target_aspect)
        top = (orig_height - crop_height) // 2
        
        # Resize directly from the cropped portion
        return img.resize((target_width, target_height), 
                          box=(0, top, orig_width, top + crop_height),
                          resample=Image.BILINEAR)


def open_and_normalize_image_cv(input_path, target_width, target_height):
    # Read image with OpenCV (loads as BGR)
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not open image at {input_path}")
    
    # Get original dimensions
    orig_height, orig_width = img.shape[:2]
    if orig_width == target_width and orig_height == target_height:
        return img
    
    # Calculate aspect ratios
    orig_aspect = orig_width / orig_height
    target_aspect = target_width / target_height
    epsilon = 0.01
    
    if abs(orig_aspect - target_aspect) < epsilon:
        # Similar aspect ratios, just resize
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    elif orig_aspect > target_aspect:  # Original is wider
        # Calculate what portion of the image to keep
        crop_width = int(orig_height * target_aspect)
        left = (orig_width - crop_width) // 2
        
        # Crop and resize
        cropped = img[:, left:left+crop_width]
        return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:  # Original is taller
        # Calculate what portion of the image to keep
        crop_height = int(orig_width / target_aspect)
        top = (orig_height - crop_height) // 2
        
        # Crop and resize
        cropped = img[top:top+crop_height, :]
        return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)