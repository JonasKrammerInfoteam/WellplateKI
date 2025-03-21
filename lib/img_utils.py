from PIL import Image


def open_and_normalize_image(input_path, target_width, target_height):
    # Open the image
    img = Image.open(input_path)
    
    # Get original dimensions
    orig_width, orig_height = img.size
    
    # Calculate original aspect ratio
    orig_aspect = orig_width / orig_height
    
    # Calculate target aspect ratio
    target_aspect = target_width / target_height
    
    # If aspect ratios don't match, crop from center
    if orig_aspect > target_aspect:  # Original is wider
        # Calculate width to crop to
        new_width = int(orig_height * target_aspect)
        left = (orig_width - new_width) // 2
        right = left + new_width
        top, bottom = 0, orig_height
        img = img.crop((left, top, right, bottom))
    elif orig_aspect < target_aspect:  # Original is taller
        # Calculate height to crop to
        new_height = int(orig_width / target_aspect)
        top = (orig_height - new_height) // 2
        bottom = top + new_height
        left, right = 0, orig_width
        img = img.crop((left, top, right, bottom))
    
    # Now resize to the target dimensions
    resized_img = img.resize((target_width, target_height), Image.LANCZOS)
    
    # Save the resized image
    return resized_img
