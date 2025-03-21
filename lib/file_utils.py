import os
import glob
from typing import Dict, List


def find_files(root_dir, file_types):
    """
    Recursively find all files of specified types in a directory
    
    Args:
        root_dir (str): Root directory to start search
        file_types (list): List of file extensions or names to search for
        
    Returns:
        dict: Dictionary with file types as keys and lists of file paths as values
    """
    results = {file_type: [] for file_type in file_types}
    
    # Walk through all directories
    for root, _, _ in os.walk(root_dir):
        # Check for each file type
        for file_type in file_types:
            # Handle both extension (*.png) and filename (metadata.json) patterns
            if file_type.startswith('*.'):
                pattern = os.path.join(root, file_type)
                files = glob.glob(pattern)
                results[file_type].extend(files)
            else:
                # For specific filenames like metadata.json
                filepath = os.path.join(root, file_type)
                if os.path.exists(filepath):
                    results[file_type].append(filepath)
    
    return results


def find_metadata_and_images(root_dir: str) -> Dict[str, List[str]]:
    """
    Recursively finds all metadata.json files and their associated images in the given directory.
    
    Args:
        root_dir (str): The root directory to start the search from
        
    Returns:
        Dict[str, List[str]]: A dictionary where:
            - keys are the full paths to metadata.json files
            - values are lists of full paths to image files in the 'images' subdirectory
    """
    result = {}
    
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if metadata.json exists in the current directory
        if "metadata.json" in filenames:
            metadata_path = os.path.join(dirpath, "metadata.json")
    
            # Look for an 'images' directory in the same directory as metadata.json
            images_dir = os.path.join(dirpath, "images")
            if os.path.isdir(images_dir):
                # Get all files in the images directory
                image_files = []
                for img_file in os.listdir(images_dir):
                    img_path = os.path.join(images_dir, img_file)
                    if os.path.isfile(img_path):
                        image_files.append(img_path)
                
                # Add to result dictionary
                result[metadata_path] = image_files
    
    return result