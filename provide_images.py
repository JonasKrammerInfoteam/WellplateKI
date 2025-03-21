#!/usr/env/bin python3
import lib
import os

import lib.file_utils
import lib.images
import lib.metadata
import lib.trainingsset


def load_trainingsdata(
        root_directory: str,
        seed: int = 42,
        batch_size: int = 32):
    
    # Validate directory
    if not os.path.isdir(root_directory):
        print(f"Error: '{root_directory}' is not a valid directory")
        return
    
    # Find PNG files and metadata.json files
    print("Searching for files ...")
    found_files = lib.file_utils.find_metadata_and_images(root_directory)
    print("Finished searching for files")
    
    # Put files into dataset
    print("Parsing image metadata. This may take a while ...")
    dataset = lib.trainingsset.TrainingDataset(seed=seed, batch_size=batch_size)
    for metadata, images in found_files.items():
        plate_id = dataset.add_plate(lib.metadata.WellPlate.from_file(metadata))
        for image in images:
            dataset.add_image(plate_id, lib.images.ImageMetadata.from_file(image))
    print("Finished parsing metadata")
    
    # save dataset
    dataset.save_to_file("dataset.json")
    return dataset


if __name__ == "__main__":
    # Directory to search (change this to your target directory)
    root_directory = input("Enter the root directory to search: ")
    load_trainingsdata(root_directory)
