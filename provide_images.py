#!/usr/env/bin python3
import lib
import os
from time import time

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
    t0 = time()
    found_files = lib.file_utils.find_metadata_and_images(root_directory)
    t1 = time()
    print(f"Finished searching for files in {t1 - t0} seconds")
    
    # Put files into dataset
    print("Gathering image metadata. This may take a while ...")
    t0 = time()
    dataset = lib.trainingsset.TrainingDataset(seed=seed, batch_size=batch_size)
    for metadata, images in found_files.items():
        plate_id = dataset.add_plate(lib.metadata.WellPlate.from_file(metadata))
        for image in images:
            dataset.add_image(plate_id, lib.images.ImageMetadata.from_file(image))
    t1 = time()
    print(f"Finished gathering metadata in {t1 - t0} seconds")
    
    # save dataset
    #dataset.save_to_file("dataset.json")

    print("Shuffling the data ...")
    t0 = time()
    dataset.shuffle()
    t1 = time()
    print(f"Finished shuffling data in {t1 - t0} seconds")
    return dataset


if __name__ == "__main__":
    # Directory to search (change this to your target directory)
    root_directory = input("Enter the root directory to search: ")
    dataloader = load_trainingsdata(root_directory)

    epoch_size = dataloader.get_batches_per_epoch()
    print(f"Epoch size: {epoch_size}")
    for i in range(epoch_size):
        next = dataloader.next_batch()
        print(f"here: {i}; next: {next[0]}")
