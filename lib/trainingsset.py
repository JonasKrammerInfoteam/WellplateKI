from lib.metadata import *
from lib.images import *
from lib.img_utils import *
from lib.transform_utils import *

from dataclasses import dataclass, field
from typing import Tuple, Any
import json
import random
from PIL import Image
from torch import Tensor


def ceildiv(a, b):
    return -(a // -b)


@dataclass
class TrainingDataset:
    """Represents a complete training dataset with multiple well plates"""
    seed: int = 42
    batch_size: int = 100
    prefetch: int = 1
    plates: List[Tuple[WellPlate, List[ImageMetadata]]] = field(default_factory=list)

    __shuffled: bool = False
    __total_images: int = 0
    __batches_per_epoch: int = 0
    
    def add_plate(self, plate):
        """Add a well plate to this dataset"""
        if self.__shuffled:
            raise RuntimeError("Cannot add data now!")
        for i, p in enumerate(self.plates):
            if plate == p[0]:
                return i
        self.plates.append((plate, []))
        return len(self.plates) - 1
    
    def get_plate(self, plate_id):
        """Get a plate by its ID"""
        return self.plates[plate_id][0]
    
    def add_image(self, plate_id, image):
        """Add an image to a well plate to this dataset"""
        if self.__shuffled:
            raise RuntimeError("Cannot add data now!")
        self.plates[plate_id][1].append(image)
        return len(self.plates[plate_id][1]) - 1
    
    def get_image(self, plate_id, image_id):
        """Get an image by its ID"""
        return self.plates[plate_id][1][image_id]

    def shuffle(self):
        """Shuffles the dataset according to the seed. Breaks the IDs!!!!!"""
        self.plates = [plate for plate in self.plates if len(plate[1]) > 0]
        random.seed(self.seed)
        for pair in self.plates:
            random.shuffle(pair[1])
        random.shuffle(self.plates)
        self.__shuffled = True
        self.__total_images = sum([len(images) for _, images in self.plates])
        print(f"Total images: {self.__total_images}")
        self.__min_bucket = min([len(plate[1]) for plate in self.plates])
        print(f"smallest bucket: {self.__min_bucket}")
        self.__max_bucket = max([len(plate[1]) for plate in self.plates])
        print(f"biggest bucket: {self.__max_bucket}")
        self.__batches_per_epoch = ceildiv((self.__min_bucket * len(self.plates) * 8) // 10, self.batch_size)
        self.__current_batch = 0
        self.__current_validation_batch = ceildiv((self.__batches_per_epoch * 8), 10)
        self.__transit_per_batch = ceildiv(self.batch_size, len(self.plates))
        print(f"Batch size: {self.batch_size}")
        print(f"Bucket count: {len(self.plates)}")
        print(f"transit through buckets per batch: {self.__transit_per_batch}")

    def calc_index(self, is_training):
        if is_training:
            primary_idx = self.__current_batch * self.batch_size
            secondary_idx = primary_idx // len(self.plates)
            primary_idx %= len(self.plates)
        else:
            primary_idx = self.__current_validation_batch * self.batch_size
            secondary_idx = primary_idx // len(self.plates)
            primary_idx %= len(self.plates)
        return primary_idx, secondary_idx

    def next_batch(self, is_training = True) -> Tuple[int, Tuple[List[Tensor], List[Tensor]]]:
        """Loads the next batch of data"""
        if not self.__shuffled:
            self.shuffle()
        primary_idx, secondary_idx = self.calc_index(is_training)

        targets = []
        inputs = []
        j = 0
        for i in range(self.batch_size):
            img_path = self.plates[(primary_idx + i) % len(self.plates)][1][secondary_idx + j].path
            img = open_and_normalize_image(img_path, 640, 480)
            inputs.append(image_to_tensor(img, False))
            targets.append(metadata_to_tensor(self.plates[(primary_idx + i) % len(self.plates)][0]))
            j = (primary_idx + i) // len(self.plates)

        if is_training:
            self.__current_batch += 1
            primary_idx, secondary_idx = self.calc_index(is_training)
            if self.__current_batch >= self.__batches_per_epoch * 0.8 or secondary_idx + self.__transit_per_batch >= (self.__min_bucket * 8) // 10 - 1:
                self.__current_batch = 0
                self.shuffle()
        else:
            self.__current_validation_batch += 1
            primary_idx, secondary_idx = self.calc_index(is_training)
            if self.__current_validation_batch >= self.__batches_per_epoch or secondary_idx + self.__transit_per_batch >= self.__min_bucket - 1:
                self.__current_validation_batch = ceildiv((self.__batches_per_epoch * 8), 10)
                self.shuffle()
        return (self.__current_batch if is_training else self.__current_validation_batch, (torch.stack(targets), torch.stack(inputs)))

    def get_batches_per_epoch(self, is_training = True) -> int:
        """Get the count of batches per epoch"""
        if not self.__shuffled:
            self.shuffle()
        train = ceildiv(self.__batches_per_epoch * 8, 10)
        if is_training:
            return train - 1 # subtract as we have one incomplete bucket?
        return self.__batches_per_epoch - train - 1 # subtract as we have one incomplete bucket?

    def to_json(self):
        """Convert the dataset to a JSON representation"""
        plates_list = []
        
        for plate, images in self.plates:
            wells_list = []
            for well in plate.wells:
                wells_list.append({
                    "well_id": well.well_id,
                    "is_filled": well.is_filled,
                    "pipette_above_well": well.pipette_above_well
                })
                
            images_list = []
            for img in images:
                images_list.append({
                    "image_id": img.image_id,
                    "path": str(img.path),
                    "size_bytes": img.size_bytes,
                    "dimensions": img.dimensions,
                    "modifications": [mod.name for mod in img.modifications]
                })
            
            plates_list.append({
                "plate": wells_list,
                "images": images_list
            })
        
        return json.dumps(plates_list, indent=2)
    
    def save_to_file(self, file_path):
        """Save the dataset to a JSON file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_json(cls, json_data):
        """Create a dataset from JSON data"""
        raise RuntimeError("Not implemented!")
