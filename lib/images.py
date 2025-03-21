from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import uuid
from enum import Enum, auto


class ImageModification(Enum):
    """Enum representing possible image modifications"""
    ORIGINAL = auto()
    CROPPED = auto()
    ROTATED = auto()
    FLIPPED = auto()
    NORMALIZED = auto()
    GRAYSCALE = auto()
    CONTRAST_ENHANCED = auto()
    NOISE_REDUCED = auto()
    AUGMENTED = auto()
    MASKED = auto()
    SEGMENTED = auto()
    OTHER = auto()


@dataclass
class ImageMetadata:
    """Represents metadata for a single image"""
    image_id: str  # Unique identifier
    path: Path  # Path to the image file
    size_bytes: int  # File size in bytes
    dimensions: tuple[int, int]  # Width, height
    modifications: List[ImageModification] = field(default_factory=list)
    
    def __post_init__(self):
        # Convert string path to Path object if needed
        if isinstance(self.path, str):
            self.path = Path(self.path)
        
        # Generate ID if not provided
        if not self.image_id:
            self.image_id = str(uuid.uuid4())
    
    @classmethod
    def from_file(cls, file_path, modifications=None, image_id=None):
        """Create an ImageMetadata instance from a file path"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Get file stats
        stats = path.stat()
        
        # Get image dimensions (using PIL if available, otherwise dummy values)
        try:
            from PIL import Image
            with Image.open(path) as img:
                dimensions = img.size
        except ImportError:
            # Fallback if PIL is not available
            dimensions = (0, 0)
            print("Warning: PIL not available, image dimensions set to (0, 0)")
        
        return cls(
            image_id=image_id or str(uuid.uuid4()),
            path=path,
            size_bytes=stats.st_size,
            dimensions=dimensions,
            modifications=modifications or []
        )
