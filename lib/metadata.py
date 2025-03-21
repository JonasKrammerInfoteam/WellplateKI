from dataclasses import dataclass, field
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class Well:
    """Represents a single well in a well plate"""
    well_id: str
    is_filled: bool
    pipette_above_well: bool
    
    def __post_init__(self):
        # Optional validation can be added here
        if not isinstance(self.well_id, str):
            raise TypeError(f"well_id must be a string, got {type(self.well_id)}")
        
        if not isinstance(self.is_filled, bool):
            raise TypeError(f"is_filled must be a boolean, got {type(self.is_filled)}")
        
        if not isinstance(self.pipette_above_well, bool):
            raise TypeError(f"pipette_above_well must be a boolean, got {type(self.pipette_above_well)}")


@dataclass
class WellPlate:
    """Represents a well plate with multiple wells"""
    wells: List[Well] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, json_data):
        """Create a WellPlate instance from JSON data"""
        if isinstance(json_data, str):
            # Parse if it's a JSON string
            data = json.loads(json_data)
        elif isinstance(json_data, list):
            # Use directly if it's already a list
            data = json_data
        else:
            raise TypeError(f"Expected str or list, got {type(json_data)}")
        
        wells = [Well(**well_data) for well_data in data]
        return cls(wells=wells)
    
    @classmethod
    def from_file(cls, file_path):
        """Create a WellPlate instance from a JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls.from_json(data)
    
    def to_json(self):
        """Convert the WellPlate to a JSON string"""
        # Convert wells to dictionaries
        wells_dict = [
            {
                "well_id": well.well_id,
                "is_filled": well.is_filled,
                "pipette_above_well": well.pipette_above_well
            }
            for well in self.wells
        ]
        return json.dumps(wells_dict)
    
    def save_to_file(self, file_path):
        """Save the WellPlate to a JSON file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    def get_well(self, well_id):
        """Get a well by its ID"""
        for well in self.wells:
            if well.well_id == well_id:
                return well
        return None
    
    def get_filled_wells(self):
        """Get all wells that are filled"""
        return [well for well in self.wells if well.is_filled]
    
    def get_wells_with_pipette(self):
        """Get all wells that have a pipette above them"""
        return [well for well in self.wells if well.pipette_above_well]
    
    def get_well_status_summary(self):
        """Get a summary of well statuses"""
        total_wells = len(self.wells)
        filled_wells = len(self.get_filled_wells())
        pipette_wells = len(self.get_wells_with_pipette())
        
        return {
            "total_wells": total_wells,
            "filled_wells": filled_wells,
            "pipette_wells": pipette_wells,
            "empty_wells": total_wells - filled_wells,
            "percent_filled": (filled_wells / total_wells) * 100 if total_wells > 0 else 0
        }
