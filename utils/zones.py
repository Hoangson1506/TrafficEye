import json
import os
import numpy as np

def load_zones(zone_path="zones.json"):
    """
    Load zones from a JSON file.
    Returns:
        dict: zone configuration with 'polygon', 'lines', 'lines_config', and 'light_zones'.
    """
    default_zones = {
        "polygon": [], 
        "lines": [],
        "lines_config": {},
        "light_zones": {
            "straight": [],
            "left": [],
            "right": []
        }
    }
    if not os.path.exists(zone_path):
        return default_zones
    
    try:
        with open(zone_path, 'r') as f:
            zones = json.load(f)
        # Ensure light_zones key exists for backward compatibility
        if "light_zones" not in zones:
            zones["light_zones"] = default_zones["light_zones"]
        return zones
    except Exception as e:
        print(f"Error loading zones: {e}")
        return default_zones

def save_zones(zones, zone_path="zones.json"):
    """
    Save zones to a JSON file.
    Args:
        zones (dict): Dictionary containing 'polygon' (list of points) and 'lines' (list of points).
    """
    try:
        # Convert numpy arrays/tuples to lists if necessary
        serializable_zones = {}
        for key, val in zones.items():
            if isinstance(val, (np.ndarray, list)):
                # Ensure points are lists of lists/ints
                serializable_zones[key] = [list(map(int, pt)) if isinstance(pt, (tuple, list, np.ndarray)) else pt for pt in val]
            else:
                serializable_zones[key] = val

        with open(zone_path, 'w') as f:
            json.dump(serializable_zones, f, indent=4)
        print(f"Zones saved to {zone_path}")
        return True
    except Exception as e:
        print(f"Error saving zones: {e}")
        return False
