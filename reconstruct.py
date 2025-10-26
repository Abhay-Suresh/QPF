# ---------------------------
# Helper to reconstruct coordinates from turn-based encoding
# ---------------------------
import numpy as np

def reconstruct_from_turns(turns, neighbors):
    """
    Reconstructs a list of 3D coordinates from a list of turn indices
    and a corresponding neighbor list.

    Args:
        turns (list): The list of turn indices.
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors)
                          that the indices refer to.

    Returns:
        list: A list of (x, y, z) tuples representing the structure.
    """
    coords = [(0, 0, 0)] # Start at the origin
    for t in turns:
        last_coord = coords[-1]
        
        # Get the specific vector from the provided neighbor list
        try:
            vector = neighbors[t]
        except IndexError:
            raise IndexError(f"Turn index {t} is out of bounds for the provided neighbor list (size {len(neighbors)}).")
        
        # Calculate the new coordinate by adding the vector
        new_coord = tuple(int(x) for x in np.add(last_coord, vector))
        
        coords.append(new_coord)
    return coords