import random
import numpy as np

# ---------------------------
# Coordinate-based encoding (randomized & overlap-safe)
# ---------------------------
def encode_coordinate_based(sequence, neighbors):
    """
    Generates a list of 3D coordinates for a sequence, ensuring no overlaps.
    The choice of next coordinate is random from available neighbors.

    Args:
        sequence (list or str): The sequence of residues (e.g., "HHPPH").
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors, bcc_neighbors).

    Returns:
        list: A list of (x, y, z) tuples representing the structure.
    """
    coords = [(0, 0, 0)]
    for i in range(1, len(sequence)):
        free_neighbors = []
        for vec in neighbors:
            # Calculate the candidate coordinate
            candidate = tuple(int(x) for x in np.add(coords[-1], vec))
            # Add it to the list if it's not already in the chain
            if candidate not in coords:
                free_neighbors.append(candidate)
        
        if not free_neighbors:
            # This can happen if the chain folds back on itself and traps a point
            raise ValueError(f"No free neighbor available for residue {sequence[i]} at step {i}!")
    
        # Choose a random coordinate from the list of valid, free neighbors
        coords.append(random.choice(free_neighbors))
    return coords

# ---------------------------
# Turn-based encoding (overlap-safe)
# ---------------------------
def encode_turn_based(sequence, neighbors):
    """
    Generates a list of "turns" (indices from the neighbor list) for a sequence.
    Ensures no overlaps.

    Args:
        sequence (list or str): The sequence of residues (e.g., "HHPPH").
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors, bcc_neighbors).

    Returns:
        list: A list of integers, where each integer is the index of the
              neighbor vector used for that step.
    """
    turns = []
    coords = [(0, 0, 0)]
    for i in range(1, len(sequence)):
        # Find all neighbor indices and coordinates that don't collide with the existing chain
        free_neighbors = []
        for idx, vec in enumerate(neighbors):
            candidate = tuple(int(x) for x in np.add(coords[-1], vec))
            if candidate not in coords:
                free_neighbors.append((idx, candidate)) # Store (index, coordinate)
        
        if not free_neighbors:
            raise ValueError(f"No free neighbor available for residue {sequence[i]} at step {i}!")
        
        # Choose one of the valid neighbors randomly
        idx, candidate_coord = random.choice(free_neighbors)
        
        # Store the index of the chosen turn
        turns.append(idx)
        # Add the new coordinate to our chain to check for future overlaps
        coords.append(candidate_coord)
    return turns

# ---------------------------
# Main Encoding Function
# ---------------------------
def protein_encoding(sequence, neighbors, method):
    """
    Encodes a protein sequence into a lattice structure using the specified
    neighbor list and method.

    Args:
        sequence (list or str): The sequence of residues (e.g., "HHPPH").
        neighbors (list): The list of neighbor vectors to use 
                          (e.g., fcc_neighbors, bcc_neighbors).
        method (str, optional): The encoding method. 
                                "coordinate" (default) or "turn".

    Returns:
        list: A list of coordinates (tuples) or turns (indices) depending
              on the chosen method.
    """
    if not neighbors:
        raise ValueError("The 'neighbors' list cannot be empty.")
        
    if method == "coordinate":
        return encode_coordinate_based(sequence, neighbors)
    elif method == "turn":
        return encode_turn_based(sequence, neighbors)
    else:
        raise ValueError("Unknown encoding method. Use 'coordinate' or 'turn'.")