import numpy as np
import math

# Based on the methodology in arXiv:2406.01547v1

# The neighbor lists from 'lattice.py' (bcc_neighbors, fcc_neighbors, etc.)
# would be imported here or passed into the function.

def calculate_qubits_paper_method(sequence_length, neighbors):
    """
    Calculates the total qubits required for a turn-based encoding
    based on the methodology in arXiv:2406.01547v1. [cite: 260]

    The function determines the calculation method based on the provided
    neighbor list (from lattice.py), matching the lattice types
    discussed or implied in the paper.

    - For FCC (12 neighbors), CPD (18 neighbors), and SC (6 neighbors),
      it uses the "orthogonal planes" method:
      Total Qubits = (ceil(log2(N_plane)) + 2) * (m - 1) [cite: 241, 242]

    - For BCC (8 neighbors), it uses the "direct encoding" method,
      which does not require plane selection qubits:
      Total Qubits = ceil(log2(N_total)) * (m - 1) [cite: 242, 257]

    Args:
        sequence_length (int): The number of residues (m) in the protein.
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors).

    Returns:
        tuple: (total_qubits, total_qubits_per_turn)
        
    Raises:
        ValueError: If the neighbor list does not match a known lattice
                    (SC, BCC, FCC, or CPD).
    """
    
    # m = number of beads (length of the sequence)
    m = sequence_length
    if m < 2:
        return 0, 0  # No turns possible for a sequence < 2

    # num_turns = m - 1 [cite: 242]
    num_turns = m - 1

    num_neighbors_total = len(neighbors)
    
    if num_neighbors_total == 12:  # FCC lattice
        # 12 neighbors total, but 4 directions per plane [cite: 61, 64]
        directions_per_plane = 4
        # Need log2(4) = 2 qubits for direction [cite: 200]
        qubits_for_direction_in_plane = math.ceil(math.log2(directions_per_plane))
        # Need 2 qubits for plane selection [cite: 231, 241]
        qubits_for_plane_selection = 2

    elif num_neighbors_total == 18:  # Cubic with Planar Diagonals (CPD)
        # 18 neighbors total, but 8 directions per plane [cite: 67, 71]
        directions_per_plane = 8
        # Need log2(8) = 3 qubits for direction [cite: 128]
        qubits_for_direction_in_plane = math.ceil(math.log2(directions_per_plane))
        # Need 2 qubits for plane selection [cite: 231, 241]
        qubits_for_plane_selection = 2

    elif num_neighbors_total == 6:  # Simple Cubic (SC) lattice
        # 6 neighbors total. This fits the orthogonal plane model.
        # e.g., in x-y plane, directions are (+x, -x, +y, -y).
        directions_per_plane = 4
        # Need log2(4) = 2 qubits for direction
        qubits_for_direction_in_plane = math.ceil(math.log2(directions_per_plane))
        # Need 2 qubits for plane selection [cite: 231, 241]
        qubits_for_plane_selection = 2

    elif num_neighbors_total == 8:  # BCC lattice
        # 8 neighbors total (body diagonals).
        # This fits the "direct encoding" method for non-orthogonal turns.
        # We encode all N=8 unique directions directly[cite: 257].
        total_unique_directions = 8
        # Need log2(8) = 3 qubits for direction [cite: 257]
        qubits_for_direction_in_plane = math.ceil(math.log2(total_unique_directions))
        # No plane selection qubits are needed 
        qubits_for_plane_selection = 0

    else:
        raise ValueError(
            f"Unrecognized lattice with {num_neighbors_total} neighbors. "
            "This function is configured for SC (6), BCC (8), FCC (12), or CPD (18)."
        )

    # Total qubits required for a single turn 
    total_qubits_per_turn = qubits_for_direction_in_plane + qubits_for_plane_selection
    
    # Final formula: (total_qubits_per_turn) * (num_turns) [cite: 242]
    total_qubits = total_qubits_per_turn * num_turns

    return total_qubits, total_qubits_per_turn

def get_neighbor_paper_map(neighbors):
    """
    Generates a map from neighbor vectors to the qubit string encoding
    described in arXiv:2406.01547v1.

    The qubit string is a concatenation of [Plane Bits][Direction Bits].
    - For SC, FCC, and CPD, this follows the orthogonal plane model 
      (e.g., "10" for z-x plane + "01" for direction) [cite: 107, 233, 235].
    - For BCC, this follows the direct encoding model 
      (e.g., "001") with no plane bits[cite: 248, 257].

    Args:
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors).

    Returns:
        dict: A map of {vector_tuple: qubit_string}

    Raises:
        ValueError: If the neighbor list does not match a known lattice
                    (SC, BCC, FCC, or CPD).
    """
    num_neighbors = len(neighbors)
    neighbor_map = {}

    # Define plane bits based on Table (13) [cite: 233, 235]
    # "01" = y-z plane
    # "10" = z-x plane
    # "11" = x-y plane
    plane_bits = {"yz": "01", "zx": "10", "xy": "11"}

    if num_neighbors == 12:  # FCC (12 neighbors)
        # 2 qubits for 4 directions [cite: 200, 241]
        # Vectors from paper Eq (1) [cite: 105] (using d=2 for (1,1,0))
        # k=0 ("00") -> Da=1,  Db=1
        # k=1 ("01") -> Da=-1, Db=1
        # k=2 ("10") -> Da=-1, Db=-1
        # k=3 ("11") -> Da=1,  Db=-1
        dir_map = {
            "00": (1, 1),
            "01": (-1, 1),
            "10": (-1, -1),
            "11": (1, -1),
        }
        
        # Map coordinates based on paper Eq (2) [cite: 107]
        for dir_bits, (da, db) in dir_map.items():
            # y-z plane: (0, Da, Db)
            neighbor_map[(0, da, db)] = plane_bits["yz"] + dir_bits
            # z-x plane: (Db, 0, Da)
            neighbor_map[(db, 0, da)] = plane_bits["zx"] + dir_bits
            # x-y plane: (Da, Db, 0)
            neighbor_map[(da, db, 0)] = plane_bits["xy"] + dir_bits

    elif num_neighbors == 18:  # CPD (18 neighbors)
        # 3 qubits for 8 directions [cite: 128]
        # Vectors from paper Eq (3) [cite: 113, 114, 123] (using d=1 for (1,1,0))
        da_vec = [1, 1, 0, -1, -1, -1, 0, 1]
        db_vec = [0, 1, 1, 1, 0, -1, -1, -1]
        
        # Map coordinates based on paper Eq (2) [cite: 107]
        for k in range(8):
            dir_bits = f"{k:03b}"
            da, db = da_vec[k], db_vec[k]
            
            # y-z plane: (0, Da, Db)
            vec_yz = (0, da, db)
            if vec_yz not in neighbor_map: # Assigns SC vectors (0,y,z)
                neighbor_map[vec_yz] = plane_bits["yz"] + dir_bits

            # z-x plane: (Db, 0, Da)
            vec_zx = (db, 0, da)
            if vec_zx not in neighbor_map: # Assigns SC vectors (x,0,z)
                neighbor_map[vec_zx] = plane_bits["zx"] + dir_bits

            # x-y plane: (Da, Db, 0)
            vec_xy = (da, db, 0)
            if vec_xy not in neighbor_map: # Assigns SC vectors (x,y,0)
                neighbor_map[vec_xy] = plane_bits["xy"] + dir_bits
                
    elif num_neighbors == 6:  # SC (6 neighbors)
        # Inferred from CPD logic: 4 directions per plane
        # 2 qubits for 4 directions (e.g., +x, -x, +y, -y)
        # We define a logical mapping for the 4 axial directions
        # k=0 ("00") -> (1, 0)
        # k=1 ("01") -> (0, 1)
        # k=2 ("10") -> (-1, 0)
        # k=3 ("11") -> (0, -1)
        
        # We must uniquely assign the 6 vectors.
        # We assign (x,y) vectors to the x-y plane,
        # and (z) vectors to the z-x plane.
        neighbor_map = {
            # x-y plane, (Da, Db, 0) [cite: 107]
            (1, 0, 0):  plane_bits["xy"] + "00", # k=0 -> (1, 0)
            (-1, 0, 0): plane_bits["xy"] + "10", # k=2 -> (-1, 0)
            (0, 1, 0):  plane_bits["xy"] + "01", # k=1 -> (0, 1)
            (0, -1, 0): plane_bits["xy"] + "11", # k=3 -> (0, -1)
            
            # z-x plane, (Db, 0, Da) [cite: 107]
            (0, 0, 1):  plane_bits["zx"] + "01", # k=1 -> (Da=1, Db=0)
            (0, 0, -1): plane_bits["zx"] + "11", # k=3 -> (Da=-1, Db=0)
        }

    elif num_neighbors == 8:  # BCC (8 neighbors)
        # "do away with the qubits involved in plane selection" 
        # "directly go for encoding directions" 
        # N=8 unique directions, so log2(8)=3 qubits 
        
        # We must map to the list in the same order as lattice.py
        # to be consistent.
        bcc_vectors = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        # Verify the input list matches the expected BCC vectors
        if set(neighbors) != set(bcc_vectors):
             raise ValueError(
                 "Input 8-neighbor list does not match internal BCC definition."
             )
        
        for k, vec in enumerate(bcc_vectors):
            neighbor_map[vec] = f"{k:03b}" # e.g., "000", "001", ...

    else:
        raise ValueError(
            f"Unrecognized lattice with {num_neighbors} neighbors. "
            "This function is configured for SC (6), BCC (8), FCC (12), or CPD (18)."
        )

    # Final validation
    if set(neighbor_map.keys()) != set(neighbors):
        raise AssertionError(
            f"Map generation failed. Expected {len(neighbors)} keys, "
            f"but got {len(neighbor_map.keys())}. "
            "Input neighbors may not match internal definitions."
        )

    return neighbor_map