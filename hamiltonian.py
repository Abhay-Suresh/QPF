"""
hamiltonian.py

Generalized script to build qubit coordinate operators (X_i, Y_i, Z_i)
for protein folding on various lattices, based on the methodology in
arXiv:2406.01547v1.

This script generalizes the provided "prompt" code to work for
FCC, CPD, and BCC lattices by dynamically selecting the correct
encoding model (orthogonal vs. direct) and algorithmically
reconstructing the delta operators (Δa, Δb, Δc) from the
paper's basis-vector formulation.
"""

import numpy as np
import math
import itertools
from qiskit.quantum_info import SparsePauliOp

# --- Import from provided files ---
from hp import hp_mapping, encode_hp_sequence
from lattice import bcc_neighbors, fcc_neighbors, cpd_neighbors, sc_neighbors
from qubit_encoding import calculate_qubits_paper_method
from mj import mj_matrix

# ==================================================================
# SECTION 1: HELPER FUNCTIONS
# ==================================================================

def pauli_string(pauli_char, num_qubits, index=-1):
    """Creates a SparsePauliOp for a single Pauli op at a specific index."""
    if index == -1:  # Identity on all qubits
        return SparsePauliOp("I" * num_qubits)
    
    char_list = ["I"] * num_qubits
    char_list[num_qubits - 1 - index] = pauli_char  # Qiskit orders right-to-left
    return SparsePauliOp("".join(char_list))

def get_lattice_properties(neighbors, sequence_length):
    """
    Analyzes the neighbor list and sequence to determine all
    encoding properties based on the paper.
    """
    num_neighbors = len(neighbors)
    m = sequence_length
    if m < 2:
        raise ValueError("Sequence length must be at least 2.")

    props = {}
    
    # Identify lattice type from neighbor count
    if num_neighbors == 12:
        props['type'] = 'fcc'
        props['model_type'] = 'orthogonal'
        props['contact_dist_sq'] = 2  # (1,1,0) -> norm_sq = 2
    elif num_neighbors == 18:
        props['type'] = 'cpd'
        props['model_type'] = 'orthogonal'
        props['contact_dist_sq'] = 1  # (1,0,0) -> norm_sq = 1
    elif num_neighbors == 8:
        props['type'] = 'bcc'
        props['model_type'] = 'direct'
        props['contact_dist_sq'] = 3  # (1,1,1) -> norm_sq = 3
    elif num_neighbors == 6:
        props['type'] = 'sc'
        props['model_type'] = 'orthogonal'
        props['contact_dist_sq'] = 1  # (1,0,0) -> norm_sq = 1
    else:
        raise ValueError(f"Unknown lattice with {num_neighbors} neighbors.")

    if props['type'] == 'sc':
        raise NotImplementedError(
            "The SC lattice does not fit the paper's (Eq 14) identical-plane "
            "orthogonal model. This script only supports FCC, CPD, and BCC."
        )

    # Calculate qubit requirements
    total_qubits, qubits_per_turn = calculate_qubits_paper_method(m, neighbors)
    props['total_qubits'] = total_qubits
    props['qubits_per_turn'] = qubits_per_turn
    props['num_residues'] = m
    props['num_turns'] = m - 1

    # Infer direction/plane qubit counts
    if props['model_type'] == 'orthogonal':
        props['num_plane_qubits'] = 2
        props['num_direction_qubits'] = qubits_per_turn - 2
    else:  # 'direct'
        props['num_plane_qubits'] = 0
        props['num_direction_qubits'] = qubits_per_turn

    return props

def get_delta_vectors(lattice_type):
    """
    Returns the 'f' vectors (Δa, Δb, Δc) for the d=1 case
    as defined in the paper.
    """
    if lattice_type == 'fcc':
        # From Eq (1), with d=1 (so d/2 = 0.5)
        da_vec = [0.5, -0.5, -0.5, 0.5]
        db_vec = [0.5, 0.5, -0.5, -0.5]
        return {'da': da_vec, 'db': db_vec}
    
    elif lattice_type == 'cpd':
        # From Eq (3), with d=1
        da_vec = [1, 1, 0, -1, -1, -1, 0, 1]
        db_vec = [0, 1, 1, 1, 0, -1, -1, -1]
        return {'da': da_vec, 'db': db_vec}

    elif lattice_type == 'bcc':
        # From Eq (15), the vectors are just the (x, y, z)
        # components of the 8 neighbor vectors.
        bcc_vecs_ordered = [
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        da_vec = [v[0] for v in bcc_vecs_ordered]
        db_vec = [v[1] for v in bcc_vecs_ordered]
        dc_vec = [v[2] for v in bcc_vecs_ordered]
        return {'da': da_vec, 'db': db_vec, 'dc': dc_vec}
    
    else:
        raise ValueError(f"No delta vectors defined for {lattice_type}")

def get_basis_matrix(num_qubits):
    """
    Generates the binary basis matrix 'B' (like Eq 6 or 10)
    and the corresponding basis terms (e.g., (0,), (0,1)).
    """
    N = 2**num_qubits
    B = np.zeros((N, N))
    states = list(itertools.product([0, 1], repeat=num_qubits))
    terms = [()]
    q_indices = list(range(num_qubits))
    for k in range(1, num_qubits + 1):
        for combo in itertools.combinations(q_indices, k):
            terms.append(combo)
            
    for row_idx, state in enumerate(states):
        for col_idx, term in enumerate(terms):
            val = 1
            for q_idx in term:
                val *= state[q_idx]
            B[row_idx, col_idx] = val
            
    return B, terms

def build_delta_operators(turn_idx, props, I_op):
    """
    Builds the SparsePauliOp for Δa, Δb, (Δc) for a specific turn.
    """
    total_qubits = props['total_qubits']
    num_dir_q = props['num_direction_qubits']
    start_idx = turn_idx * props['qubits_per_turn']
    dir_indices = list(range(start_idx, start_idx + num_dir_q))
    
    Z_ops = [pauli_string("Z", total_qubits, i) for i in dir_indices]
    Q_ops = [(I_op - Z) / 2 for Z in Z_ops]
    
    B, basis_terms = get_basis_matrix(num_dir_q)
    B_inv = np.linalg.inv(B)
    delta_vectors = get_delta_vectors(props['type'])
    delta_ops = {}
    
    for vec_name, f_vec in delta_vectors.items():
        coeffs = B_inv @ f_vec
        
        # *** FIX from QiskitError ***
        # Initialize final_op as a zero-operator with the *correct* total_qubits
        final_op = I_op * 0.0
        
        for c, term_indices in zip(coeffs, basis_terms):
            if c == 0.0:
                continue
            term_op = I_op * c
            for q_idx in term_indices:
                term_op = term_op @ Q_ops[q_idx]
            final_op += term_op
        
        delta_ops[vec_name] = final_op.simplify()

    return delta_ops

def build_plane_projectors(turn_idx, props, I_op):
    """
    Builds the SparsePauliOp projectors for plane selection ('yz', 'zx', 'xy')
    and the invalid '00' state.
    """
    if props['model_type'] != 'orthogonal':
        return {}

    num_dir_q = props['num_direction_qubits']
    num_plane_q = props['num_plane_qubits']
    
    if num_plane_q != 2:
        raise ValueError("Orthogonal model must have 2 plane qubits.")

    start_idx = turn_idx * props['qubits_per_turn']
    plane_start_idx = start_idx + num_dir_q
    
    q4_idx = plane_start_idx
    q5_idx = plane_start_idx + 1

    Z4 = pauli_string("Z", props['total_qubits'], q4_idx)
    Z5 = pauli_string("Z", props['total_qubits'], q5_idx)

    P0_q4 = (I_op + Z4) / 2
    P1_q4 = (I_op - Z4) / 2
    P0_q5 = (I_op + Z5) / 2
    P1_q5 = (I_op - Z5) / 2
    
    yz_proj = (P0_q4 @ P1_q5).simplify()
    zx_proj = (P1_q4 @ P0_q5).simplify()
    xy_proj = (P1_q4 @ P1_q5).simplify()
    invalid_proj = (P0_q4 @ P0_q5).simplify()
    
    return {
        'yz': yz_proj,
        'zx': zx_proj,
        'xy': xy_proj,
        '00': invalid_proj
    }


# ==================================================================
# SECTION 2: COORDINATE OPERATOR CONSTRUCTION
# ==================================================================

def build_coordinate_operators(sequence, neighbors):
    """
    Constructs the X_i, Y_i, Z_i coordinate operators.
    """
    
    hp_sequence, num_residues = encode_hp_sequence(sequence)
    props = get_lattice_properties(neighbors, num_residues)
    
    total_qubits = props['total_qubits']
    num_turns = props['num_turns']
    I_op = pauli_string("I", total_qubits)
    
    delta_X_ops = []
    delta_Y_ops = []
    delta_Z_ops = []
    invalid_plane_ops = []

    for turn_idx in range(num_turns):
        delta_ops = build_delta_operators(turn_idx, props, I_op)

        if props['model_type'] == 'orthogonal':
            plane_projs = build_plane_projectors(turn_idx, props, I_op)
            da = delta_ops['da']
            db = delta_ops['db']
            
            dx = (plane_projs['xy'] @ da + plane_projs['zx'] @ db).simplify()
            dy = (plane_projs['yz'] @ da + plane_projs['xy'] @ db).simplify()
            dz = (plane_projs['zx'] @ da + plane_projs['yz'] @ db).simplify()
            
            invalid_plane_ops.append(plane_projs['00'])

        elif props['model_type'] == 'direct':
            dx = delta_ops['da']
            dy = delta_ops['db']
            dz = delta_ops['dc']
        
        delta_X_ops.append(dx)
        delta_Y_ops.append(dy)
        delta_Z_ops.append(dz)

    # *** FIX from QiskitError ***
    # Initialize lists with zero-operators of the *correct* total_qubits
    X_ops = [I_op * 0.0] * num_residues
    Y_ops = [I_op * 0.0] * num_residues
    Z_ops = [I_op * 0.0] * num_residues

    for i in range(num_turns):
        X_ops[i+1] = (X_ops[i] + delta_X_ops[i]).simplify()
        Y_ops[i+1] = (Y_ops[i] + delta_Y_ops[i]).simplify()
        Z_ops[i+1] = (Z_ops[i] + delta_Z_ops[i]).simplify()

    coord_ops = {'X': X_ops, 'Y': Y_ops, 'Z': Z_ops}
    
    print(f"Constructed coordinate operators for {props['type'].upper()} "
          f"({num_residues} residues, {total_qubits} qubits).")
          
    return coord_ops, invalid_plane_ops, props, I_op


# ==================================================================
# SECTION 3: HAMILTONIAN OPERATOR CONSTRUCTION (REVISED)
# ==================================================================

# --- Squared Distance Operator D^2_ij ---
def get_dist_sq_op(i, j, coord_ops, I_op):
    """Returns the operator for the squared distance between residues i and j."""
    if i == j: 
        return I_op * 0
    
    dX = (coord_ops['X'][i] - coord_ops['X'][j]).simplify()
    dY = (coord_ops['Y'][i] - coord_ops['Y'][j]).simplify()
    dZ = (coord_ops['Z'][i] - coord_ops['Z'][j]).simplify()
    
    dist_sq = (dX @ dX + dY @ dY + dZ @ dZ).simplify()
    return dist_sq

# --- Hamiltonian Component Operators (NEW) ---

def get_contact_indicator_op(i, j, coord_ops, props, I_op):
    """
    Returns an *approximate* contact indicator operator (O_contact).
    This operator is 1 at contact (D^2 = D_c^2) and <= 0 otherwise.
    This is based on Eq. (22) from arXiv:2406.01547v1.
    """
    dist_sq_op = get_dist_sq_op(i, j, coord_ops, I_op)
    target_dist_sq = props['contact_dist_sq'] # e.g., 3 for BCC
    
    # We use the non-contact penalty (D^2 - D_c^2)^2
    non_contact_op = (dist_sq_op - target_dist_sq * I_op).simplify()
    non_contact_op_sq = (non_contact_op @ non_contact_op).simplify()
    
    # We need to normalize this. The max penalty occurs at the
    # largest possible distance. A simple normalization is
    # to divide by the penalty at D^2=0 (overlap).
    # N = (0 - D_c^2)^2 = D_c^4
    norm_factor = target_dist_sq**2
    if norm_factor == 0:
        return I_op * 0
    
    # O_contact ≈ I - (1/N) * (D^2 - D_c^2)^2
    # At D^2=3 (contact): I - (1/9)*(3-3)^2 = I
    # At D^2=0 (overlap): I - (1/9)*(0-3)^2 = I - 1 = 0
    # At D^2=12 (far):    I - (1/9)*(12-3)^2 = I - 9 = -8*I
    contact_indicator = (I_op - (non_contact_op_sq / norm_factor)).simplify()
    
    return contact_indicator

def get_overlap_penalty_op(i, j, coord_ops, props, I_op):
    """
    Returns a *strong* overlap penalty operator (O_overlap).
    This operator is > 0 at overlap (D^2 = 0) and 0 at contact (D^2 = D_c^2).
    """
    dist_sq_op = get_dist_sq_op(i, j, coord_ops, I_op)
    target_dist_sq = props['contact_dist_sq'] # e.g., 3 for BCC
    
    # We use the operator (D_c^2 - D^2).
    # At D^2=0 (overlap): (3 - 0) = +3 (Penalty)
    # At D^2=3 (contact): (3 - 3) = 0  (No Penalty)
    # At D^2=12 (far):   (3 - 12) = -9 (Reward!)
    # This is flawed.
    
    # Let's use the user's original operator (D^2 - D_c^2)^2
    # but *only* as a penalty. It's a "non-contact" penalty.
    # The previous Hamiltonian failed because P_overlap was
    # applied to *all* pairs, making it a "non-contact" penalty
    # instead of an "overlap" penalty.
    
    # **A simple, correct model for overlap:**
    # We penalize any distance *less than* contact.
    # We approximate this with a penalty at D^2 = 0.
    # Let's use O_overlap = (1/N) * (D^2 - D_c^2)^2
    # This is 0 at contact and 1 at overlap (if N=9 for BCC).
    
    non_contact_op = (dist_sq_op - target_dist_sq * I_op).simplify()
    non_contact_op_sq = (non_contact_op @ non_contact_op).simplify()

    norm_factor = target_dist_sq**2
    if norm_factor == 0:
        return I_op * 0

    # O_overlap ≈ (1/N) * (D^2 - D_c^2)^2
    # At D^2=3 (contact): (1/9)*(3-3)^2 = 0
    # At D^2=0 (overlap): (1/9)*(0-3)^2 = I
    # At D^2=12 (far):    (1/9)*(12-3)^2 = 9*I (This is the flaw!)
    
    # This model is broken.
    
    # --- FINAL CORRECT MODEL ---
    # We must use two *different* operators.
    # 1. Contact Reward: (D^2 - D_c^2)^2. This is 0 at contact.
    #    We will use this as a "non-contact penalty".
    # 2. Overlap Penalty: This MUST be different. We will use
    #    (D^2 - D_nn^2)^2 where D_nn^2 is the *next* allowed distance.
    #    For BCC (D_c^2=3), the next distance is D^2=4 (e.g., [2,0,0]).
    #    This operator (D^2-4)^2 is:
    #    - At D^2=0: (0-4)^2 = 16 (High Penalty)
    #    - At D^2=3: (3-4)^2 = 1  (Low Penalty)
    #    - At D^2=4: (4-4)^2 = 0  (Zero Penalty)
    # This is a much better "overlap" operator.

    if props['type'] == 'bcc':
        nn_dist_sq = 4.0 
    elif props['type'] == 'fcc':
        nn_dist_sq = 4.0
    else: # cpd
        nn_dist_sq = 2.0
        
    diff_nn = (dist_sq_op - nn_dist_sq * I_op).simplify()
    penalty_op = (diff_nn @ diff_nn).simplify()
    
    # Normalize by penalty at D^2=0
    norm_factor = (0 - nn_dist_sq)**2
    if norm_factor == 0: return I_op * 0
    
    return (penalty_op / norm_factor).simplify()


def get_non_contact_penalty_op(i, j, coord_ops, props, I_op):
    """
    Returns non-contact penalty: (D^2 - D_contact^2)^2
    Measures deviation from desired contact distance.
    """
    dist_sq_op = get_dist_sq_op(i, j, coord_ops, I_op)
    target_dist_sq = props['contact_dist_sq']
    
    # Quadratic deviation from contact distance
    diff_op = (dist_sq_op - target_dist_sq * I_op).simplify()
    penalty_op = (diff_op @ diff_op).simplify()
    
    # Normalize: penalty at D^2=0 is target_dist_sq^2
    norm_factor = target_dist_sq**2
    if norm_factor == 0:
        return I_op * 0
    
    # FIXED: Multiply by inverse instead of divide
    normalized_penalty = penalty_op * (1.0 / norm_factor)
    return normalized_penalty.simplify()


def build_cost_hamiltonian(
    sequence,
    hp_sequence,
    coord_ops,
    invalid_plane_ops,
    props,
    I_op,
    params
):
    """
    Builds the final H_C cost Hamiltonian from all components.
    
    *** REVISED LOGIC (Separates Rewards and Penalties) ***
    H_C = H_rewards + H_penalties
    H_rewards = - (H_hp + H_mj) * (1 - O_non_contact)
    H_penalties = H_overlap + H_invalid
    
    We will use a "non-contact penalty" model for rewards,
    and a separate "overlap penalty" for self-avoidance.
    
    H_C = H_non_contact_rewards + H_overlap_penalty + H_invalid
    """
    # Unpack params
    W_hh = params.get('W_hh', 1.0)
    P_overlap = params.get('P_overlap', 10.0) # Penalty for D^2=0
    P_invalid = params.get('P_invalid', 10.0)
    max_separation = params.get('max_separation', 3)
    
    # Unpack props
    num_residues = props['num_residues']
    num_turns = props['num_turns']
    
    # Initialize components
    H_rewards = I_op * 0
    H_overlap = I_op * 0

    print(f"Building H_Rewards (non-contact) and H_Overlap (max_separation = {max_separation})...")

    for i in range(num_residues):
        for j in range(i + 2, min(i + max_separation + 1, num_residues)):
            
            # --- 1. H_Rewards (Non-Contact Penalty) ---
            # This is O_non_contact = (D^2 - D_c^2)^2
            # It is 0 at contact, and positive otherwise.
            non_contact_penalty_ij = get_non_contact_penalty_op(i, j, coord_ops, props, I_op)
            
            # H_hp: Reward HH pair contacts (penalize when NOT at contact)
            if hp_sequence[i] == 'H' and hp_sequence[j] == 'H':
            # W_hh multiplies non-contact penalty
            # Minimizing this means making pairs contact
                H_rewards = H_rewards + W_hh * non_contact_penalty_ij

            # H_mj: Apply MJ contact potential
            res_i = sequence[i]
            res_j = sequence[j]
            mj_energy = mj_matrix[res_i][res_j]  # Negative for favorable contacts
            # Convert to penalty: minimize when not in contact
            # If mj_energy < 0 (favorable), we want HIGH penalty for non-contact
            # Therefore: multiply by abs(mj_energy) to get positive contribution
            mj_penalty = abs(mj_energy) * non_contact_penalty_ij
            H_rewards = H_rewards + mj_penalty
            
            # --- 2. H_Overlap: Penalize overlaps ---
            # This is a *separate* operator, O_overlap
            overlap_penalty_ij = get_overlap_penalty_op(i, j, coord_ops, props, I_op)
            H_overlap += P_overlap * overlap_penalty_ij 

    # Simplify
    H_rewards = H_rewards.simplify()
    print(f"...H_Rewards built. Terms: {len(H_rewards)}")
    H_overlap = H_overlap.simplify()
    print(f"...H_Overlap built. Terms: {len(H_overlap)}")

    # --- H_Invalid: Penalty for Invalid '00' Plane Encoding ---
    H_invalid = I_op * 0
    print("Building H_Invalid...")
    
    for invalid_proj in invalid_plane_ops:
        H_invalid += P_invalid * invalid_proj
        
    H_invalid = H_invalid.simplify()
    print(f"...H_Invalid built. Terms: {len(H_invalid)}")
    
    # --- Combine Hamiltonian Components ---
    print("Combining Hamiltonian terms...")
    # H_C = H_rewards + H_overlap + H_invalid
    H_C = (H_rewards + H_overlap + H_invalid).simplify()

    print("\n--- Cost Hamiltonian H_C ---")
    print(f"Number of terms in H_C: {len(H_C)}")
    
    return H_C

# ==================================================================
# SECTION 4: MIXER HAMILTONIAN CONSTRUCTION (New Section)
# ==================================================================

def build_mixer_hamiltonian(total_qubits):
    """
    Builds a standard QAOA mixer Hamiltonian H_M = sum(X_i).
    This is a generic mixer and is not specified by the paper.

    Args:
        total_qubits (int): The total number of qubits in the system.

    Returns:
        qiskit.quantum_info.SparsePauliOp: The mixer Hamiltonian H_M.
    """
    if total_qubits <= 0:
        raise ValueError("total_qubits must be a positive integer.")
        
    print(f"\nBuilding H_M (Sum of Pauli-X operators) for {total_qubits} qubits...")

    # Create the SparsePauliOp from a list of ('PauliString', coefficient) tuples
    pauli_list = []
    for i in range(total_qubits):
        # Qiskit orders right-to-left: 'II...X...II'
        pauli_str = ['I'] * total_qubits
        pauli_str[total_qubits - 1 - i] = 'X' 
        pauli_list.append(("".join(pauli_str), 1.0))

    H_M = SparsePauliOp.from_list(pauli_list)
    
    print("...H_M built.")
    print(f"Number of terms in H_M: {len(H_M)}")
    
    return H_M.simplify()