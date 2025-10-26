#!/usr/bin/env python
# coding: utf-8

"""
results_interpreter.py

Interprets and validates the results from a QAOA run.

This file provides generalized functions to:
1.  Find the most probable *valid* bitstring from the results.
2.  Decode the bitstring into a 3D structure.
3.  Validate the structure for self-avoidance (overlaps).
4.  Compute the classical energy of the predicted structure.
5.  Plot the optimization history and final probability distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Import from our other project files
from qubit_encoding import get_neighbor_paper_map
from reconstruct import reconstruct_from_turns

def _permute_qiskit_bits(qiskit_turn_bits, num_dir_q, num_plane_q):
    """
    Converts Qiskit bitstring to paper encoding format.
    
    Qiskit order: [MSB ... LSB] = [q_n-1 ... q_0]
    Paper order: [plane_bits][direction_bits]
    
    For BCC: qiskit "101" -> direction "101" (direct, no plane)
    For FCC: qiskit "11010" -> plane "11" + direction "010"
    """
    # Qiskit string is right-to-left
    q_bits_reversed = qiskit_turn_bits[::-1]
    
    # Extract direction and plane bits in order
    direction_bits = q_bits_reversed[:num_dir_q]
    plane_bits = q_bits_reversed[num_dir_q:num_dir_q + num_plane_q]
    
    # Reconstruct in paper order: [plane][direction]
    encoding_string = plane_bits + direction_bits
    
    return encoding_string


def decode_bitstring_to_coords(bitstring, neighbors, props, verbose=True):
    """
    Decodes a Qiskit bitstring into a list of 3D coordinates.

    Args:
        bitstring (str): The Qiskit bitstring (MSB to LSB).
        neighbors (list): The list of neighbor vectors (e.g., fcc_neighbors).
        props (dict): The lattice properties dictionary from hamiltonian.py.
        verbose (bool): If True, print detailed decoding steps.

    Returns:
        tuple: (coords, turn_indices, is_valid)
            - coords (list): List of (x,y,z) tuples.
            - turn_indices (list): List of integer turn indices.
            - is_valid (bool): True if decoding was successful.
    """
    
    qubits_per_turn = props['qubits_per_turn']
    num_turns = props['num_turns']
    num_dir_q = props['num_direction_qubits']
    num_plane_q = props['num_plane_qubits']
    model_type = props['model_type']
    
    # 1. Build the bitstring-to-vector map
    try:
        neighbor_map = get_neighbor_paper_map(neighbors)
        bitstring_to_vector = {v: k for k, v in neighbor_map.items()}
        vector_to_index = {vec: i for i, vec in enumerate(neighbors)}
    except (ValueError, AssertionError) as e:
        if verbose:
            print(f"❌ ERROR: Failed to build neighbor map: {e}")
        return None, [], False

    # 2. Chunk the Qiskit bitstring
    # Qiskit string is [Turn_N-1_bits]...[Turn_1_bits][Turn_0_bits]
    chunks = [bitstring[i:i + qubits_per_turn] 
              for i in range(0, len(bitstring), qubits_per_turn)]
    
    # Reverse to get turns in order [Turn_0, Turn_1, ...]
    qiskit_turn_bitstrings = chunks[::-1]
    
    if len(qiskit_turn_bitstrings) != num_turns:
        if verbose:
            print(f"❌ ERROR: Bitstring length mismatch. Expected {num_turns} turns.")
        return None, [], False

    # 3. Decode each turn
    turn_vectors = []
    turn_indices = []
    is_valid = True

    if verbose:
        print("\n" + "-"*60)
        print("Turn Decoding (Qiskit bits -> Paper bits -> Vector):")
        print("-"*60)
    
    for i, q_bits in enumerate(qiskit_turn_bitstrings):
        
        # Permute Qiskit bits to match the paper's encoding string
        encoding_string = _permute_qiskit_bits(q_bits, num_dir_q, num_plane_q)

        vector = bitstring_to_vector.get(encoding_string)
        
        if vector:
            turn_vectors.append(vector)
            turn_indices.append(vector_to_index[vector])
            if verbose:
                plane_info = ""
                if model_type == 'orthogonal':
                    plane_bits = encoding_string[:2]
                    plane = {"01": "y-z", "10": "z-x", "11": "x-y", "00": "INVALID"}.get(plane_bits)
                    plane_info = f"(plane: {plane})"
                print(f"Turn {i}: {q_bits} → '{encoding_string}' → {str(vector):>15} {plane_info}")
        else:
            turn_vectors.append(None)
            turn_indices.append(None)
            is_valid = False
            if verbose:
                print(f"Turn {i}: {q_bits} → '{encoding_string}' → ❌ INVALID ENCODING")
            if not verbose:
                # Fail fast if in silent mode
                break
            
    # 4. Reconstruct coordinates
    if not is_valid:
        return None, [], False
        
    try:
        coords = reconstruct_from_turns(turn_indices, neighbors)
        return coords, turn_indices, True
    except IndexError as e:
        if verbose:
            print(f"❌ ERROR: Failed to reconstruct coords: {e}")
        return None, [], False

def compute_classical_energy(coords, sequence, hp_sequence, mj_matrix, props, params):
    """
    Computes classical energy for a given structure.
    
    Matches the QAOA cost Hamiltonian exactly:
    E = sum_i,j [ W_hh * (HH non-contact) + MJ * (non-contact) 
                 + P_overlap * (overlap penalty) ]
    """
    energy = 0.0
    num_contacts = 0
    favorable_contacts = 0
    
    W_hh = params.get('W_hh', 1.0)
    P_overlap = params.get('P_overlap', 10.0)
    max_separation = params.get('max_separation', num_residues - 1)
    
    num_residues = len(coords)
    contact_dist_sq = props['contact_dist_sq']
    
    # Determine next allowed distance (for overlap penalty)
    if props['type'] == 'bcc':
        # BCC allowed: 3 (contact), 4, 5, 8, 11, 12, 19, ...
        next_dist_sq = 4.0
    elif props['type'] == 'fcc':
        # FCC allowed: 2 (contact), 4, 6, 8, 10, ...
        next_dist_sq = 4.0
    else:  # cpd
        # CPD allowed: 1 (contact), 2, 4, 5, ...
        next_dist_sq = 2.0
    
    # Normalization factors
    non_contact_norm = contact_dist_sq ** 2
    if non_contact_norm == 0:
        non_contact_norm = 1.0
    
    overlap_norm = next_dist_sq ** 2
    if overlap_norm == 0:
        overlap_norm = 1.0
    
    # MAIN LOOP: All residue pairs within separation limit
    for i in range(num_residues):
        for j in range(i + 2, min(i + max_separation + 1, num_residues)):
            
            # Calculate squared distance
            dist_sq = sum((coords[i][k] - coords[j][k])**2 for k in range(3))
            
            # ===== NON-CONTACT PENALTY (Reward Signal) =====
            non_contact_penalty = ((dist_sq - contact_dist_sq) ** 2) / non_contact_norm
            
            # HH pair reward
            if hp_sequence[i] == 'H' and hp_sequence[j] == 'H':
                energy += W_hh * non_contact_penalty
            
            # MJ potential reward
            res_i = sequence[i]
            res_j = sequence[j]
            mj_energy = mj_matrix[res_i][res_j]
            
            # Favorable interactions (mj_energy < 0) are rewarded
            # Energy term: -mj_energy * non_contact_penalty
            # This gives POSITIVE contribution (penalty for non-contact)
            energy += (-mj_energy) * non_contact_penalty
            
            # ===== OVERLAP PENALTY (Repulsion) =====
            overlap_penalty = ((next_dist_sq - dist_sq) ** 2) / overlap_norm
            energy += P_overlap * overlap_penalty
            
            # ===== TRACKING =====
            if abs(dist_sq - contact_dist_sq) < 0.5:
                num_contacts += 1
            
            if abs(dist_sq - contact_dist_sq) < 0.5 and mj_energy < 0:
                favorable_contacts += 1
    
    return {
        'total_energy': energy,
        'num_contacts': num_contacts,
        'favorable_contacts': favorable_contacts,
        'num_residues': num_residues
    }

def plot_results(energy_history, final_counts, best_qaoa_energy):
    """Generates plots for energy convergence and probability distribution."""
    
    print("\n" + "="*60)
    print("  VISUALIZATION")
    print("="*60)

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Energy convergence
        ax1.plot(energy_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=best_qaoa_energy, color='r', linestyle='--', label=f'Best: {best_qaoa_energy:.3f}')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Energy', fontsize=12)
        ax1.set_title('QAOA Optimization Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Probability distribution
        total_shots = sum(final_counts.values())
        top_10 = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        def truncate_bstr(b):
            if len(b) > 16:
                return b[:8] + '...' + b[-8:]
            return b
            
        bitstrings = [truncate_bstr(b) for b, _ in top_10]
        probs = [c / total_shots for _, c in top_10]

        ax2.bar(range(len(probs)), probs, color='steelblue', edgecolor='black')
        ax2.set_xlabel('State (top 10)', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Measurement Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(bitstrings)))
        ax2.set_xticklabels(bitstrings, rotation=60, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\n⚠️ Matplotlib not found. Skipping plots.")
    except Exception as e:
        print(f"\n⚠️ An error occurred during plotting: {e}")

def analyze_energy_landscape(energy_history):
    """
    Analyzes optimization trajectory to verify convergence.
    
    Args:
        energy_history (list): List of energy values from each optimizer iteration
        
    Returns:
        dict: Statistics about convergence
        
    Prints:
        - Initial vs final energy
        - Total improvement percentage
        - Convergence status
        - Oscillation analysis
    """
    
    print("\n" + "="*70)
    print("ENERGY CONVERGENCE ANALYSIS")
    print("="*70)
    
    if not energy_history:
        print("❌ ERROR: Empty energy history!")
        return None
    
    history = np.array(energy_history)
    
    # ===== BASIC STATISTICS =====
    print("\n📊 OPTIMIZATION TRAJECTORY:")
    print(f"   Initial energy:     {history[0]:12.6f}")
    print(f"   Final energy:       {history[-1]:12.6f}")
    improvement = history[0] - history[-1]
    print(f"   Total improvement:  {improvement:12.6f}")
    
    if abs(history[0]) > 1e-10:
        pct_improve = 100 * improvement / abs(history[0])
    else:
        pct_improve = 0
    print(f"   Improvement %:      {pct_improve:12.1f}%")
    
    # ===== CONVERGENCE CHECK =====
    print("\n🎯 CONVERGENCE STATUS:")
    last_10 = history[-10:] if len(history) >= 10 else history
    variance = np.var(last_10)
    std_dev = np.std(last_10)
    
    print(f"   Last 10 iterations variance: {variance:.8f}")
    print(f"   Last 10 iterations std dev:  {std_dev:.8f}")
    
    if variance < 1e-6:
        status = "✅ EXCELLENT: Converged smoothly"
        convergence_score = 100
    elif variance < 1e-3:
        status = "🟡 GOOD: Converged with minor fluctuations"
        convergence_score = 75
    elif variance < 0.01:
        status = "🟠 MARGINAL: Weak convergence, noisy"
        convergence_score = 50
    else:
        status = "❌ POOR: No convergence, highly unstable"
        convergence_score = 0
    
    print(f"   Status: {status}")
    print(f"   Convergence Score: {convergence_score}/100")
    
    # ===== OSCILLATION ANALYSIS =====
    print("\n📈 IMPROVEMENT ANALYSIS:")
    diffs = np.diff(history)
    improving = np.sum(diffs < 0)  # Negative = energy decreased
    worsening = np.sum(diffs > 0)  # Positive = energy increased
    total = len(diffs)
    
    print(f"   Improving steps:    {improving:4d}/{total} ({100*improving/total:5.1f}%)")
    print(f"   Worsening steps:    {worsening:4d}/{total} ({100*worsening/total:5.1f}%)")
    
    # ===== STAGNATION DETECTION =====
    print("\n⏱️  STAGNATION DETECTION:")
    
    # Check first half vs second half
    mid = len(history) // 2
    first_half_improvement = history[0] - history[mid]
    second_half_improvement = history[mid] - history[-1]
    
    print(f"   First half improvement:  {first_half_improvement:.6f}")
    print(f"   Second half improvement: {second_half_improvement:.6f}")
    
    if second_half_improvement < 0.1 * first_half_improvement:
        print(f"   ⚠️  WARNING: Optimizer stalled in second half")
        stagnation_detected = True
    else:
        print(f"   ✅ Optimizer making progress throughout")
        stagnation_detected = False
    
    # ===== LOCAL MINIMUM DETECTION =====
    print("\n🔍 LOCAL MINIMUM DETECTION:")
    if improving < 0.5 * total:
        print(f"   ⚠️  SUSPICIOUS: Less than 50% of steps improved")
        print(f"      Likely stuck in local minimum")
        local_minimum = True
    else:
        print(f"   ✅ Good balance of improvements")
        local_minimum = False
    
    # ===== TRAJECTORY SMOOTHNESS =====
    print("\n📉 TRAJECTORY SMOOTHNESS:")
    if len(history) > 1:
        smoothness = np.mean(np.abs(np.diff(diffs)))
        print(f"   Average step variance: {smoothness:.8f}")
        
        if smoothness < 1e-3:
            print(f"   ✅ Smooth trajectory (ideal)")
        elif smoothness < 0.1:
            print(f"   🟡 Slightly noisy but manageable")
        else:
            print(f"   ❌ Very noisy - optimizer struggling")
    
    # ===== RECOMMENDATIONS =====
    print("\n💡 RECOMMENDATIONS:")
    
    if convergence_score >= 75:
        print("   ✅ Excellent optimization run!")
        print("      Result is trustworthy for interpretation")
        recommendation = "ACCEPT"
        
    elif convergence_score >= 50:
        print("   🟡 Acceptable optimization")
        print("      Try running again with more iterations")
        print("      or different initial parameters")
        recommendation = "RETRY"
        
    elif local_minimum:
        print("   ❌ Optimizer stuck in local minimum")
        print("      Try: 1) More QAOA layers (increase reps)")
        print("           2) Different initial parameters")
        print("           3) Reduce overlap penalty P_overlap")
        recommendation = "RESTART"
        
    elif stagnation_detected:
        print("   ⚠️  Optimizer stalled")
        print("      Increase maxiter parameter")
        recommendation = "INCREASE_ITERATIONS"
    else:
        print("   ❓ Mixed results")
        recommendation = "INVESTIGATE"
    
    # ===== RETURN DICTIONARY =====
    stats = {
        'initial_energy': float(history[0]),
        'final_energy': float(history[-1]),
        'total_improvement': float(improvement),
        'improvement_percent': float(pct_improve),
        'convergence_variance': float(variance),
        'convergence_score': convergence_score,
        'improving_steps': int(improving),
        'total_steps': int(total),
        'stagnation_detected': bool(stagnation_detected),
        'local_minimum_detected': bool(local_minimum),
        'recommendation': recommendation
    }
    
    print("\n" + "="*70)
    return stats

def interpret_qaoa_results(
    result, final_counts, energy_history, optimization_time,
    sequence, hp_sequence, neighbors, mj_matrix, props, params
):
    """
    Main function to decode, validate, and report QAOA results.
    Finds the most probable *valid* structure.
    """
    
    print("\n" + "="*60)
    print("  STRUCTURE PREDICTION RESULTS")
    print("="*60)
    
    total_shots = sum(final_counts.values())
    best_qaoa_energy = result.fun

    # --- UPDATED LOGIC: Find best *valid* bitstring ---
    
    # 1. Sort all counts from most to least probable
    sorted_counts = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)

    # 2. Print raw top 5 outcomes for user inspection
    print("Top 5 Raw Measurement Outcomes (may be invalid):")
    print("-"*60)
    for i, (bitstring, count) in enumerate(sorted_counts[:5], 1):
        prob = count / total_shots
        print(f"{i}. {bitstring}: {count:4d} shots ({prob:6.2%})")

    # 3. Iterate through sorted list to find the first valid one
    best_valid_bitstring = None
    best_valid_coords = None
    best_valid_probability = 0
    found_at_rank = -1

    print("\nSearching for the most probable VALID structure...")
    for rank, (bitstring, count) in enumerate(sorted_counts):
        
        # Try to decode in "silent" mode
        coords, turns, is_valid = decode_bitstring_to_coords(
            bitstring, neighbors, props, verbose=False
        )
        
        if is_valid:
            best_valid_bitstring = bitstring
            best_valid_coords = coords
            best_valid_probability = count / total_shots
            found_at_rank = rank
            break # Found it!
            
    # 4. Check if we found any valid structure at all
    if best_valid_bitstring is None:
        print("❌ FATAL: No valid structures were found in any measurement results.")
        print("This may indicate a problem with the Hamiltonian or optimization.")
        # Plot results anyway to help debug
        plot_results(energy_history, final_counts, best_qaoa_energy)
        return

    # 5. --- We found a valid structure! ---
    # Now, print all details about *this* structure.
    print(f"✓ Found valid structure at rank {found_at_rank}!")
    print(f"  Bitstring: {best_valid_bitstring}")
    print(f"  Probability: {best_valid_probability:.4f} ({best_valid_probability*100:.1f}%)")
    
    # Re-run decoder one time in verbose mode to show the details
    decode_bitstring_to_coords(best_valid_bitstring, neighbors, props, verbose=True)

    # 6. Reconstruct 3D structure and validate
    print("\n" + "-"*60)
    print("Predicted 3D Structure:")
    print("-"*60)
    for i, coord in enumerate(best_valid_coords):
        print(f" Residue {i} ({sequence[i]:>3}): {coord}")
    
    # Validation checks
    print("\n" + "-"*60)
    print("Validation:")
    print("-"*60)
    
    # Check for self-intersection
    if len(set(best_valid_coords)) != len(best_valid_coords):
        print("❌ WARNING: Structure contains overlaps!")
        duplicates = [coord for coord, count in Counter(best_valid_coords).items() if count > 1]
        print(f"   Overlapping positions: {duplicates}")
    else:
        print("✓ Structure is self-avoiding (no overlaps)")
    
    '''
    # Compute classical energy
    classical_energy, num_contacts = compute_classical_energy(
        best_valid_coords, sequence, hp_sequence, mj_matrix, props, params
    )
    print(f"✓ Classical energy: {classical_energy:.4f}")
    print(f"✓ QAOA estimated energy (from optimizer): {best_qaoa_energy:.4f}")
    print(f"✓ Number of contacts: {num_contacts}")
    '''
        
    # 7. Plot results
    plot_results(energy_history, final_counts, best_qaoa_energy)
    
    print("\n✅ QAOA ANALYSIS COMPLETE!")
    print(f"Total optimization runtime: {optimization_time:.1f} seconds")

    return best_valid_coords

def visualize_protein_fold(coords, sequence, hp_sequence, lattice_name, 
                          neighbors, title=None, show_lattice=True):
    """
    Visualizes the final folded protein structure on 3D lattice.
    
    Shows:
    - Final protein fold (backbone chain)
    - Lattice points (semi-transparent)
    - Lattice structure (edges connecting lattice points)
    - Color-coded residues (blue=hydrophobic, red=polar)
    
    Args:
        coords (list): List of (x, y, z) tuples for each residue
        sequence (list): Protein sequence (e.g., ['A', 'S', 'V', 'K', 'F'])
        hp_sequence (list): HP model sequence (e.g., ['H', 'P', 'H', 'P', 'H'])
        lattice_name (str): Name of lattice ('FCC', 'BCC', 'CPD')
        neighbors (list): Neighbor vectors for the lattice
        title (str, optional): Custom title. Default: "{Lattice} Protein Folding"
        show_lattice (bool): Whether to show lattice structure (default: True)
    
    Returns:
        fig: Plotly figure object (can be saved with fig.write_html())
    
    Raises:
        ValueError: If coords or sequence empty or mismatched length
        ImportError: If plotly not available
    
    Example:
        >>> coords = [(0,0,0), (1,1,0), (0,0,1), (-1,1,1)]
        >>> sequence = ['A', 'V', 'K', 'S']
        >>> hp_seq = ['H', 'H', 'P', 'P']
        >>> fig = visualize_protein_fold(
        ...     coords, sequence, hp_seq, 'FCC', fcc_neighbors
        ... )
        >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly not installed. Install with: pip install plotly")
    
    # ===== INPUT VALIDATION =====
    if not coords or not sequence:
        raise ValueError("coords and sequence cannot be empty")
    
    if len(coords) != len(sequence):
        raise ValueError(
            f"Length mismatch: {len(coords)} coords vs {len(sequence)} residues"
        )
    
    if len(coords) < 2:
        raise ValueError("Need at least 2 residues for visualization")
    
    # ===== SETUP =====
    fig = go.Figure()
    
    # Set title
    if title is None:
        title = f"Protein Folding on {lattice_name} Lattice"
    
    # Color mapping: Hydrophobic (blue), Polar (red)
    colors = []
    for hp in hp_sequence:
        if hp == 'H':
            colors.append('blue')
        else:  # 'P'
            colors.append('red')
    
    # ===== PLOT PROTEIN BACKBONE =====
    x_coords, y_coords, z_coords = zip(*coords)
    
    # Add backbone trace
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.9,
            line=dict(color='black', width=1)
        ),
        line=dict(
            color='darkgray',
            width=4
        ),
        text=[f"Res {i}: {sequence[i]} ({hp_sequence[i]})" 
              for i in range(len(sequence))],
        hovertemplate='<b>%{text}</b><br>Pos: (%{x}, %{y}, %{z})<extra></extra>',
        name='Protein Backbone'
    ))
    
    # ===== LATTICE VISUALIZATION =====
    if show_lattice:
        
        # Convert coords to set for fast lookup
        protein_coords = set(coords)
        
        # Find bounding box
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        z_vals = [c[2] for c in coords]
        
        padding = 2
        min_x, max_x = min(x_vals) - padding, max(x_vals) + padding
        min_y, max_y = min(y_vals) - padding, max(y_vals) + padding
        min_z, max_z = min(z_vals) - padding, max(z_vals) + padding
        
        # ===== LATTICE POINTS =====
        lattice_points = []
        
        if lattice_name.upper() == 'FCC':
            # FCC condition: sum of coordinates must be even
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    for z in range(min_z, max_z + 1):
                        if (x + y + z) % 2 == 0:
                            if (x, y, z) not in protein_coords:
                                lattice_points.append((x, y, z))
        
        elif lattice_name.upper() == 'BCC':
            # BCC: all integer points (no constraint)
            # Exclude protein coordinates
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    for z in range(min_z, max_z + 1):
                        if (x, y, z) not in protein_coords:
                            lattice_points.append((x, y, z))
        
        elif lattice_name.upper() == 'CPD':
            # CPD: all integer points (union of SC and FCC)
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    for z in range(min_z, max_z + 1):
                        if (x, y, z) not in protein_coords:
                            lattice_points.append((x, y, z))
        
        else:
            raise ValueError(f"Unknown lattice: {lattice_name}")
        
        # Add lattice points
        if lattice_points:
            lx, ly, lz = zip(*lattice_points)
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode='markers',
                marker=dict(
                    size=2,
                    color='darkgray',
                    opacity=0.3
                ),
                name='Lattice Points',
                hoverinfo='skip'
            ))
            
            # ===== LATTICE STRUCTURE (EDGES) =====
            lattice_set = set(lattice_points)
            line_x, line_y, line_z = [], [], []
            
            for point in lattice_points:
                for vec in neighbors:
                    neighbor = tuple(point[i] + vec[i] for i in range(3))
                    
                    # Draw line only if neighbor exists and coordinate is "greater"
                    # (to avoid drawing same edge twice)
                    if neighbor in lattice_set and neighbor > point:
                        line_x.extend([point[0], neighbor[0], None])
                        line_y.extend([point[1], neighbor[1], None])
                        line_z.extend([point[2], neighbor[2], None])
            
            if line_x:
                fig.add_trace(go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    line=dict(
                        color='lightgray',
                        width=0.5
                    ),
                    opacity=0.15,
                    name='Lattice Structure',
                    hoverinfo='skip'
                ))
    
    # ===== LAYOUT =====
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=900,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    # Add legend annotations
    fig.add_annotation(
        text="<b>Legend:</b><br>🔵 Blue = Hydrophobic (H)<br>🔴 Red = Polar (P)",
        xref="paper", yref="paper",
        x=0.02, y=0.15,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font=dict(size=10),
        align="left"
    )
    
    return fig