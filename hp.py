"""
Hydrophobic-Polar (HP) encoding for amino acids.
Extended to include all 20 standard amino acids with biochemical accuracy.

Classifications based on Kyte-Doolittle hydrophobicity scale:
- H (Hydrophobic): Nonpolar, tend to be buried in protein core
- P (Polar): Charged or polar, tend to be on surface
- I (Intermediate): Mixed character (can be either, depends on context)

Standard mapping for QAOA:
Map intermediate to P (favor surface) for conservative folding.
"""

hp_mapping = {
    # HYDROPHOBIC (Nonpolar) - 9 residues
    "A": "H",  # Alanine - small, nonpolar
    "V": "H",  # Valine - branched, hydrophobic
    "I": "H",  # Isoleucine - branched, hydrophobic
    "L": "H",  # Leucine - branched, hydrophobic
    "M": "H",  # Methionine - nonpolar, contains S
    "F": "H",  # Phenylalanine - aromatic, hydrophobic
    "W": "H",  # Tryptophan - aromatic, hydrophobic
    "P": "H",  # Proline - imino acid, hydrophobic
    "C": "H",  # Cysteine - can form disulfides, mostly hydrophobic
    
    # POLAR/CHARGED - 11 residues
    "R": "P",  # Arginine - positively charged, polar
    "N": "P",  # Asparagine - polar, uncharged
    "D": "P",  # Aspartic acid - negatively charged
    "Q": "P",  # Glutamine - polar, uncharged
    "E": "P",  # Glutamic acid - negatively charged
    "H": "P",  # Histidine - can be charged/polar
    "K": "P",  # Lysine - positively charged
    "S": "P",  # Serine - polar, uncharged
    "T": "P",  # Threonine - polar, uncharged
    "Y": "P",  # Tyrosine - aromatic but polar (OH group)
    "G": "P",  # Glycine - flexible, no sidechain
}

def encode_hp_sequence(sequence):
    """
    Encodes protein sequence to HP model.
    
    Args:
        sequence: String or list of single-letter amino acid codes
    
    Returns:
        tuple: (hp_sequence, num_residues)
        
    Raises:
        KeyError: If unknown amino acid encountered
    """
    if not sequence:
        raise ValueError("Sequence cannot be empty")
    
    hp_sequence = []
    for aa in sequence:
        if aa not in hp_mapping:
            raise KeyError(f"Unknown amino acid: {aa}")
        hp_sequence.append(hp_mapping[aa])
    
    return hp_sequence, len(sequence)


def get_hydrophobic_pairs(hp_sequence):
    """
    Returns list of (i, j) indices for HH pairs.
    Useful for validation and energy computation.
    """
    pairs = []
    for i in range(len(hp_sequence)):
        for j in range(i + 1, len(hp_sequence)):
            if hp_sequence[i] == 'H' and hp_sequence[j] == 'H':
                pairs.append((i, j))
    return pairs