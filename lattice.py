# Latiice Geometires and their Neighbor Vectors

# ---------------------------
# BCC (Body Centered Cubic) lattice neighbor vectors (8 neighbors)
# These vectors point along the "body diagonals"
# These vectors have norm sqrt(3) ≈ 1.732
# ---------------------------
bcc_neighbors = [
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
]

# ---------------------------
# FCC (Face Centered Cubic) lattice neighbor vectors (12 neighbors)
# These vectors point along the "plane diagonals"
# (This is the lattice from your original file and Fig 1(f) in arXiv:2406.01547v1 )
# These vectors have norm sqrt(2) ≈ 1.414
# ---------------------------
fcc_neighbors = [
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
]

# ---------------------------
# Cubic with Planar Diagonals (18 neighbors)
# As defined in paper arXiv:2406.01547v1 (Fig 1(c)  and Fig 3 [cite: 89])
# The paper defines this as having 18 degrees of freedom[cite: 71],
# combining Simple Cubic (norm 1) and FCC (norm sqrt(2)) neighbors.
# ---------------------------
cpd_neighbors = [
    # 6 Simple Cubic neighbors (norm 1)
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
    # 12 FCC neighbors (norm sqrt(2))
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
]

# ---------------------------
# SC (Simple Cubic) lattice neighbor vectors (6 neighbors)
# These vectors point along the axes (norm 1)
# ---------------------------
sc_neighbors = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]