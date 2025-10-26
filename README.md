# QAOA Protein Folding Simulation

## Project Overview

This project implements a Quantum Approximate Optimization Algorithm (QAOA) pipeline using Qiskit to predict the three-dimensional structure of a protein sequence on various lattice models (BCC, FCC, CPD). The methodology is primarily based on the qubit-efficient encoding scheme described in **arXiv:2406.01547v1**.

The goal is to find the lowest energy conformation of the protein, considering hydrophobic interactions (HP model) and Miyazawa-Jernigan (MJ) contact potentials, while penalizing invalid structures (overlaps and incorrect encodings).

---

## Meeting Submission Criteria

This repository addresses the submission requirements as follows:

1.  **Qiskit Code:**
    * The complete, well-commented Qiskit-based pipeline is implemented across several Python modules (`.py` files) and driven by a Jupyter Notebook (`quantum_protein_folding.ipynb`).
    * A separate notebook (`choice.ipynb`) provides analysis justifying the design choices.
    * See the **Code Structure** section below for details on each file.

2.  **Lattice Design Documentation:**
    * **Choice:** The code supports Body-Centered Cubic (BCC), Face-Centered Cubic (FCC), and Cubic with Planar Diagonals (CPD) lattices, defined in `lattice.py`. BCC is used in the example notebook, justified by the analysis in `choice.ipynb`.
    * **Advantages (Qubit Efficiency):** The encoding follows arXiv:2406.01547v1, using a turn-based representation analyzed in `choice.ipynb`.
        * For FCC and CPD ("orthogonal" model), it uses 2 qubits for plane selection and log2(directions_per_plane) qubits for direction within the plane (`qubit_encoding.py`).
        * For BCC ("direct" model), it encodes the 8 neighbor directions directly using ceil(log2(8))=3 qubits per turn, eliminating plane selection qubits (`qubit_encoding.py`).
        * This turn-based encoding scales linearly with sequence length (N-1 turns) and logarithmically with the number of neighbors/directions, offering significant qubit savings compared to coordinate-based encodings for longer sequences.
    * Helper functions for generating randomized, overlap-safe coordinate or turn encodings for classical comparison are in `lattice_encoding.py`.

3.  **Hamiltonian Design:**
    * The Cost Hamiltonian ($H_C$) is constructed algorithmically in `hamiltonian.py` based on the paper's formulation.
        * **Coordinate Operators:** It builds Pauli operators representing residue coordinates (X_i, Y_i, Z_i) based on the chosen lattice and qubit encoding scheme.
        * **Energy Terms:**
            * **HP Interaction:** Rewards contacts between Hydrophobic ('H') residues (defined in `hp.py`) using a non-contact penalty formulation `(D^2 - D_c^2)^2`.
            * **MJ Potential:** Incorporates Miyazawa-Jernigan contact energies (`mj.py`) for all residue pairs, also using the non-contact penalty approach. Favorable contacts (negative MJ energy) are rewarded.
            * **Overlap Penalty:** Uses a *separate* penalty term `(D^2 - D_nn^2)^2` (where D_nn is the next-nearest neighbor distance) to strongly penalize configurations where residues occupy the same or too-close lattice sites.
            * **Invalid State Penalty:** Penalizes the '00' qubit state for plane selection in orthogonal models, which doesn't correspond to a valid plane.
    * The Mixer Hamiltonian ($H_M$) is the standard sum of single-qubit Pauli-X operators, also built in `hamiltonian.py`.
    * The complexity comparison between lattices (in terms of Hamiltonian terms) is analyzed in `choice.ipynb`.

4.  **Energy Landscape Analysis:**
    * The QAOA optimization progress (energy vs. iteration) is tracked during the optimization (`folding_qaoa.py`).
    * The `results_interpreter.py` script includes:
        * `plot_results`: Visualizes the energy convergence history and the probability distribution of the final measurement outcomes.
        * `analyze_energy_landscape`: Provides a textual analysis of the optimization trajectory, assessing convergence quality, stagnation, and potential local minima.

5.  **Performance Report:**
    * **Predicted vs. Expected:** The primary output is the most probable *valid* bitstring found after QAOA optimization. This bitstring is decoded into 3D coordinates (`results_interpreter.py`). While a specific "expected" structure isn't predefined for arbitrary sequences, the code validates the predicted structure for self-avoidance (no overlaps). The classical energy of the predicted structure can be computed for comparison (though the function `compute_classical_energy` is commented out in the notebook's final interpretation step, it is available in `results_interpreter.py`).
    * **Quantum Resources:** The number of qubits required is determined by the sequence length and lattice type, calculated in `qubit_encoding.py` using the paper's efficient method. The analysis in `choice.ipynb` demonstrates BCC's qubit efficiency (12 qubits for 5 residues) compared to other lattices (16-20 qubits) and coordinate encoding (>70 qubits). The circuit depth depends on the number of QAOA repetitions (`qaoa_reps`).

6.  **3D Visualization:**
    * `results_interpreter.py` contains the `visualize_protein_fold` function, which uses Plotly to generate an interactive 3D plot of the predicted protein backbone on the chosen lattice structure. Hydrophobic/Polar residues are color-coded. An example plot is generated at the end of the `quantum_protein_folding.ipynb` notebook.

---

## Code Structure

* `quantum_protein_folding.ipynb`: 🧪 Main Jupyter Notebook to configure and run the QAOA experiment.
* `choice.ipynb`: 📊 Analysis notebook comparing encoding schemes and lattice choices, justifying the selection of Turn-Based Encoding and the BCC lattice based on qubit efficiency and Hamiltonian complexity.
* `lattice.py`: Defines neighbor vectors for BCC, FCC, CPD, and SC lattices.
* `hp.py`: Defines the Hydrophobic-Polar mapping for amino acids.
* `mj.py`: Contains the Miyazawa-Jernigan contact potential matrix.
* `qubit_encoding.py`: Implements the qubit calculation and mapping based on arXiv:2406.01547v1.
* `hamiltonian.py`: Builds the coordinate operators, cost Hamiltonian ($H_C$), and mixer Hamiltonian ($H_M$).
* `folding_qaoa.py`: Contains the logic for building the QAOA circuit and running the optimization using Qiskit Aer's statevector simulator and SciPy's minimizer.
* `results_interpreter.py`: Decodes QAOA results, validates structures, analyzes energy convergence, computes classical energy, and generates plots (energy history, probability distribution, 3D fold).
* `lattice_encoding.py`: (Helper) Functions to generate classical coordinate/turn-based encodings.
* `reconstruct.py`: (Helper) Function to rebuild coordinates from a turn-based encoding.
* `images/`: Directory containing plots generated by the notebooks.

---

## How to Run

1.  **Prerequisites:** Ensure you have Python 3 and pip installed. Install the required libraries:
    ```bash
    pip install qiskit qiskit-aer numpy scipy matplotlib plotly pandas seaborn
    ```
    *(Note: `pandas` and `seaborn` are used by `choice.ipynb`)*.

2.  **Run QAOA Experiment:**
    * Open the `quantum_protein_folding.ipynb` notebook.
    * Modify parameters in the **"EXPERIMENT CONFIGURATION"** cell (sequence, lattice, QAOA reps, etc.).
    * Run the cells sequentially. This will perform the folding simulation and generate results, including plots in the `images/` directory.

3.  **Run Design Choice Analysis:**
    * Open the `choice.ipynb` notebook.
    * Run the cells sequentially. This notebook generates comparative plots (encoding efficiency, qubit cost per lattice, state space size, Hamiltonian terms) saved in the `images/` directory, along with a textual justification for choosing the Turn-Based Encoding on a BCC lattice.

---

## References

* **IBM Quantum QAOA Tutorial:** [https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/quantum-approximate-optimization-algorithm)
* **Core Methodology Paper (Qubit Encoding & Hamiltonian):** [https://arxiv.org/html/2406.01547v1](https://arxiv.org/html/2406.01547v1)