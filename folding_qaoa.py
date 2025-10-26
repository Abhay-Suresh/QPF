"""
qaoa_optimizer.py

Runs the QAOA optimization using a statevector simulator.

This file provides a generalized function to:
1.  Build the parameterized QAOA ansatz.
2.  Define a correct cost function based on statevector expectation
    (H_C is non-diagonal, so we cannot use counts).
3.  Run the optimization using scipy.minimize.
4.  Run a final high-shot measurement with the optimal parameters.
"""

import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter

class QAOACallback:
    """A helper class to store optimization history and provide a callback."""
    def __init__(self, H_C, backend, ansatz, pass_manager):
        self._H_C = H_C
        self._backend = backend
        self._ansatz = ansatz
        self._pass_manager = pass_manager
        self.iteration = 0
        self.energy_history = []
        self.best_energy = float('inf')
        self._start_time = time.time()
        
        # --- CRITICAL FIX: Use Statevector Expectation ---
        # H_C is non-diagonal (has X, Y terms). We cannot compute
        # expectation from Z-basis-measurement counts.
        # We must use the statevector.
        print("Using Statevector-based expectation (H_C is non-diagonal).")

    def cost_function(self, params):
        """
        Cost function for QAOA optimization (calculates exact expectation).
        """
        # 1. Bind parameters to the ansatz (which has NO measurements)
        param_dict = {}
        n = len(params) // 2
        for i in range(n):
            param_dict[f'β{i}'] = float(params[i])
            param_dict[f'γ{i}'] = float(params[n+i])
        
        bound_circuit = self._ansatz.assign_parameters(param_dict)
        
        # *** FIX: Add save_statevector() instruction ***
        # We make a copy to avoid modifying the class's template ansatz
        circuit_to_run = bound_circuit.copy()
        circuit_to_run.save_statevector() # Tells the sim to save the state
        
        # 2. Transpile the circuit *with* the save instruction
        transpiled_circuit = self._pass_manager.run(circuit_to_run)
        
        # 3. Run simulation
        job = self._backend.run(transpiled_circuit)
        result = job.result()

        # 4. Get the saved statevector
        # Now it will find the state saved by save_statevector()
        statevector = result.get_statevector() 
        
        # 5. Compute exact expectation value: <E> = <ψ|H_C|ψ>
        expectation = float(np.real(statevector.expectation_value(self._H_C)))
        
        # 6. Store history
        self.energy_history.append(expectation)
        return expectation

    def callback_print(self, xk):
        """
        Callback function for scipy.minimize to print progress.
        'xk' is the current parameter vector.
        """
        self.iteration += 1
        current_energy = self.energy_history[-1] # Get last computed energy
        
        if current_energy < self.best_energy:
            self.best_energy = current_energy
            print(f"✓ Iter {self.iteration:3d}: E = {current_energy:8.4f} (NEW BEST)")
        elif self.iteration % 5 == 0:
            print(f"  Iter {self.iteration:3d}: E = {current_energy:8.4f}")

    def get_runtime(self):
        return time.time() - self._start_time

def build_qaoa_circuit(total_qubits, H_C, H_M, reps):
    """
    Builds the parameterized QAOA circuit.
    
    Args:
        total_qubits (int): Total number of qubits.
        H_C (SparsePauliOp): Cost Hamiltonian.
        H_M (SparsePauliOp): Mixer Hamiltonian.
        reps (int): Number of QAOA layers (p).

    Returns:
        tuple: (qaoa_ansatz, qaoa_circuit_measured)
            - ansatz: Circuit without measurements (for statevector sim).
            - circuit_measured: Circuit with measurements (for final counts).
    """
    beta_params = [Parameter(f'β{i}') for i in range(reps)]
    gamma_params = [Parameter(f'γ{i}') for i in range(reps)]
    
    ansatz = QuantumCircuit(total_qubits)
    ansatz.h(range(total_qubits)) # Initial state
    ansatz.barrier()
    
    for p in range(reps):
        # Cost Hamiltonian evolution: exp(-i * gamma * H_C)
        cost_evolution = PauliEvolutionGate(
            H_C, 
            time=gamma_params[p],
            synthesis=LieTrotter(reps=1)
        )
        ansatz.append(cost_evolution, range(total_qubits))
        ansatz.barrier()
        
        # Mixer Hamiltonian evolution: exp(-i * beta * H_M)
        mixer_evolution = PauliEvolutionGate(
            H_M,
            time=beta_params[p],
            synthesis=LieTrotter(reps=1)
        )
        ansatz.append(mixer_evolution, range(total_qubits))
        ansatz.barrier()
    
    # Create a second version of the circuit *with* measurements
    qaoa_circuit_measured = ansatz.copy()
    qaoa_circuit_measured.measure_all()
    
    return ansatz, qaoa_circuit_measured

def run_qaoa_optimization(
    H_C, H_M, reps, 
    maxiter=30, shots=4096
):
    """
    Runs the full QAOA optimization workflow.

    Args:
        H_C (SparsePauliOp): Cost Hamiltonian.
        H_M (SparsePauliOp): Mixer Hamiltonian.
        reps (int): Number of QAOA layers (p).
        maxiter (int): Max iterations for the COBYLA optimizer.
        shots (int): Number of shots for the final measurement.

    Returns:
        tuple: (result, final_counts, energy_history, optimization_time)
            - result: The full result object from scipy.minimize.
            - final_counts (dict): High-shot counts from the optimal parameters.
            - energy_history (list): Energy at each iteration.
            - optimization_time (float): Total time for optimization.
    """
    
    total_qubits = H_C.num_qubits
    num_params = 2 * reps
    
    print("\n" + "="*60)
    print("  QAOA OPTIMIZATION SETUP")
    print("="*60)
    
    # 1. Setup Backend (Statevector for cost function)
    backend = AerSimulator(method='statevector')
    print(f"Backend: {backend.name} (method: {backend.options.method})")
    
    # 2. Build QAOA circuits (one for statevector, one for measurement)
    qaoa_ansatz, qaoa_circuit_measured = build_qaoa_circuit(
        total_qubits, H_C, H_M, reps
    )
    
    print(f"\n✓ QAOA Circuit created:")
    print(f"  - Layers (p): {reps}")
    print(f"  - Total qubits: {total_qubits}")
    print(f"  - Parameters: {num_params}")

    # 3. Setup Transpiler
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    print("\n✓ Transpiler configured (level 1)")
    
    # 4. Setup Optimization Callback
    qaoa_callback = QAOACallback(H_C, backend, qaoa_ansatz, pass_manager)

    # 5. Run Optimization
    print("\n" + "="*60)
    print("  STARTING OPTIMIZATION")
    print("="*60)
    
    initial_params = np.array([0.1] * reps + [0.1] * reps)
    
    print(f"Optimizer: COBYLA")
    print(f"Max iterations: {maxiter}")
    print(f"Initial parameters (fixed): {initial_params}")
    print("-" * 60)
    
    result = minimize(
        qaoa_callback.cost_function,
        initial_params,
        method='COBYLA',
        callback=qaoa_callback.callback_print,
        options={
            'maxiter': maxiter,
            'disp': False,
            'rhobeg': 0.5,
            'tol': 1e-10
        }
    )
    
    optimization_time = qaoa_callback.get_runtime()
    energy_history = qaoa_callback.energy_history
    
    print("-" * 60)
    print(f"✓ Optimization complete in {optimization_time:.1f} seconds")
    print(f"  Final energy (statevector): {result.fun:.4f}")

    # 6. Run Final Measurement
    print("\n" + "="*60)
    print(f"  FINAL MEASUREMENT ({shots} shots)")
    print("="*60)

    # Bind optimal params to the circuit *with* measurements
    optimal_params = result.x
    param_dict = {}
    for i in range(reps):
        param_dict[f'β{i}'] = float(optimal_params[i])
        param_dict[f'γ{i}'] = float(optimal_params[i + reps])

    final_circuit = qaoa_circuit_measured.assign_parameters(param_dict)
    final_transpiled = pass_manager.run(final_circuit)

    # Run on a simulator that supports shots
    meas_backend = AerSimulator(method='automatic')
    final_job = meas_backend.run(final_transpiled, shots=shots)
    final_counts = final_job.result().get_counts()
    
    print(f"✓ Final measurement complete. Top result: {max(final_counts, key=final_counts.get)}")

    return result, final_counts, energy_history, optimization_time