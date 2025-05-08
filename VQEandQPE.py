import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Qiskit imports for VQE and QPE
from qiskit import Aer, transpile, QuantumRegister, QuantumCircuit
from qiskit.visualization import plot_histogram
from QPCA.quantumUtilities.numpy_matrix import NumPyMatrix
from qiskit.circuit.library import PhaseEstimation, EfficientSU2
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.algorithms import VQE, NumPyEigensolver
from qiskit.opflow import MatrixOp, PauliExpectation, CircuitSampler, I, X, Y, Z
from qiskit_aer.primitives import Estimator
import scipy.linalg as la
from scipy.signal import find_peaks
from qiskit import ClassicalRegister
from qiskit import IBMQ
import time

# try:
#     IBMQ.load_account()
# except:
#     token = '31e5ce4597f6c1f55076a8f6e2d4810a0aaf53ba57047e995835ce486b994453ab2275a532b02087581ffa01579b26037d69ccb6b9e67e7cdb4d78d495fc2dee'
#     IBMQ.enable_account(token)
    
# provider = IBMQ.get_provider()
# backend = provider.get_backend('ibm_brisbane')
backend = Aer.get_backend("qasm_simulator")

def load_and_prepare_data(file_path=None, delimiter=',', header='infer', dimensionality_reduction=None):
    """
    Load and prepare data for quantum PCA.
    
    Args:
        file_path: Path to the dataset file (CSV format)
        delimiter: Delimiter for CSV file
        header: Header for CSV file
        dimensionality_reduction: Reduce dimensionality using classical PCA before quantum PCA
    
    Returns:
        matrix: Covariance matrix
        original_data: Original standardized data
        reduced_data: Data after dimensionality reduction (if applied)
    """
    if file_path is None:
        # Generate synthetic data if no file provided
        print("No dataset provided. Generating synthetic data...")
        np.random.seed(42)
        n_samples = 100
        n_features = 4
        
        # Create dataset with some structure
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 3 * X[:, 1] + 2 + 0.5 * np.random.randn(n_samples)
        X[:, 2] = -2 * X[:, 3] + 1 + 0.3 * np.random.randn(n_samples)
        
        data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        print(f"Generated synthetic data with {n_samples} samples and {n_features} features")
    else:
        # Load from file
        data = pd.read_csv(file_path, delimiter=delimiter, header=header)
        print(f"Loaded data from {file_path}")
        print(f"Data shape: {data.shape}")
    
    # Remove any non-numeric columns
    numeric_columns = data.select_dtypes(include=np.number).columns
    if len(numeric_columns) < data.shape[1]:
        print(f"Removing {data.shape[1] - len(numeric_columns)} non-numeric columns")
        data = data[numeric_columns]
    
    # Remove rows with missing values
    initial_rows = data.shape[0]
    data = data.dropna()
    if data.shape[0] < initial_rows:
        print(f"Removed {initial_rows - data.shape[0]} rows with missing values")
    
    print(f"Final data shape: {data.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Apply dimensionality reduction if requested
    reduced_data = None
    if dimensionality_reduction is not None and dimensionality_reduction < data.shape[1]:
        print(f"Applying dimensionality reduction to {dimensionality_reduction} features")
        classical_pca = PCA(n_components=dimensionality_reduction)
        reduced_data = classical_pca.fit_transform(X_scaled)
        
        # Compute covariance matrix of reduced data
        cov_matrix = np.cov(reduced_data, rowvar=False)
        
        # Show explained variance
        explained_variance = classical_pca.explained_variance_ratio_
        print(f"Explained variance with {dimensionality_reduction} components: {sum(explained_variance):.2f}")
    else:
        # Compute covariance matrix of original data
        cov_matrix = np.cov(X_scaled, rowvar=False)
    
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print("Sample of covariance matrix (first 5x5):")
    print(np.round(cov_matrix[:min(5, cov_matrix.shape[0]), :min(5, cov_matrix.shape[1])], 3))
    
    return cov_matrix, X_scaled, reduced_data

def prepare_matrix_for_qpca(matrix):
    """
    Prepare a matrix for quantum PCA by ensuring it's Hermitian and normalized.
    
    Args:
        matrix: Input matrix (should already be Hermitian, like a covariance matrix)
    
    Returns:
        normalized_matrix: Normalized matrix
        norm_factor: Normalization factor
    """
    # Ensure the matrix is Hermitian (should be for a covariance matrix)
    if not np.allclose(matrix, matrix.T.conj()):
        print("Warning: Input matrix is not perfectly Hermitian. Symmetrizing...")
        matrix = (matrix + matrix.T.conj()) / 2
    
    # Compute the trace for normalization
    norm_factor = np.trace(matrix)
    normalized_matrix = matrix / norm_factor
    
    # Display eigenvalues and eigenvectors (only first few for large matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f'Matrix shape: {matrix.shape}')
    print(f'Normalization factor: {norm_factor:.4f}')
    
    # For large matrices, only show first few eigenvalues
    max_to_display = min(5, len(eigenvalues))
    for i in range(max_to_display):
        print(f'eigenvalue {i+1}: {eigenvalues[i]:.4f}')
    
    return normalized_matrix, norm_factor, eigenvalues, eigenvectors

def vqe_find_ground_state(matrix, shots=1024, max_iterations=100):
    """
    Use VQE to find the ground state (smallest eigenvalue/eigenvector) of the matrix.
    
    Args:
        matrix: Input matrix as numpy array
        shots: Number of shots for the quantum simulation
        max_iterations: Maximum number of iterations for the optimizer
    
    Returns:
        eigenvalue: Estimated smallest eigenvalue
        eigenstate: Estimated eigenstate (eigenvector)
    """
    n_qubits = int(np.log2(matrix.shape[0]))
    
    try:
        # Create Pauli operator representation of the matrix
        operator = MatrixOp(matrix)
        
        # Create a parameterized circuit for the ansatz
        ansatz = EfficientSU2(n_qubits, reps=3, entanglement='linear')
        
        # Initialize the optimizer
        optimizer = COBYLA(maxiter=max_iterations)
        
        # Set up backend with shots
        backend.set_options(shots=shots)
        
        # Use VQE to find the ground state
        vqe = VQE(ansatz, optimizer, quantum_instance=backend)
        result = vqe.compute_minimum_eigenvalue(operator)
        
        # Extract eigenvalue
        eigenvalue = result.eigenvalue.real
        
        # Get the eigenstate as a statevector using a separate statevector simulation
        eigenstate_circuit = ansatz.bind_parameters(result.optimal_parameters)
        
        # Use statevector_simulator specifically for getting the statevector
        backend_sv = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(eigenstate_circuit, backend_sv)
        job = backend_sv.run(transpiled_circuit)
        eigenstate = job.result().get_statevector().data
        
        # Convert to numpy array
        eigenstate = np.array(eigenstate)
        
        return eigenvalue, eigenstate
    
    except Exception as e:
        print(f"Error in VQE: {e}")
        # Fall back to classical eigenvalue solver
        print("Falling back to classical eigensolver...")
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = eigenvalues.argsort()[0]  # Get index of smallest eigenvalue
        return eigenvalues[idx], eigenvectors[:, idx]
    
def deflate_matrix(matrix, eigenvalue, eigenvector):
    """
    Deflate a matrix by removing the contribution of an eigenvector.
    
    Args:
        matrix: Input matrix
        eigenvalue: Eigenvalue to remove
        eigenvector: Corresponding eigenvector
    
    Returns:
        deflated_matrix: Matrix with eigenvector contribution removed
    """
    # Ensure eigenvector is normalized
    eigenvector = eigenvector / np.linalg.norm(eigenvector)
    
    # Deflate matrix using Hotelling's deflation
    deflated_matrix = matrix - eigenvalue * np.outer(eigenvector, eigenvector)
    
    # Re-symmetrize to handle numerical errors
    deflated_matrix = (deflated_matrix + deflated_matrix.T) / 2
    
    return deflated_matrix

def find_largest_eigenvalues(matrix, n_components, shots=1024, max_iterations=100):
    """
    Find the largest eigenvalues and eigenvectors of a matrix using VQE with deflation.
    
    Args:
        matrix: Input matrix as numpy array
        n_components: Number of components (eigenvalues/eigenvectors) to find
        shots: Number of shots for VQE
        max_iterations: Maximum iterations for optimizer
    
    Returns:
        eigenvalues: List of eigenvalues (in descending order)
        eigenvectors: List of corresponding eigenvectors
    """
    # We need to invert the matrix to find largest eigenvalues with VQE (which finds smallest)
    # First, find the maximum eigenvalue to shift the matrix
    max_eig = np.max(np.linalg.eigvalsh(matrix)) * 1.1  # Add 10% buffer
    
    # Shift and invert: A' = max_eig*I - A
    # This transforms the problem so that the largest eigenvalue becomes the smallest
    inverted_matrix = max_eig * np.eye(matrix.shape[0]) - matrix
    
    eigenvalues = []
    eigenvectors = []
    
    current_matrix = inverted_matrix.copy()
    
    for i in range(n_components):
        print(f"Finding eigenvalue/eigenvector {i+1}...")
        
        # Find the smallest eigenvalue/eigenvector of the current matrix
        inv_eigenval, inv_eigenvec = vqe_find_ground_state(
            current_matrix, 
            shots=shots, 
            max_iterations=max_iterations
        )
        
        # Convert back to original eigenvalue
        eigenvalue = max_eig - inv_eigenval
        eigenvector = inv_eigenvec
        
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        
        # Deflate the matrix to find the next eigenvector
        if i < n_components - 1:
            current_matrix = deflate_matrix(current_matrix, inv_eigenval, eigenvector)
    
    return eigenvalues, eigenvectors

def qpe_estimate_eigenvalues(matrix, eigenvectors, resolution=5, n_shots=1000):
    """
    Use Quantum Phase Estimation to refine eigenvalue estimates.
    
    Args:
        matrix: Input matrix
        eigenvectors: Eigenvectors to estimate eigenvalues for
        resolution: Number of qubits for phase estimation
        n_shots: Number of measurement shots
    
    Returns:
        refined_eigenvalues: More precisely estimated eigenvalues
    """
    # Unitary operator for PE
    u_op = NumPyMatrix(matrix, evolution_time=2*np.pi)
    # try:
        # Try to get the provider if already logged in
        # provider = IBMQ.get_provider()
    # except:
    #     # If not logged in, load the account from disk if saved previously
    #     try:
    #         IBMQ.load_account()
    #         provider = IBMQ.get_provider()
    #     except:
    #         # If not saved previously, enable the account with the token
    #         IBMQ.enable_account('bbc1b3c771eb0cb5932e5b5df216bebfb698f2bb299059d9e5102e75d964912060c01c0c78eb5e3226ef11037d88c7232a958153c443d59c458def0bc24ead8e')
    #         provider = IBMQ.get_provider()
    # backend = provider.get_backend('ibm_sherbrooke')
    refined_eigenvalues = []
    
    for i, eigenvector in enumerate(eigenvectors):
        print(f"Running QPE for eigenvector {i+1}...")
        
        # Number of qubits needed for the state
        n_state_qubits = int(np.log2(matrix.shape[0]))
        
        # Ensure eigenvector is normalized and convert to complex if needed
        eigenvector = eigenvector / np.linalg.norm(eigenvector)
        if not np.iscomplexobj(eigenvector):
            eigenvector = eigenvector.astype(complex)
        
        # Create the eigenstate preparation circuit
        eigenstate_circuit = QuantumCircuit(n_state_qubits)
        
        # Prepare the eigenstate
        # For a general state, we would use state preparation
        # But for demonstration, we use a simple approach for small matrices
        if n_state_qubits == 1:
            # For 1-qubit case
            theta = 2 * np.arccos(abs(eigenvector[0]))
            phi = np.angle(eigenvector[1]) - np.angle(eigenvector[0]) if abs(eigenvector[0]) > 1e-10 else 0
            eigenstate_circuit.ry(theta, 0)
            eigenstate_circuit.rz(phi, 0)
        elif n_state_qubits == 2:
            # For 2-qubit case, try using StatePreparation with properly normalized vector
            try:
                from qiskit.circuit.library import StatePreparation
                eigenstate_circuit = QuantumCircuit(n_state_qubits)
                eigenstate_circuit.append(StatePreparation(eigenvector), range(n_state_qubits))
            except Exception as e:
                print(f"Error in state preparation: {e}")
                print("Using approximate preparation...")
                # Simple approximate preparation
                if abs(eigenvector[0]) > 0.1:
                    eigenstate_circuit.ry(2*np.arccos(abs(eigenvector[0])), 0)
                if len(eigenvector) >= 4 and abs(eigenvector[0])+abs(eigenvector[1]) > 0.1:
                    eigenstate_circuit.ry(2*np.arccos(abs(eigenvector[0])/np.sqrt(abs(eigenvector[0])**2+abs(eigenvector[1])**2)), 1)
        else:
            # For larger matrices - use StatePreparation with properly normalized vector
            try:
                from qiskit.circuit.library import StatePreparation
                eigenstate_circuit = QuantumCircuit(n_state_qubits)
                eigenstate_circuit.append(StatePreparation(eigenvector), range(n_state_qubits))
            except Exception as e:
                print(f"Error in state preparation for larger matrix: {e}")
                print("Using classical eigenvalue...")
                # Return classical eigenvalue estimate
                eigenvalue = np.vdot(eigenvector, matrix @ eigenvector) / np.vdot(eigenvector, eigenvector)
                refined_eigenvalues.append(eigenvalue.real)
                continue
        
        # Create the phase estimation circuit
        pe = PhaseEstimation(resolution, u_op)
        
        # Create the full circuit
        qr_pe = QuantumRegister(resolution, 'pe')
        qr_state = QuantumRegister(n_state_qubits, 'state')
        cr = ClassicalRegister(resolution, 'c')
        qpe_circuit = QuantumCircuit(qr_pe, qr_state, cr)
        
        # Prepare eigenstate
        qpe_circuit.append(eigenstate_circuit, qr_state)
        
        # Apply phase estimation
        qpe_circuit.append(pe, qr_pe[:] + qr_state[:])
        
        # Measure phase register
        qpe_circuit.measure(qr_pe, cr)
        
        # Run the circuit
        transpiled_circuit = transpile(qpe_circuit, backend)
        job = backend.run(transpiled_circuit, shots=n_shots)
        result = job.result()
        counts = result.get_counts()
        
        # Extract phase from measurement results
        phase_estimates = []
        for bitstring, count in counts.items():
            phase = int(bitstring, 2) / (2**resolution)
            phase_estimates.extend([phase] * count)
        
        # Calculate mean phase
        mean_phase = np.mean(phase_estimates)
        
        # Convert phase to eigenvalue (accounting for possible 2π wrapping)
        eigenvalue = mean_phase
        refined_eigenvalues.append(eigenvalue)
    
    # Convert to actual eigenvalues by multiplying with normalization factor
    return refined_eigenvalues

def run_hybrid_quantum_pca(matrix, n_components=2, vqe_shots=1024, qpe_resolution=5, qpe_shots=1000):
    """
    Run hybrid quantum PCA using VQE for eigenvectors and QPE for eigenvalues.
    
    Args:
        matrix: Input matrix
        n_components: Number of principal components to find
        vqe_shots: Number of shots for VQE
        qpe_resolution: Resolution for QPE
        qpe_shots: Number of shots for QPE
    
    Returns:
        eigenvalues: Estimated eigenvalues
        eigenvectors: Estimated eigenvectors
    """
    print("\n=== Step 1: Preparing matrix ===")
    normalized_matrix, norm_factor, classical_eigenvalues, classical_eigenvectors = prepare_matrix_for_qpca(matrix)
    
    print("\n=== Step 2: Finding eigenvectors using VQE ===")
    vqe_eigenvalues, vqe_eigenvectors = find_largest_eigenvalues(
        normalized_matrix, 
        n_components=n_components,
        shots=vqe_shots
    )
    
    # Convert eigenvalues back to original scale
    vqe_eigenvalues = [eig * norm_factor for eig in vqe_eigenvalues]
    
    print("\n=== Step 3: Refining eigenvalues using QPE ===")
    # For small matrices, QPE might not add much value over VQE
    # But for larger ones, it can improve precision
    if matrix.shape[0] <= 4:  # For small matrices
        try:
            qpe_eigenvalues = qpe_estimate_eigenvalues(
                normalized_matrix,
                vqe_eigenvectors,
                resolution=qpe_resolution,
                n_shots=qpe_shots
            )
            # Convert eigenvalues back to original scale
            qpe_eigenvalues = [eig * norm_factor for eig in qpe_eigenvalues]
        except Exception as e:
            print(f"Error in QPE: {e}")
            print("Using VQE eigenvalues instead.")
            qpe_eigenvalues = vqe_eigenvalues
    else:
        print("Matrix too large for QPE simulation. Using VQE eigenvalues.")
        qpe_eigenvalues = vqe_eigenvalues
    
    # Return results - use QPE eigenvalues if available, otherwise VQE
    if len(qpe_eigenvalues) == n_components:
        return qpe_eigenvalues, vqe_eigenvectors
    else:
        return vqe_eigenvalues, vqe_eigenvectors

def main(file_path=None, delimiter=',', header='infer', dimensionality_reduction=None, 
         n_components=2, vqe_shots=1024, qpe_resolution=5, qpe_shots=1000, debug=False):
    """
    Main function to run hybrid quantum PCA on a dataset.
    
    Args:
        file_path: Path to the dataset file (CSV format)
        delimiter: Delimiter for CSV file
        header: Header for CSV file
        dimensionality_reduction: Reduce dimensionality using classical PCA before quantum PCA
        n_components: Number of principal components to find
        vqe_shots: Number of shots for VQE
        qpe_resolution: Resolution for QPE
        qpe_shots: Number of shots for QPE
        debug: Whether to print debug information
    """
    # Load and prepare data
    print("=== Step 1: Loading and preparing data ===")
    # try:
    #     # Try to get the provider if already logged in
    #     provider = IBMQ.get_provider()
    # except:
    #     # If not logged in, load the account from disk if saved previously
    #     try:
    #         IBMQ.load_account()
    #         provider = IBMQ.get_provider()
    #     except:
    #         # If not saved previously, enable the account with the token
    #         IBMQ.enable_account('bbc1b3c771eb0cb5932e5b5df216bebfb698f2bb299059d9e5102e75d964912060c01c0c78eb5e3226ef11037d88c7232a958153c443d59c458def0bc24ead8e')
    #         provider = IBMQ.get_provider()
    # backends = provider.get_backend('ibm_sherbrooke')
    #start time count
    start_time = time.time()
    
    matrix, original_data, reduced_data = load_and_prepare_data(
        file_path=file_path, 
        delimiter=delimiter,
        header=header,
        dimensionality_reduction=dimensionality_reduction
    )
    
    # Check if the matrix size is suitable for quantum simulation
    matrix_size = matrix.shape[0]
    qubits_needed = int(np.log2(matrix_size)) * 2 + qpe_resolution  # Approximate
    
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Estimated qubits needed: {qubits_needed}")
    
    if matrix_size > 4 and dimensionality_reduction is None:
        print(f"WARNING: Matrix size {matrix_size} may be too large for simulation.")
        print("Consider using dimensionality reduction with the --reduce parameter.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborting.")
            return
    
    # Run hybrid quantum PCA
    # print("\n=== Step 2: Running Hybrid Quantum PCA (VQE + QPE) ===")
    quantum_eigenvalues, quantum_eigenvectors = run_hybrid_quantum_pca(
        matrix=matrix,
        n_components=n_components,
        vqe_shots=vqe_shots,
        qpe_resolution=qpe_resolution,
        qpe_shots=qpe_shots
    )
    
    # Compare with classical PCA
    print("\n=== Step 3: Comparing with Classical PCA ===")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Display results
    print("\n=== Hybrid Quantum PCA Results (VQE + QPE) ===")
    for i, (eigenval, eigenvec) in enumerate(zip(quantum_eigenvalues, quantum_eigenvectors)):
        if i < min(n_components, 5):  # Show only first few for large matrices
            print(f'Eigenvalue {i+1}: {eigenval:.4f}')
            if matrix_size <= 4:  # Show eigenvectors only for small matrices
                print(f'Eigenvector {i+1}: {np.real(eigenvec).round(3)}')
    
    print("\n=== Classical PCA Results ===")
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        if i < min(n_components, 5):  # Show only first few for large matrices
            print(f'Eigenvalue {i+1}: {eigenval:.4f}')
            if matrix_size <= 4:  # Show eigenvectors only for small matrices
                print(f'Eigenvector {i+1}: {eigenvec.round(3)}')
    
    # Visualize the comparison
    print("\n=== Visualizing Results ===")
    plt.figure(figsize=(12, 6))
    
    # For large matrices, limit to top 10 eigenvalues
    max_to_display = min(n_components, 5)
    
    # plt.subplot(1, 2, 1)
    # plt.bar(range(len(quantum_eigenvalues[:max_to_display])), quantum_eigenvalues[:max_to_display])
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    # plt.title('Hybrid Quantum PCA Results')
    # plt.xticks(range(len(quantum_eigenvalues[:max_to_display])))
    
    # plt.subplot(1, 2, 2)
    # plt.bar(range(max_to_display), eigenvalues[:max_to_display])
    # plt.xlabel('Principal Component')
    # plt.ylabel('Eigenvalue')
    # plt.title('Classical PCA Results')
    # plt.xticks(range(max_to_display))
    
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('comparison.png')
    # Print elapsed time
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    
    # Calculate error metrics
    if matrix_size <= 4 and len(quantum_eigenvalues) > 0:  # Only for small matrices
        print("\n=== Error Analysis ===")
        mse_eigenvalues = np.mean([(q - c)**2 
                                  for q, c in zip(quantum_eigenvalues[:max_to_display], 
                                                  eigenvalues[:max_to_display])])
        print(f"Mean Squared Error for eigenvalues: {mse_eigenvalues:.6f}")
        
        # Compare MSE for different shot counts
        if matrix_size <= 4:  # Only for small matrices
            print("\n=== Comparing Error vs QPE Shots ===")
            
            shot_counts = [10, 100, 200]
            mse_values = []
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Store original results
            original_eigenvalues = quantum_eigenvalues.copy()
            
            for shots in shot_counts:
                print(f"\nRunning with {shots} QPE shots...")
                
                # Re-run quantum PCA with different shot count
                quantum_eigen_values, _ = run_hybrid_quantum_pca(
                    matrix=matrix,
                    n_components=shots,
                    vqe_shots=vqe_shots,
                    qpe_resolution=qpe_resolution,
                    qpe_shots=qpe_shots  # Using QPE shots here instead of VQE shots
                )
                
                # Calculate MSE - ensure same dimensions
                actual_display = min(max_to_display, len(quantum_eigen_values), len(eigenvalues))
                mse = np.mean([(q - c)**2 
                              for q, c in zip(quantum_eigen_values[:actual_display], 
                                              eigenvalues[:actual_display])])
                mse_values.append(mse)
                print(f"Mean Squared Error with {shots} QPE shots: {mse:.6f}")
                
                # Plot eigenvalues comparison - ensure same dimensions
                ax1.plot(range(1, actual_display+1), 
                        quantum_eigen_values[:actual_display], 
                        marker='o', label=f'{shots} shots')
        
            # Add classical eigenvalues to the plot - ensure same dimensions
            actual_display = min(max_to_display, len(quantum_eigen_values), len(eigenvalues))
            ax1.plot(range(1, actual_display+1), eigenvalues[:actual_display], 
                    marker='s', linestyle='--', color='black', label='Classical')
            
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Eigenvalue')
            ax1.set_title('Eigenvalues: Quantum vs Classical')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MSE vs shots
            ax2.plot(shot_counts, mse_values, marker='o', linestyle='-', color='blue')
            ax2.set_xlabel('Number of QPE Shots')  # Changed from VQE to QPE
            ax2.set_ylabel('Mean Squared Error')
            ax2.set_title('Error vs QPE Shots')  # Changed from VQE to QPE
            ax2.set_xscale('log')  # Log scale makes it easier to see differences
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('quantum_pca_qpe_error_analysis.png')  # Changed filename
            plt.show()
            
            # Restore original results
            quantum_eigenvalues = original_eigenvalues

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hybrid Quantum PCA (VQE + QPE) on a dataset')
    parser.add_argument('--file', type=str, help='Path to the dataset file (CSV format)', default=None)
    parser.add_argument('--delimiter', type=str, help='Delimiter for CSV file', default=',')
    parser.add_argument('--header', type=str, help='Header for CSV file', default='infer')
    parser.add_argument('--reduce', type=int, help='Reduce dimensionality using classical PCA', default=None)
    parser.add_argument('--components', type=int, help='Number of principal components to find', default=2)
    parser.add_argument('--vqe_shots', type=int, help='Number of shots for VQE', default=1024)
    parser.add_argument('--qpe_resolution', type=int, help='Resolution for QPE', default=5)
    parser.add_argument('--qpe_shots', type=int, help='Number of shots for QPE', default=1000)
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    main(
        file_path='Iris.csv',
        delimiter=',',
        header='infer',
        dimensionality_reduction=2,
        n_components=10,
        vqe_shots=1024,
        qpe_resolution=5,
        qpe_shots=100,
        debug=False,
    )