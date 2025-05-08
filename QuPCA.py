import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Import QPCA module - assuming it's properly installed or in the current directory
# FROM THIS REPO https://github.com/Eagle-quantum/QuPCA
from QPCA.decomposition.Qpca import QPCA
from qiskit import Aer
from qiskit.primitives import BackendSampler

# 1. Load the Iris dataset
print("Loading Iris dataset...")
iris = pd.read_csv('Iris.csv')
print(f"Dataset shape: {iris.shape}")
print(iris.head())

# 2. Preprocess the data - extract only the feature columns
# We'll use the 4 numerical features (sepal length, sepal width, petal length, petal width)
features = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

# 3. Scale the features for better results
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. Compute the covariance matrix - this is what QPCA will work with
cov_matrix = np.cov(features_scaled.T)
print("\nCovariance Matrix:")
print(cov_matrix)

# 5. Classical PCA for comparison
# Compute eigenvalues and eigenvectors classically
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]  # Sort in descending order
classical_eigenvalues = eigenvalues[idx]
classical_eigenvectors = eigenvectors[:, idx]

print("\nClassical PCA Results:")
print(f"Eigenvalues: {classical_eigenvalues}")
print(f"Explained variance ratio: {classical_eigenvalues / sum(classical_eigenvalues)}")

# 6. Apply QPCA 
print("\nApplying QPCA...")
# Initialize backend
backend = BackendSampler(Aer.get_backend('qasm_simulator'))

# Setup QPCA parameters
resolution = 5  # Number of qubits for phase estimation
n_shots = 10000  # Number of measurements

# Initialize QPCA
qpca = QPCA()

# Fit the model
print("Fitting QPCA model...")
qpca.fit(cov_matrix, resolution=resolution, optimized_qram=True)

# Reconstruct eigenvalues and eigenvectors
print("Reconstructing eigenvalues and eigenvectors...")
try:
    quantum_eigenvalues, quantum_eigenvectors = qpca.eigenvectors_reconstruction(
        n_shots=n_shots, 
        n_repetitions=1, 
        backend=backend, 
        eigenvalue_threshold=0.01,
        abs_tolerance=0.001
    )
    
    # Sort results
    q_idx = quantum_eigenvalues.argsort()[::-1]
    quantum_eigenvalues = quantum_eigenvalues[q_idx]
    quantum_eigenvectors = quantum_eigenvectors[:, q_idx]
    
    print("\nQPCA Results:")
    print(f"Quantum Eigenvalues: {quantum_eigenvalues}")
    print(f"Quantum Eigenvectors: {quantum_eigenvectors}")
    
    # 7. Compare classical and quantum results
    print("\nComparing Classical and Quantum Results:")
    
    # Benchmark the results
    try:
        results = qpca.spectral_benchmarking(
            eigenvector_benchmarking=True,
            eigenvalues_benchmarching=True,
            sign_benchmarking=True,
            print_distances=True,
            only_first_eigenvectors=False,
            plot_delta=True,
            distance_type='l2',
            error_with_sign=True,
            hide_plot=False,
            print_error=True
        )
        print("Benchmarking completed successfully")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
    
    # 8. Project data onto principal components
    # Let's use the first 2 PCs from both methods
    
    # Classical projection
    classical_projection = features_scaled @ classical_eigenvectors[:, :2]
    
    # Quantum projection (if available)
    quantum_projection = features_scaled @ quantum_eigenvectors[:, :2]
    
    # 9. Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the projection of the data using classical PCA
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
    for species in iris['Species'].unique():
        indices = iris['Species'] == species
        ax1.scatter(
            classical_projection[indices, 0],
            classical_projection[indices, 1],
            c=colors[species],
            label=species,
            alpha=0.7
        )
    ax1.set_title('Classical PCA')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the projection of the data using QPCA
    for species in iris['Species'].unique():
        indices = iris['Species'] == species
        ax2.scatter(
            quantum_projection[indices, 0],
            quantum_projection[indices, 1],
            c=colors[species],
            label=species,
            alpha=0.7
        )
    ax2.set_title('Quantum PCA')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('pca_comparison.png')

except Exception as e:
    print(f"Error during QPCA reconstruction: {e}")
    print("This may happen due to insufficient measurements or resolution.")
    print("Try increasing n_shots or resolution parameter.")