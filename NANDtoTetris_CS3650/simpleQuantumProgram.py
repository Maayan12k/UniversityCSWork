from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Alice generates a random secret bit string
alice_secret_key = '101001'

# Create a quantum circuit with a pair of qubits for Alice and Bob
qc = QuantumCircuit(2, 2)

# Alice prepares qubit
if alice_secret_key[0] == '1':
    qc.x(0)

# Alice chooses a basis (0 for Z-basis, 1 for X-basis) randomly for encoding
import numpy as np
alice_basis = np.random.randint(2)
if alice_basis == 1:
    qc.h(0)  # Apply a Hadamard gate if X-basis

# Alice sends the qubit to Bob
qc.barrier()

# Bob randomly chooses a basis for measurement
bob_basis = np.random.randint(2)
if bob_basis == 1:
    qc.h(1)  # Apply a Hadamard gate if X-basis

# Perform a measurement
qc.measure([0, 1], [0, 1])

# Simulate the quantum circuit
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1).result()
counts = result.get_counts()

# Bob measures the qubit
bob_result = list(counts.keys())[0]

# Bob tells Alice his basis choice
alice_basis_message = str(alice_basis)
bob_basis_message = str(bob_basis)

# Alice and Bob compare their basis choice
if alice_basis_message == bob_basis_message:
    shared_key = bob_result[0]  # The shared secret key bit

# Print the results
print("Alice's secret key:", alice_secret_key)
print("Bob's secret key:", shared_key)
