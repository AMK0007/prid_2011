import torch
import torchreid
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator,Aer

def encode_with_zzfeaturemap(feature_vectors, num_qubits):
    """
    Encodes feature vectors into quantum states using ZZFeatureMap.
    """
    zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuits = []

    for vector in feature_vectors:
        # Normalize vector to match feature map input size
        vector = vector[:num_qubits] if len(vector) >= num_qubits else np.pad(vector, (0, num_qubits - len(vector)))
        # Map the vector to a quantum circuit
        circuit = zz_feature_map.assign_parameters(vector)
        # Transpile the circuit for the given backend
        transpiled_circuit = transpile(circuit, backend)
        transpiled_circuits.append(transpiled_circuit)
        
    return transpiled_circuits

def preprocess_batch(batch):
    """
    Preprocesses the batch to extract feature vectors, PIDs, and CAMIDs.
    """
    if 'img' in batch and 'pid' in batch and 'camid' in batch:
        imgs, pids, camids = batch['img'], batch['pid'], batch['camid']
    else:
        raise ValueError("Batch does not contain the expected keys: 'img', 'pid', 'camid'.")

    # Reduce each tracklet (sequence of images) to a feature vector (mean pooling)
    feature_vectors = imgs.mean(dim=1).view(imgs.size(0), -1)
    return feature_vectors.cpu().numpy(), pids, camids

def parameter_shift_gradient(circuit, params, backend, observable, qubits):
    """
    Compute gradients of a quantum circuit using the parameter shift rule.

    Args:
        circuit: Parameterized QuantumCircuit.
        params: Current parameter values (1D numpy array).
        backend: Qiskit backend for simulation.
        observable: Measurement operator (e.g., Pauli-Z observable).
        qubits: Qubits to measure.

    Returns:
        Gradients for each parameter (numpy array).
    """
    gradients = np.zeros_like(params)
    shift = np.pi / 2  # Parameter shift value
    
    # Transpile the circuit once for the backend
    transpiled_circuit = transpile(circuit, backend)
    
    for i in range(len(params)):
        # Create shifted parameters
        shifted_params_plus = params.copy()
        shifted_params_minus = params.copy()
        shifted_params_plus[i] += shift
        shifted_params_minus[i] -= shift

        # Assign shifted parameters to the circuit
        circuit_plus = transpiled_circuit.assign_parameters(shifted_params_plus)
        circuit_minus = transpiled_circuit.assign_parameters(shifted_params_minus)

        # Ensure the circuit includes a measurement
        circuit_plus.measure_all()  # Measure all qubits
        circuit_minus.measure_all()  # Measure all qubits

        # Execute the circuits to compute expectation values
        counts_plus = execute_transpiled_circuit(circuit_plus, backend)
        counts_minus = execute_transpiled_circuit(circuit_minus, backend)

        exp_plus = expectation_value_from_counts(counts_plus, qubits, observable)
        exp_minus = expectation_value_from_counts(counts_minus, qubits, observable)

        # Compute gradient using the parameter shift rule
        gradients[i] = (exp_plus - exp_minus) / 2

    return gradients

def execute_transpiled_circuit(transpiled_circuit, backend):
    """
    Execute a transpiled circuit and return measurement counts.
    """
    result = backend.run(transpiled_circuit, shots=1024).result()
    return result.get_counts()

def expectation_value_from_counts(counts, qubits, observable):
    """
    Compute the expectation value from measurement counts.
    """
    total_shots = sum(counts.values())
    expectation = 0

    for bitstring, count in counts.items():
        # Map bitstring to eigenvalue (+1 or -1) based on observable
        eigenvalue = 1
        for q in qubits:
            eigenvalue *= 1 if bitstring[q] == '0' else -1
        expectation += eigenvalue * (count / total_shots)

    return expectation

def quantum_convolution_with_gradients(circuits, feature_vectors, num_qubits, reps=1):
    """
    Applies a quantum convolutional layer using parameterized circuits,
    computes the transpiled circuits, and calculates gradients with the
    parameter shift rule.

    Args:
        circuits: List of QuantumCircuit objects.
        feature_vectors: Feature vectors to map into parameters.
        num_qubits: Number of qubits for the quantum convolution.
        reps: Number of repetitions for the RealAmplitudes circuit.

    Returns:
        transpiled_circuits: Transpiled circuits after applying convolution.
        gradients_list: Gradients of parameters for each circuit.
    """
    backend = AerSimulator()  # Transpile target backend

    # Number of parameters for RealAmplitudes circuit
    num_parameters = num_qubits * (reps + 1)

    # Ensure feature vectors are appropriately sized
    feature_vectors = feature_vectors[:, :num_parameters]  # Slice to match parameters

    # Define the quantum convolutional layer
    qconv_layer = RealAmplitudes(num_qubits, reps=reps)

    transpiled_circuits = []
    gradients_list = []

    for i, circuit in enumerate(circuits):
        # Ensure qubit alignment
        if circuit.num_qubits < num_qubits:
            print(f"Circuit qubits ({circuit.num_qubits}) less than required ({num_qubits}). Padding with identity gates.")
            identity = QuantumCircuit(num_qubits)
            circuit = circuit.compose(identity)

        # Map feature vector to parameters
        parameters = feature_vectors[i]

        # Assign parameters and combine with the input circuit
        qconv = qconv_layer.assign_parameters(parameters)
        combined_circuit = circuit.compose(qconv, qubits=list(range(num_qubits)))

        # Ensure the combined circuit includes a measurement
        combined_circuit.measure_all()  # Measure all qubits

        # Transpile the combined circuit
        transpiled_circuit = transpile(combined_circuit, backend)
        transpiled_circuits.append(transpiled_circuit)

        # Compute gradients for the circuit using parameter shift rule
        gradients = parameter_shift_gradient(qconv_layer, parameters, backend, 'Z', list(range(num_qubits)))
        gradients_list.append(gradients)

    return transpiled_circuits, gradients_list


if __name__ == '__main__':
    num_qubits = 3
    reps = 2

    # Initialize VideoDataManager for PRID2011 dataset
    try:
        datamanager = torchreid.data.VideoDataManager(
            root='G:/K214502/prid_2011',
            sources='prid2011',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100,
            seq_len=15,
            sample_method='evenly',
            transforms=['random_flip', 'random_crop', 'resize', 'normalize']
        )
        train_loader = datamanager.train_loader
    except Exception as e:
        print(f"Error initializing data manager: {e}")
        exit()

    for batch_idx, data in enumerate(train_loader):
        try:
            # Preprocess and extract feature vectors
            feature_vectors, pids, camids = preprocess_batch(data)
            print(f"Processed Batch {batch_idx + 1}:")
            print(f"  - Feature vectors shape: {feature_vectors.shape}")

            # Encode feature vectors using ZZFeatureMap
            transpiled_circuits = encode_with_zzfeaturemap(feature_vectors, num_qubits)
            print(f"  - Number of transpiled circuits: {len(transpiled_circuits)}")

            # Apply quantum convolutional layer with gradients
            qconv_circuits, gradients_list = quantum_convolution_with_gradients(
                transpiled_circuits, feature_vectors, num_qubits, reps=reps
            )
            print(f"  - Transpiled circuits for quantum convolution: {len(qconv_circuits)}")
            print(f"  - Gradients for first circuit:\n{gradients_list[0]}")

        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            break
