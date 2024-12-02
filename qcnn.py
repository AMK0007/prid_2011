import torch
import torchreid
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

def quantum_convolution(circuits, feature_vectors, num_qubits=3, reps=1):
    """
    Applies a quantum convolutional layer using parameterized circuits and transpiles them.
    """
    backend = Aer.get_backend('qasm_simulator')  # Transpile target backend
    
    # The total number of parameters in RealAmplitudes for a given number of qubits and repetitions
    num_parameters = num_qubits * (reps+1)  # Example, this can vary based on your choice of quantum layers
    # Ensure feature vectors are of appropriate size to map to quantum parameters
    # Reduce or pad the feature vector size to match num_parameters
    feature_vectors = feature_vectors[:, :num_parameters]  # Slice feature vectors to match number of parameters

    # Define the quantum convolutional circuit
    qconv_layer = RealAmplitudes(num_qubits, reps=reps)

    transpiled_circuits = []
    for i, circuit in enumerate(circuits):
        input_qubits=circuit.num_qubits
        if input_qubits< num_qubits:
            print(f'input{input_qubits}  num:{num_qubits}')
        # Map the sliced feature vector to parameters
        parameters = feature_vectors[i]
        
        # Bind parameters and combine with the input circuit
        qconv = QuantumCircuit(num_qubits)
        qconv=qconv.compose(qconv_layer.assign_parameters(parameters))
        combined_circuit=circuit.compose(qconv,qubits=list(range(num_qubits)))
        # Transpile the circuit for the specific backend
        
        transpiled_circuit = transpile(combined_circuit, backend)
        transpiled_circuits.append(transpiled_circuit)

    return transpiled_circuits

# Encode the feature vectors using ZZFeatureMap
def encode_with_zzfeaturemap(feature_vectors, num_qubits=3):
    zz_feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    backend = Aer.get_backend('qasm_simulator')  # Backend for transpilation
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
    # Extract data using the correct keys
    if 'img' in batch and 'pid' in batch and 'camid' in batch:
        imgs, pids, camids = batch['img'], batch['pid'], batch['camid']
    else:
        raise ValueError("Batch does not contain the expected keys: 'img', 'pid', 'camid'.")

    # Reduce each tracklet (sequence of images) to a feature vector (mean pooling)
    feature_vectors = imgs.mean(dim=1).view(imgs.size(0), -1)  # Adjust based on image tensor shape
    return feature_vectors.cpu().numpy(), pids, camids

if __name__ == '__main__':
    # Create the VideoDataManager for PRID2011 dataset
    datamanager = torchreid.data.VideoDataManager(
        root='C:/Users/ahmed/OneDrive/Desktop/prid_2011',
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

    for batch_idx, data in enumerate(train_loader):
        try:
            # Preprocess and extract feature vectors
            feature_vectors, pids, camids = preprocess_batch(data)
            print(f"Processed Batch {batch_idx + 1}:")
            print(f"  - Feature vectors shape: {feature_vectors.shape}")

            # Encode feature vectors using ZZFeatureMap
            transpiled_circuits = encode_with_zzfeaturemap(feature_vectors, num_qubits=3)
            print(f"  - Number of transpiled circuits: {len(transpiled_circuits)}")

            # Apply quantum convolutional layer
            qconv_circuits = quantum_convolution(transpiled_circuits, feature_vectors, num_qubits=3)
            print(f"  - Transpiled circuits for quantum convolution:")
            print(f"    Example transpiled circuit:\n{qconv_circuits[0]}")

        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {e}")
            break


        
