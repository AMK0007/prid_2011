import torch
import numpy as np
from torchreid import data
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import Aer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, average_precision_score

def zz_feature_map_encoding(image, num_qubits):
    """
    Encode the classical image data using ZZFeatureMap in Qiskit.
    
    Args:
    - image: The input image data, typically a flattened vector of pixel values.
    - num_qubits: The number of qubits for encoding. This should match the number of features in the image.
    
    Returns:
    - quantum_circuit: The quantum circuit representing the encoded image.
    """
    # Normalize image to the range [0, 2*pi] for encoding
    normalized_image = np.array(image) / np.max(image) * 2 * np.pi

    # Ensure the size of the image matches the number of qubits
    if len(normalized_image) != num_qubits:
        raise ValueError(f"Image size {len(normalized_image)} does not match the number of qubits {num_qubits}.")
    
    # Create a quantum circuit with the required number of qubits
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)  # reps=2 for some entanglement depth
    
    # Bind the normalized image data to the parameters of the circuit
    param_dict = {param: normalized_image[i] for i, param in enumerate(feature_map.parameters)}

    # Assign parameters to the quantum circuit
    quantum_circuit = feature_map.assign_parameters(param_dict)
    
    return quantum_circuit


if __name__ == '__main__':

    datamanager = data.VideoDataManager(
        root='',
        sources='prid2011',
        height=128,  # Image height after resizing
        width=64,  # Image width after resizing
        batch_size_train=16,
        batch_size_test=50,
        seq_len=15,  # Sequence length for videos
        sample_method='evenly',
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']
    )
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    # Use GPU for training if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fully connected neural network (MLP)
    fcn_classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True, random_state=42)

    # Training Loop
    epochs = 1

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Extract quantum features from training set
        X_train_features = []
        y_train = []
        
        # Check what the train_loader returns
        for batch_idx, item in enumerate(train_loader):
            images = item['img']  # Access the 'img' tensor
            labels = item['pid']  # Access the 'pid' (person ID)

            images = images.numpy()  # Convert tensor to numpy array after moving to CPU
            for image in images:
                # Flatten the image and calculate num_qubits based on the flattened size
                flattened_image = image.flatten()
                num_qubits = flattened_image.shape[0]  # Number of qubits = number of pixels in the image
                
                # Apply ZZFeatureMap encoding to the image
                quantum_circuit = zz_feature_map_encoding(flattened_image, num_qubits)
                print(quantum_circuit)
                
                # Use quantum simulator to process the quantum circuit
                backend = Aer.get_backend('qasm_simulator')
                result = backend.run(quantum_circuit).result()
                statevector = result.get_statevector()

                # Convert quantum state to classical feature vector (for simplicity)
                feature_vector = np.real(statevector).flatten()[:num_qubits]
                
                # Append feature and label to training data
                X_train_features.append(feature_vector)
                y_train.append(labels[batch_idx])

                # Clean up variables to free memory
                del image, quantum_circuit, result, statevector, feature_vector

        # Convert to numpy arrays for sklearn
        X_train_features = np.array(X_train_features)
        y_train = np.array(y_train)
        
        # Train the MLP classifier
        fcn_classifier.fit(X_train_features, y_train)
        
        print(f"Epoch {epoch + 1} completed.")
        
        # Optional: Add evaluation on the test set here using similar feature extraction for the test set
