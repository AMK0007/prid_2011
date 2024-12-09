import torch
import numpy as np
from PIL import Image
import os
from torchreid import data
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader

# Use the DataManager to handle preprocessing
def load_data_with_datamanager(dataset_path):
    # Define the DataManager for PRID2011
    datamanager = data.VideoDataManager(
        root=dataset_path,
        sources='prid2011',
        height=256,  # Image height after resize
        width=128,  # Image width after resize
        batch_size_train=32,
        batch_size_test=100,
        seq_len=15,  # Sequence length for videos
        sample_method='evenly',
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']
    )
    return datamanager

# Define the quantum angle embedding
def angle_embedding(image):
    return 2 * np.pi * image.flatten()  # Flatten and map pixel values to angle in [0, 2π]

# Quantum Convolution Layer
def quantum_convolution_layer(num_qubits, angles, prev_circuit=None):
    conv_circuit = QuantumCircuit(num_qubits)
    
    if prev_circuit:
        conv_circuit.append(prev_circuit, range(num_qubits))  # Include previous layer operations
    
    for i, angle in enumerate(angles):
        conv_circuit.rz(angle, i)  # Apply rotation gate (Rz)
    
    for i in range(num_qubits - 1):
        conv_circuit.cx(i, i + 1)
    
    return conv_circuit

# Quantum Pooling Layer
def quantum_pooling_layer(num_qubits, conv_circuit):
    pooling_circuit = conv_circuit.copy()
    pooling_circuit.measure_all()  # Measure all qubits to collapse the quantum state
    return pooling_circuit

# Create the Quantum CNN
def create_qcnn(num_qubits, angles):
    conv1 = quantum_convolution_layer(num_qubits, angles)  # First convolution layer
    pooled1 = quantum_pooling_layer(num_qubits, conv1)  # Pooling layer 1
    
    conv2 = quantum_convolution_layer(num_qubits, angles, pooled1)  # Second convolution layer
    pooled2 = quantum_pooling_layer(num_qubits, conv2)  # Pooling layer 2
    
    return pooled2  # Return the final quantum state

# Function to extract quantum features using the QCNN
def extract_quantum_features(angles, num_qubits=8192):
    circuit = create_qcnn(num_qubits, angles)
    
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    
    result = backend.run(transpiled_circuit, shots=1024).result()
    counts = result.get_counts()
    
    classical_features = np.array([list(counts.values())[0]])  # Simplified feature extraction
    return classical_features

# Training loop with multiple epochs
def train_qcnn_with_epochs(datamanager, epochs=1, batch_size=32, lr=0.001, num_qubits=8):
    # Set up data loaders
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader
    
    # Use GPU for training if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fully connected neural network (MLP)
    fcn_classifier = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1, warm_start=True, random_state=42)

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Extract quantum features from training set
        X_train_features = []
        y_train = []
        
        # Check what the train_loader returns
        for batch_idx, item in enumerate(train_loader):
            images = item['img']  # Access the 'img' tensor
            labels = item['pid']     # Access the 'pid' (person ID)

            images = images.numpy()  # Convert tensor to numpy array after moving to CPU
            
            # Preprocess images: Convert images to quantum features
            for image in images:
                angles = angle_embedding(image)
                features = extract_quantum_features(angles, num_qubits)
                X_train_features.append(features)
            y_train.extend(labels.numpy())  # Extend with batch labels
        
        # Train the classifier on extracted quantum features
        fcn_classifier.fit(X_train_features, y_train)
        
        # Evaluate on the test set
        X_test_features = []
        y_test = []
        
        for batch_idx, item in enumerate(test_loader):
            # Ensure images are tensors, move to CPU if using GPU
            images = item['img']  # Access the 'img' tensor
            labels = item['pid']     # Access the 'pid' (person ID)
            images = images.numpy()  # Convert tensor to numpy array after moving to CPU
            
            # Preprocess images: Convert images to quantum features
            for image in images:
                angles = angle_embedding(image)
                features = extract_quantum_features(angles, num_qubits)
                X_test_features.append(features)
            y_test.extend(labels.numpy())
        
        # Predict on test data
        y_pred = fcn_classifier.predict(X_test_features)
        
        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of the FCN classifier: {accuracy * 100:.2f}%")
        
        # Rank-1 and Rank-5 accuracy
        rank1_acc = rank_accuracy(y_test, y_pred, rank=1)
        rank5_acc = rank_accuracy(y_test, y_pred, rank=5)
        
        print(f'Rank-1 Accuracy: {rank1_acc * 100:.2f}%')
        print(f'Rank-5 Accuracy: {rank5_acc * 100:.2f}%')
        
        # Calculate Mean Average Precision (mAP)
        y_pred_probs = fcn_classifier.predict_proba(X_test_features)
        mAP = mean_average_precision(y_test, y_pred_probs)
        print(f'Mean Average Precision (mAP): {mAP:.2f}')
        
# Function to calculate Rank-1 and Rank-5 accuracies
def rank_accuracy(true_ids, predicted_ids, rank=1):
    correct = 0
    for true_id, pred_ids in zip(true_ids, predicted_ids):
        if true_id in pred_ids[:rank]:
            correct += 1
    return correct / len(true_ids)

# Function to calculate Mean Average Precision (mAP)
def mean_average_precision(true_ids, predicted_scores):
    return np.mean([average_precision_score(true_ids == id, scores) for id, scores in predicted_scores.items()])

# Initialize DataManager
if __name__ == '__main__':
    dataset_path = ''  # Modify with the actual path
    datamanager = load_data_with_datamanager(dataset_path)
    # Train the model for multiple epochs
    train_qcnn_with_epochs(datamanager, epochs=1)
