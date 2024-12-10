import torch
import numpy as np
from torchreid import data
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, average_precision_score

import numpy as np
from qiskit import QuantumCircuit

# Angle encoding function
def angle_embedding(image):
    return 2 * np.pi * image.flatten()  # Flatten and map pixel values to angle in [0, 2π]
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Angle encoding function
def angle_encoding(data, num_qubits, circuit=None):
    """
    Encodes classical data into quantum circuit using angle encoding.
    
    Parameters:
    - data: The input classical data (should be normalized or scaled appropriately)
    - num_qubits: Number of qubits in the quantum circuit
    - circuit: A pre-existing quantum circuit to apply encoding to (optional)
    
    Returns:
    - A quantum circuit with the data encoded into it
    """
    # If no circuit is provided, create a new quantum circuit
    if circuit is None:
        circuit = QuantumCircuit(64)
    
    # Normalize or scale your data if needed (here we assume the data is already scaled)
    for i in range(len(data)):
        # Encode each classical feature into a qubit's rotation (RX, RY, or RZ)
        angle = data[i]  # For simplicity, we directly use the data values as angles
        circuit.rx(angle, i)  # Apply RX rotation on qubit i
    
    return circuit

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

    # Training Loop
    epochs = 1
    num_qubits=8
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
                features = angle_encoding(angles, num_qubits)
                print(features)
        #         X_train_features.append(features)
        #     y_train.extend(labels.numpy())  # Extend with batch labels
        
        # # Train the classifier on extracted quantum features
        # fcn_classifier.fit(X_train_features, y_train)
        
        # # Evaluate on the test set
        # X_test_features = []
        # y_test = []
        
        # for batch_idx, item in enumerate(test_loader):
        #     # Ensure images are tensors, move to CPU if using GPU
        #     images = item['img']  # Access the 'img' tensor
        #     labels = item['pid']     # Access the 'pid' (person ID)
        #     images = images.numpy()  # Convert tensor to numpy array after moving to CPU
            
        #     # Preprocess images: Convert images to quantum features
        #     for image in images:
        #         angles = angle_embedding(image)
        #         features = extract_quantum_features(angles, num_qubits)
        #         X_test_features.append(features)
        #     y_test.extend(labels.numpy())
        
        # # Predict on test data
        # y_pred = fcn_classifier.predict(X_test_features)
        
        # # Evaluate accuracy
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Accuracy of the FCN classifier: {accuracy * 100:.2f}%")
        
        # # Rank-1 and Rank-5 accuracy
        # rank1_acc = rank_accuracy(y_test, y_pred, rank=1)
        # rank5_acc = rank_accuracy(y_test, y_pred, rank=5)
        
        # print(f'Rank-1 Accuracy: {rank1_acc * 100:.2f}%')
        # print(f'Rank-5 Accuracy: {rank5_acc * 100:.2f}%')
        
        # # Calculate Mean Average Precision (mAP)
        # y_pred_probs = fcn_classifier.predict_proba(X_test_features)
        # mAP = mean_average_precision(y_test, y_pred_probs)
        # print(f'Mean Average Precision (mAP): {mAP:.2f}')
        
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
