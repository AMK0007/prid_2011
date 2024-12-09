import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pennylane as qml
from sklearn.metrics.pairwise import cosine_similarity

# Quantum Amplitude Encoding with PennyLane
class AmplitudeEncoder(torch.nn.Module):
    def __init__(self, input_size=64*128*3, num_qubits=None):
        super(AmplitudeEncoder, self).__init__()
        self.input_size = input_size
        self.num_qubits = num_qubits if num_qubits else int(np.ceil(np.log2(self.input_size)))

        # Define quantum device
        self.device = qml.device("default.qubit", wires=self.num_qubits)

    def forward(self, x):
        batch_size = x.size(0)

        # Normalize the image data to [0, 1] before encoding into quantum states
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)  # Added small epsilon to avoid division by zero

        # Flatten the image to convert the pixels into a vector for encoding
        x = x.view(batch_size, -1)

        quantum_states = []

        for i in range(batch_size):
            image_vector = x[i].cpu().detach().numpy()

            # Map image values to amplitudes
            amplitudes = np.sqrt(image_vector / np.sum(image_vector)) if np.sum(image_vector) != 0 else np.zeros_like(image_vector)

            # Quantum encoding using PennyLane
            @qml.qnode(self.device)
            def encode_amplitude():
                for j in range(self.num_qubits):
                    qml.RY(amplitudes[j], wires=j)
                return [qml.expval(qml.PauliZ(wires=j)) for j in range(self.num_qubits)]

            # Execute the quantum circuit
            quantum_state = encode_amplitude()
            quantum_states.append(quantum_state)

        return torch.tensor(quantum_states).float().cuda()  # Return tensor

class QuantumConvLayer(torch.nn.Module):
    def __init__(self, num_qubits, device):
        super(QuantumConvLayer, self).__init__()
        self.num_qubits = num_qubits
        self.device = device

    def forward(self, x):
        @qml.qnode(self.device)
        def quantum_convolution():
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.num_qubits)]

        feature_maps = torch.stack([torch.tensor(quantum_convolution()) for _ in x])
        return feature_maps

class QuantumPoolingLayer(torch.nn.Module):
    def __init__(self, num_qubits):
        super(QuantumPoolingLayer, self).__init__()
        self.num_qubits = num_qubits

    def forward(self, x):
        return [torch.mean(feature_map) for feature_map in x]  # Average pooling

class QuantumReIDModel(torch.nn.Module):
    def __init__(self, num_qubits, num_classes):
        super(QuantumReIDModel, self).__init__()
        self.encoder = AmplitudeEncoder(input_size=64*128*3, num_qubits=num_qubits).cuda()
        self.qconv1 = QuantumConvLayer(num_qubits, device="default.qubit").cuda()
        self.qpool1 = QuantumPoolingLayer(num_qubits).cuda()
        self.qconv2 = QuantumConvLayer(num_qubits, device="default.qubit").cuda()
        self.qpool2 = QuantumPoolingLayer(num_qubits).cuda()
        self.fc = torch.nn.Linear(num_qubits, num_classes).cuda()

    def forward(self, x):
        quantum_states = self.encoder(x)
        x = self.qconv1(quantum_states)
        x = self.qpool1(x)
        x = self.qconv2(x)
        x = self.qpool2(x)

        x = torch.tensor(x).float().unsqueeze(0).cuda()  # Ensure tensor shape is correct
        x = self.fc(x.view(x.size(0), -1))  # Fully connected layer for classification
        return x

# Image Preprocessing and Loading
def load_and_preprocess_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

# Define transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Folder paths for cam_a and cam_b
cam_a_path = "E:/Mustafa UNI/github/prid_2011/prid2011/prid_2011/single_shot/cam_a"
cam_b_path = "E:/Mustafa UNI/github/prid_2011/prid2011/prid_2011/single_shot/cam_b"

# Initialize the Quantum ReID Model
model = QuantumReIDModel(num_qubits=8, num_classes=200).cuda()

def compute_rank_accuracy():
    rank_1_accuracy = 0
    rank_5_accuracy = 0
    total_queries = len(query_images)
    
    for query_image in query_images:
        query_img_path = os.path.join(cam_b_path, query_image)
        query_image_tensor = load_and_preprocess_image(query_img_path, transform=transform_pipeline).unsqueeze(0).cuda()
        
        query_features = model(query_image_tensor).cpu().detach().numpy()
        
        similarities = []
        for gallery_image in gallery_images:
            gallery_img_path = os.path.join(cam_a_path, gallery_image)
            gallery_image_tensor = load_and_preprocess_image(gallery_img_path, transform=transform_pipeline).unsqueeze(0).cuda()

            gallery_features = model(gallery_image_tensor).cpu().detach().numpy()
            similarity = cosine_similarity(query_features, gallery_features)
            similarities.append((gallery_image, similarity[0][0]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        correct_gallery_image = query_image.split('_')[0]

        if similarities[0][0].split('_')[0] == correct_gallery_image:
            rank_1_accuracy += 1
        
        for i in range(5):
            if similarities[i][0].split('_')[0] == correct_gallery_image:
                rank_5_accuracy += 1
                break
    
    rank_1_accuracy /= total_queries
    rank_5_accuracy /= total_queries
    
    return rank_1_accuracy, rank_5_accuracy

def main():
    # Load gallery and query images
    global gallery_images, query_images
    gallery_images = sorted(os.listdir(cam_a_path))  # Gallery images (cam_a)
    query_images = sorted(os.listdir(cam_b_path))    # Query images (cam_b)

    rank_1_accuracy, rank_5_accuracy = compute_rank_accuracy()
    print(f"Rank-1 Accuracy: {rank_1_accuracy:.2f}")
    print(f"Rank-5 Accuracy: {rank_5_accuracy:.2f}")

if __name__ == "__main__":
    main()