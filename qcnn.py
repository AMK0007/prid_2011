import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml

# Define the quantum circuit using PennyLane
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    print(f"qnode {inputs.shape}")
    
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}

class HybridReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(HybridReIDModel, self).__init__()
        # Pre-trained ResNet50 for feature extraction
        self.backbone = torchreid.models.build_model(
            name='resnet50', 
            num_classes=num_classes,  
            loss='softmax',  
            pretrained=True  
        ).cuda()  # Moving to GPU

        # Quantum layers
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Additional layer to reduce the dimensionality of quantum features
       # self.fc_quantum = torch.nn.Linear(24, 12)  # Project from 24 to 12

        # Fully connected layer for final classification
        self.fc = torch.nn.Linear(12, num_classes)  # Final classification layer

    def forward(self, x):
        # Get features from the backbone
        features = self.backbone(x)
        print(f"backbone {features.shape}")
        # Ensure features fit into quantum layers
        batch_size = features.size(0)
        feature_size = features.size(1)
        
        # Initialize list to hold quantum features
        quantum_features = []
        
        # Process input in chunks of 64 features at a time
        for i in range(0, feature_size, 64):
            # Select chunk of features (max size of 64)
            chunk = features[:, i:i+64]
            
            # Ensure that the chunk has a valid size before processing
            if chunk.size(1) == 0:
                continue  # Skip empty chunks
            
            # Process the chunk through quantum layers
            half_size = chunk.size(1) // 2  # Divide the chunk into two parts
            chunk_1 = chunk[:, :half_size]  # First half of the chunk
            chunk_2 = chunk[:, half_size:]  # Second half of the chunk
            
            # Ensure that both chunks have valid sizes
            if chunk_1.size(1) > 0 and chunk_2.size(1) > 0:
                q1 = self.qlayer1(chunk_1)
                print(f"batch {i} q1 {q1.shape}")
                q2 = self.qlayer2(chunk_2)
                print(f"batch {i} {q2.shape}")
                # Append quantum outputs
                quantum_features.append(torch.cat([q1, q2], dim=1))
        # Concatenate all quantum features
        if len(quantum_features) > 0:
            quantum_features = torch.cat(quantum_features, dim=1)
        # Ensure that quantum features have a valid size before passing to fc_quantum
        print("Quantum features shape before fc_quantum:", quantum_features.shape)
        
        # Flatten or reshape quantum features to match input size (24) for fc_quantum
        # quantum_features = quantum_features.view(batch_size, -1)  # Flatten if needed
        # if quantum_features.size(1) > 24:
        #     quantum_features = quantum_features[:, :24]  # Slice if necessary

        # quantum_features = self.fc_quantum(quantum_features)
        
        # Ensure quantum_features has the expected shape
        quantum_features = quantum_features.view(batch_size, -1)  # Flatten if needed
        # Slice or reshape quantum_features to match fc input size (12 features)
        if quantum_features.size(1) > 12:
            quantum_features = quantum_features[:, :12]

        out = self.fc(quantum_features)
        
        return out

# Wrap the execution code in the main guard
if __name__ == '__main__':
    # Define the transformations using torchvision
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the data manager for PRID2011
    datamanager = torchreid.data.VideoDataManager(
        root='',  
        sources='prid2011',  
        height=256,  
        width=128,  
        batch_size_train=32,  
        batch_size_test=89,  
        seq_len=15,  
        sample_method='evenly',  
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']  
    )
    
    # Access the train and test loaders
    train_loader = datamanager.train_loader  
    test_loader = datamanager.test_loader  

    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    # Number of classes in the dataset (number of unique IDs in the training set)
    num_classes = datamanager.num_train_pids

    # Instantiate the hybrid model
    model = HybridReIDModel(num_classes).cuda()

    # Build the optimizer and scheduler
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam', 
        lr=0.0003  
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',  
        stepsize=20  
    )

    # Create the training engine
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        pooling_method='avg',
        use_gpu=True,
    )

    # Train the model
    engine.run(
        max_epoch=1,  
        save_dir='log/hybrid_resnet505',  
        print_freq=1,  
        test_only=False
    )
