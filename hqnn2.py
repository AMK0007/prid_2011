import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml

# Define the quantum circuit using PennyLane
n_qubits = 8
n_layers = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the number of quantum layers dynamically
weight_shapes = {"weights": (n_layers, n_qubits)}

class HybridReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(HybridReIDModel, self).__init__()

        # Pre-trained ResNet50 for feature extraction
        self.backbone = torchreid.models.build_model(
            name='resnet50', 
            num_classes=num_classes,  
            loss='softmax',  
            pretrained=True,
        ).cuda()

        #self.backbonefc = torch.nn.Linear(2048, 256)  # Reduce feature size

        # Quantum layers dynamically stored in ModuleList
        self.qlayers = torch.nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_layers)
        ])

        # Fully connected layer for final classification
        self.fc = torch.nn.Linear(n_layers * n_qubits, num_classes)  # Output adjusted for n_layers

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        #features = self.backbonefc(features)

        # Split feature vector dynamically into `n_layers` parts
        part_size = features.size(1) // n_layers  # Divide features into equal parts
        quantum_outputs = []

        for i in range(n_layers):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < n_layers - 1 else None  # Ensure last partition includes remaining features
            x_part = features[:, start_idx:end_idx]  # Extract segment
            quantum_outputs.append(self.qlayers[i](x_part))  # Apply corresponding quantum layer

        # Concatenate quantum outputs
        x = torch.cat(quantum_outputs, dim=1)

        # Fully connected output
        x = self.fc(x)
        return x

# Wrap execution code in main guard
if __name__ == '__main__':
    # Define transformations
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create data manager for PRID2011
    datamanager = torchreid.data.VideoDataManager(
        root='',  
        sources='prid2011',  
        height=256,  
        width=128,  
        batch_size_train=8,  
        batch_size_test=32,  
        seq_len=15,  
        sample_method='evenly',  
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']  
    )

    # Access train and test loaders
    datamanager.train_loader.num_workers = 0  # Ensure compatibility with Windows
    train_loader = datamanager.train_loader  
    test_loader = datamanager.test_loader  

    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    # Get number of unique identities (classes)
    num_classes = datamanager.num_train_pids

    # Instantiate hybrid model
    model = HybridReIDModel(num_classes).cuda()

    # Build optimizer and scheduler
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

    # Create training engine
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
        max_epoch=30,  
        save_dir='log/hybrid_resnet50_nlayers4',  
        print_freq=1,  
        test_only=False,
        eval_freq=1
    )
