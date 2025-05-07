import torch
import torchreid
from torchvision import transforms
import pennylane as qml

# Define the quantum circuit using PennyLane
n_qubits = 5
n_layers = 64
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
            pretrained=True,  # Using pre-trained weights
        ).cuda()

        # Freeze all layers except the final few (fine-tune deeper layers)
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze all layers initially
        
        # Unfreeze the last few layers (for example, the last block of ResNet50)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True  # Unfreeze the deeper layers

        # Quantum layers
        self.qlayers = torch.nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_layers)
        ])

        # Fully connected layer for final classification
        self.fc = torch.nn.Linear(n_layers * n_qubits, num_classes)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # Now, this will fine-tune based on the "requires_grad"
        
        # Quantum layers and fully connected output
        quantum_outputs = []
        part_size = features.size(1) // n_layers
        for i in range(n_layers):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < n_layers - 1 else None
            x_part = features[:, start_idx:end_idx]
            quantum_outputs.append(self.qlayers[i](x_part))

        x = torch.cat(quantum_outputs, dim=1)
        x = self.fc(x)
        return x

# Wrap execution code in main guard
if __name__ == '__main__':
    # Define transformations
    transform_pipeline = [
        'random_flip',       # Random horizontal flip
        'random_rotate',     # Random rotation
        'random_crop',       # Random resized crop
        'color_jitter',      # Color jitter
        'normalize'          # Normalize (mean, std)
    ]
    # Create data manager for PRID2011
    datamanager = torchreid.data.VideoDataManager(
        root='',  
        sources='prid2011',  
        height=256,  
        width=128,  
        batch_size_train=8,  
        batch_size_test=64,  
        seq_len=6,  
        sample_method='random',  
        transforms=transform_pipeline ,
        num_instances=4,
        workers=8
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
# Build optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,  # Smaller learning rate for fine-tuning
        weight_decay=5e-4  # L2 regularization to prevent overfitting
    )

    # Scheduler for adjusting the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # Gradual learning rate reduction

    # Create training engine
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        pooling_method='avg',
        use_gpu=True
    )
    engine.run(
        max_epoch=30,
        save_dir='log/hybrid_resnet50_dynamic_layers7-16(3)',
        print_freq=1,
        test_only=False,
        eval_freq=1
    )

