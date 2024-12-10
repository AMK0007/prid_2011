import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml
from pennylane import numpy as np

# Define the quantum circuit using PennyLane
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}

# Hybrid Model: Using quantum layers with a ResNet-based model for person re-id
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

        # Fully connected layer
        self.fc = torch.nn.Linear(2048 + 5 * 2, num_classes)  # ResNet feature + quantum features

    def forward(self, x):
        # Get features from the backbone
        features = self.backbone(x)

        # Split features and pass through quantum layers
        x_1, x_2 = torch.split(features, features.size(1)//2, dim=1)  # Split the features for the quantum layers
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        
        # Concatenate quantum outputs with classical features
        x = torch.cat([x_1, x_2], dim=1)
        
        # Final classification layer
        x = self.fc(x)
        return x


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
        batch_size_test=100,  
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
        max_epoch=25,  
        save_dir='log/hybrid_resnet50',  
        print_freq=5,  
        test_only=False  
    )
