import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml

# Define the quantum circuit using PennyLane
n_qubits = 6
n_layers = 12  # Increased quantum layers depth
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define weight shapes for quantum layers
weight_shapes = {"weights": (n_layers, n_qubits)}

# Define the hybrid model with 4 quantum layers
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
        self.backbonefc = torch.nn.Linear(2048, 128)

        # Increased quantum layers (from 2 to 4)
        self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Fully connected layer for classification
        self.fc = torch.nn.Linear(24, num_classes)  # 4 * 6 = 24 quantum outputs

    def forward(self, x):
        features = self.backbone(x)
        features = self.backbonefc(features)

        # Splitting features for quantum processing (4-way split)
        quarter_size = features.size(1) // 4
        x_1 = features[:, :quarter_size]
        x_2 = features[:, quarter_size:2*quarter_size]
        x_3 = features[:, 2*quarter_size:3*quarter_size]
        x_4 = features[:, 3*quarter_size:]

        # Passing through four quantum layers
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)

        # Concatenating quantum outputs and classifying
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x = self.fc(x)
        return x

# Wrap execution in the main guard
if __name__ == '__main__':
    # Define the transformations using torchvision
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the data manager for PRID2011 dataset
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

    # Access train and test loaders
    train_loader = datamanager.train_loader  
    test_loader = datamanager.test_loader  
    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    # Get number of classes
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
        save_dir='log/hybrid_resnet50_q4layers',  
        print_freq=1,  
        test_only=False,
        eval_freq=1
    )
