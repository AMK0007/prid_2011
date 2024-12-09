import torch
import torch.nn as nn
import pennylane as qml
from torchvision import transforms
import torchreid
from torchreid.reid.optim import build_optimizer, build_lr_scheduler

# Quantum Device Setup (using default.qubit for simulation)
dev = qml.device("default.qubit", wires=4)  # Use 4 qubits for the quantum part

# Define the Quantum Convolutional Layer
class QuantumConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(QuantumConvolutionalLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Debugging the shape of input tensor
        print(f"Input shape before quantum layer: {x.shape}")

        # Check if the input tensor has 4 dimensions (batch_size, channels, height, width)
        if len(x.shape) != 4:
            raise RuntimeError(f"Input tensor must be 4D (batch_size, channels, height, width), got {x.shape}")

        # Define the quantum circuit for convolution
        @qml.qnode(dev)
        def quantum_conv(x):
            # Ensure x is a 2D array with the proper shape
            batch_size, num_channels, height, width = x.shape
            
            # Limit the number of channels to 4 qubits (wires)
            num_channels = min(num_channels, 4)
            
            features = []  # Collect quantum features
            for i in range(num_channels):
                # Access the input for the i-th channel at the 0th batch index
                channel_data = x[0, i]  # This selects the first batch element for channel i
                
                # Check if channel_data is a scalar or not
                if channel_data.ndim == 0:  # Already a scalar
                    scalar_input = channel_data.item()  # Convert to scalar
                else:
                    # If it's a tensor, flatten or index to get a single scalar
                    scalar_input = channel_data.flatten()[0].item()  # Flatten and take the first element

                # Apply the RX gate to the corresponding qubit (wire)
                qml.RX(scalar_input, wires=i)
                
                # Apply the RY gate to the corresponding qubit (wire)
                qml.RY(scalar_input, wires=i)

            # Perform measurement (expectation value of PauliZ)
            return [qml.expval(qml.PauliZ(i)) for i in range(num_channels)]  # Measurement after gates

        # Apply the quantum convolutional layer on the input tensor
        quantum_features = torch.tensor(quantum_conv(x.detach().cpu().numpy()))

        # Debugging the shape of output tensor
        print(f"Output shape after quantum layer: {quantum_features.shape}")
        
        # Convert quantum features back to the appropriate shape for further processing
        # Check the shape and adjust the reshape size accordingly
        batch_size = x.shape[0]
        
        # Print the size of the tensor before reshaping
        print(f"Size before reshaping: {quantum_features.size()}")

        # Adjust reshape for the smaller size (4 features)
        quantum_features = quantum_features.view(batch_size, self.out_channels, 1, 4)  # Example reshape for 4 features

        # Print the shape after reshaping
        print(f"Shape after reshaping: {quantum_features.shape}")

        return quantum_features.to(x.device)


# Quantum Pooling Layer - Apply classical pooling here as a placeholder
class QuantumPoolingLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(QuantumPoolingLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Debugging the shape of input tensor before pooling
        print(f"Input shape before pooling: {x.shape}")
        
        if len(x.shape) != 4:
            raise RuntimeError(f"Input tensor must be 4D (batch_size, channels, height, width), got {x.shape}")
        
        return self.pool(x)


# Define the Quanvolutional Neural Network (QNN) with PennyLane integration
class QuanvolutionalNet(nn.Module):
    def __init__(self, num_classes):
        super(QuanvolutionalNet, self).__init__()
        
        # Encoding the feature map (classical part)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.quantum_conv1 = QuantumConvolutionalLayer(64, 128)
        self.pool2 = QuantumPoolingLayer(kernel_size=2, stride=2)

        self.fc = nn.Linear(128 * 1 * 4, num_classes)  # Adjust based on reshape size

    def forward(self, x):
        # Classical processing
        x = self.conv1(x)
        x = self.pool1(x)

        # Quantum processing
        x = self.quantum_conv1(x)
        x = self.pool2(x)

        # Flatten the output and apply fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Main guard for running the code
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

    # Use the datamanager's dataset to get the length of the datasets
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    # Split query and gallery datasets
    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    # Print the lengths for train, query, and gallery datasets
    print(f"Number of training tracklets: {len(train_loader.dataset)}")
    print(f"Number of query tracklets: {len(query_loader.dataset)}")
    print(f"Number of gallery tracklets: {len(gallery_loader.dataset)}")

    # Build the Quanvolutional model
    model = QuanvolutionalNet(num_classes=3).cuda()

    # Build the optimizer
    optimizer = build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    # Build the learning rate scheduler
    scheduler = build_lr_scheduler(
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
        save_dir='log/quanvolutional_net',
        print_freq=5,
        test_only=False
    )
