import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml

# Define the quantum circuit using PennyLane
n_qubits = 8  # Increased qubits for better feature representation
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer
n_layers = 12  # Increased layers for better feature extraction
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Fully Quantum Model for person re-id
class QuantumReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(QuantumReIDModel, self).__init__()
        self.feature_reduction = torch.nn.Linear(98304, n_qubits * 16)  # Adjusted input feature reduction
        self.qlayers = torch.nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(12)])
        self.bn = torch.nn.BatchNorm1d(n_qubits * 12)  # Adjust batch normalization to match output size
        self.dropout = torch.nn.Dropout(0.4)  # Increased dropout for better generalization
        self.fc = torch.nn.Linear(n_qubits * 12, num_classes)  # Adjusted final layer size
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        x = self.feature_reduction(x)  # Reduce input feature size
        
        split_size = x.size(1) // 12
        q_outputs = [layer(x[:, i*split_size:(i+1)*split_size]) for i, layer in enumerate(self.qlayers)]
        
        x = torch.cat(q_outputs, dim=1)
        x = self.bn(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        
        return x

# Execution Code
if __name__ == '__main__':
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    
    num_classes = datamanager.num_train_pids
    print(f"Number of Training IDs (Classes): {num_classes}")  
    
    model = QuantumReIDModel(num_classes).cuda()
    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=20)
    
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        pooling_method='avg',
        use_gpu=True,
    )
    
    engine.run(
        max_epoch=30,
        save_dir='log/quantum_reid',
        print_freq=1,
        test_only=False,
        eval_freq=1
    )