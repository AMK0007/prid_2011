import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms
import pennylane as qml

# ==== PARAMETERS ====
n_qubits = 6
n_layers =16

# Define the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit using PennyLane
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.0, normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Set weight shapes using generic parameters
weight_shapes = {"weights": (n_layers, n_qubits)}

# ==== Hybrid Model Definition ====
class HybridReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(HybridReIDModel, self).__init__()
        self.backbone = torchreid.models.build_model(
            name='resnet50',
            num_classes=num_classes,
            loss='softmax',
            pretrained=True,
        ).cuda()

        # Classical head projecting into n_layers * n_qubits
        self.backbonefc = torch.nn.Linear(2048, n_layers * (2**n_qubits))

        # Create quantum layers dynamically
        self.qlayers = torch.nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_layers)
        ])

        # Final classification layer
        self.fc = torch.nn.Linear(n_layers * n_qubits, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.backbonefc(features)

        # Split into chunks for each quantum layer
        chunks = torch.chunk(features, n_layers, dim=1)
        quantum_outputs = []

        for i in range(n_layers):
            q_out = self.qlayers[i](chunks[i])
            quantum_outputs.append(q_out)

        # Concatenate outputs from all quantum layers
        x = torch.cat(quantum_outputs, dim=1)
        x = self.fc(x)
        return x

# ==== Main Training Code ====
if __name__ == '__main__':

    datamanager = torchreid.data.VideoDataManager(
        root='',
        sources='prid2011',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=64,
        seq_len=15,
        sample_method='evenly',
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']
    )

    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader
    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    num_classes = datamanager.num_train_pids
    model = HybridReIDModel(num_classes).cuda()

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
        save_dir='log/hybrid_resnet50_dynamic_layers16',
        print_freq=1,
        test_only=True,
        eval_freq=1
    )
