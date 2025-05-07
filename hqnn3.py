import torch
import torchreid
import pennylane as qml

# Quantum setup
n_qubits = 6
n_layers = 32
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')  # Changed from Amplitude to AngleEmbedding
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits)}

class HybridReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(HybridReIDModel, self).__init__()

        self.backbone = torchreid.models.build_model(
            name='resnet50',
            num_classes=num_classes,
            loss='softmax',
            pretrained=True,
        ).cuda()

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        self.features_dim = 2048  # ResNet-50 final feature dim
        self.part_dim = self.features_dim // n_layers

        # Projector to match quantum input size
        self.projector = torch.nn.Linear(self.part_dim, n_qubits)

        self.qlayers = torch.nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(n_layers)
        ])

        self.fc = torch.nn.Linear(n_layers * n_qubits, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        quantum_outputs = []

        for i in range(n_layers):
            start_idx = i * self.part_dim
            end_idx = (i + 1) * self.part_dim if i < n_layers - 1 else None
            x_part = features[:, start_idx:end_idx]
            x_part = self.projector(x_part)  # Reduce to n_qubits
            quantum_outputs.append(self.qlayers[i](x_part))

        x = torch.cat(quantum_outputs, dim=1)
        x = self.fc(x)
        return x

# Main logic
if __name__ == '__main__':
    transform_pipeline = [
        'random_flip',
        'random_rotate',
        'random_crop',
        'color_jitter',
        'normalize'
    ]

    datamanager = torchreid.data.VideoDataManager(
        root='',
        sources='prid2011',
        height=256,
        width=128,
        batch_size_train=8,
        batch_size_test=64,
        seq_len=6,
        sample_method='random',
        transforms=transform_pipeline,
        num_instances=4,
        workers=8
    )

    datamanager.train_loader.num_workers = 0
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    num_classes = datamanager.num_train_pids

    model = HybridReIDModel(num_classes).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

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
        save_dir='log/hybrid_resnet50_qml_angleembed_fixed',
        print_freq=1,
        test_only=False,
        eval_freq=1
    )
