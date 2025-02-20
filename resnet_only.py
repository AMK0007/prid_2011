import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms

# Define the Hybrid Model but only use ResNet50
class ResNetReIDModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNetReIDModel, self).__init__()
        # Load Pre-trained ResNet50 from TorchReID
        self.backbone = torchreid.models.build_model(
            name='resnet50', 
            num_classes=num_classes,  
            loss='softmax',  
            pretrained=True,
        ).cuda()  # Move to GPU

    def forward(self, x):
        features = self.backbone(x)
        return features

# Main Execution
if __name__ == '__main__':
    # PRID2011 Data Manager
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

    # Get Data Loaders
    train_loader = datamanager.train_loader  
    test_loader = datamanager.test_loader  
    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']

    # Number of unique IDs in dataset
    num_classes = datamanager.num_train_pids

    # Instantiate Model (ResNet50 Only)
    model = ResNetReIDModel(num_classes).cuda()

    # Optimizer and Scheduler
    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=0.0003)
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=20)

    # Training Engine (ResNet50 Only)
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager, model, optimizer, scheduler=scheduler, pooling_method='avg', use_gpu=True
    )

    # Train ResNet50
    engine.run(
        max_epoch=30,  
        save_dir='log/resnet50',  
        print_freq=1,  
        test_only=False,
        eval_freq=1
    )

