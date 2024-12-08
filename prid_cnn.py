import torch
import torchreid
import torchreid.reid
import torchreid.reid.data.datasets
import torchreid.reid.data.datasets.video
from torchvision import transforms

# Wrap the execution code in the main guard
if __name__ == '__main__':
    # Define the transformations using torchvision
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random flip for augmentation
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # Random crop and resize
        transforms.Resize((256, 128)),  # Resize the images to 256x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    ])

    # Create the data manager for PRID2011
    datamanager = torchreid.data.VideoDataManager(
        root='',  # Root directory for the dataset
        sources='prid2011',  # Dataset name
        height=256,  # Image height after resize
        width=128,  # Image width after resize
        batch_size_train=32,  # Batch size for training
        batch_size_test=100,  # Batch size for testing
        seq_len=15,  # Sequence length (for video datasets)
        sample_method='evenly',  # Sampling method for sequence
        transforms=['random_flip', 'random_crop', 'resize', 'normalize']  # Transformations to apply
    )

    # Use the datamanager's dataset to get the length of the datasets
    train_loader = datamanager.train_loader  # Access train loader directly
    test_loader = datamanager.test_loader  # Access test loader directly

    # Print the lengths for train, query, and gallery datasets
    print(f"Number of training tracklets: {len(train_loader.dataset)}")
    print(test_loader)

    # Since 'test_loader' contains both query and gallery data, we can split them manually
    query_loader = test_loader['prid2011']['query']
    gallery_loader = test_loader['prid2011']['gallery']


    print(f"Number of query tracklets: {len(query_loader.dataset)}")
    print(f"Number of gallery tracklets: {len(gallery_loader.dataset)}")

    # Build the model
    model = torchreid.models.build_model(
        name='resnet50',  # Model name
        num_classes=datamanager.num_train_pids,  # Number of classes (people)
        loss='softmax',  # Loss function
        pretrained=True  # Use pretrained weights
    ).cuda()  # Move the model to GPU

    # Build the optimizer
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',  # Optimizer type
        lr=0.0003  # Learning rate
    )

    # Build the learning rate scheduler
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',  # Type of scheduler
        stepsize=20  # Stepsize for the scheduler
    )

    # Create the training engine
    engine = torchreid.engine.VideoSoftmaxEngine(
        datamanager,
        model,
        optimizer,
        scheduler=scheduler,
        pooling_method='avg',  # Pooling method
        use_gpu=True ,
    )

    # Train the model
    engine.run(
        max_epoch=25,  # Number of training epochs
        save_dir='log/resnet50',  # Directory to save logs and model
        print_freq=5,  # Frequency to print training logs
        test_only=False  # Set to True if only testing
    )
