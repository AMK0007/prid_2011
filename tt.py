import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

# Dataset Class for PRID2011
class PRID2011Dataset(Dataset):
    def __init__(self, root_dir, camera, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            camera (str): 'camera_a' or 'camera_b'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.camera = camera
        self.transform = transform
        
        # Load all the persons and images from the specified camera
        self.persons = os.listdir(os.path.join(root_dir, camera))
        self.images = []
        
        for person in self.persons:
            person_dir = os.path.join(root_dir, camera, person)
            person_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
            self.images.extend(person_images)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        
        # Extract person id
        person_id = img_name.split('/')[-2]
        
        return image, person_id

# Define transformations (similar to torchreid)
transform_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare DataLoader for PRID2011
def get_dataloader(root_dir, batch_size=32):
    # Create datasets for camera_a (query) and camera_b (gallery)
    query_dataset = PRID2011Dataset(root_dir, 'cam_a/', transform=transform_pipeline)
    gallery_dataset = PRID2011Dataset(root_dir, 'cam_b/', transform=transform_pipeline)
    
    # Create DataLoader for query and gallery
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return query_loader, gallery_loader

# Training Loop
def train(model, query_loader, gallery_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over gallery images (train on gallery data)
        for inputs, labels in tqdm(gallery_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs = inputs.cuda()
            labels = torch.tensor([int(label.split('/')[-2]) for label in labels]).cuda()  # Convert to tensor
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(gallery_loader)}")
    
    print("Training Complete")

# Function to evaluate the model on query data
def evaluate(model, query_loader, gallery_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(query_loader, desc="Evaluating", unit="batch"):
            inputs = inputs.cuda()
            labels = torch.tensor([int(label.split('/')[-2]) for label in labels]).cuda()  # Convert to tensor
            
            # Get predictions for the query
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate correct predictions
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    # Convert to numpy arrays for evaluation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = 100 * correct_preds / total_preds
    
    # Print evaluation results
    print("Accuracy: {:.2f}%".format(accuracy))
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Main function
def main():
    # Load the PRID2011 dataset (replace with the correct path)
    root_dir = 'G:/K214502/FYP/prid_2011/prid2011/prid_2011/multi_shot/'

    # Get query and gallery DataLoader
    query_loader, gallery_loader = get_dataloader(root_dir)

    # Load Pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(os.listdir(os.path.join(root_dir, 'cam_b'))))  # num_classes based on gallery
    model = model.cuda()  # Move model to GPU

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Train the model for 5 epochs
    train(model, query_loader, gallery_loader, criterion, optimizer, num_epochs=5)

    # Evaluate the model on query data
    evaluate(model, query_loader, gallery_loader)

if __name__ == '__main__':
    main()
