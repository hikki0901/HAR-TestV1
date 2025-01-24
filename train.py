import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import dataloader

# Load a pre-trained ResNet model and modify it for classification
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

class_names = dataloader.train_dataset.classes  # This gets category names as a list
num_classes = len(class_names)  # Get number of classes

model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust output layer to match class count

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        #print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
        #      f"Train Acc: {correct/total:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
        #      f"Val Acc: {val_correct/val_total:.4f}")
        # Open the file in append mode so it does not overwrite previous results
        with open("train.txt", "a") as f:
            f.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
                    f"Train Acc: {correct/total:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, "
                    f"Val Acc: {val_correct/val_total:.4f}\n")


# Train the model
train_model(model, dataloader.train_loader, dataloader.val_loader, epochs=10)


