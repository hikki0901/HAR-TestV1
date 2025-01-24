import torch
import train
import dataloader

# Testing
train.model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in dataloader.test_loader:
        images, labels = images.to(train.device), labels.to(train.device)
        outputs = train.model(images)
        _, predicted = outputs.max(1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

with open("train.txt", "a") as f:
            f.write(f"Test Accuracy: {test_correct / test_total:.4f}")