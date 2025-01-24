import torch
import train
import dataloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Testing
train.model.eval()
test_correct, test_total = 0, 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in dataloader.test_loader:
        images, labels = images.to(train.device), labels.to(train.device)
        outputs = train.model(images)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train.class_names, yticklabels=train.class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# Save the confusion matrix as a PNG file
plt.savefig("Confusion_Matrix.png")

# Save accuracy results
with open("train.txt", "a") as f:
    f.write(f"Test Accuracy: {test_correct / test_total:.4f}\n")