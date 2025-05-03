# bodypart classifier based on ResNet50

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import models, transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping (21 original classes → 8 grouped classes)
label_mapping = {
    3: 0,
    0: 1,
    2: 2,
    13: 2,
    20: 2,
    4: 3,
    5: 3,
    7: 3,
    8: 3,
    9: 3,
    16: 3,
    21: 3,
    1: 4,
    6: 4,
    11: 4,
    12: 4,
    19: 4,
    10: 5,
    15: 5,
    17: 6,
    18: 6,
    14: 7,
}

# Load CSV and image paths
csv_path = "train_data.csv"
image_dir = "bodyparts_classification/train"
data = pd.read_csv(csv_path)
image_paths = [
    os.path.join(image_dir, f"{uid}-c.png") for uid in data["SOPInstanceUID"]
]
labels = [label_mapping[label] for label in data["Target"]]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Image transformations
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Create datasets and loaders
train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
val_dataset = CustomDataset(X_val, y_val, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Build the model
def build_resnet50(num_classes=8):
    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load("resnet50-0676ba61.pth"))
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )
    return model


model = build_resnet50()
model.to(device)
summary(model, input_size=(64, 3, 224, 224))

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
history = {key: [] for key in ["loss", "val_loss", "accuracy", "val_accuracy"]}

for epoch in range(30):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    history["loss"].append(running_loss / len(train_loader))
    history["accuracy"].append(acc)
    print(
        f"Epoch [{epoch + 1}], Train Loss: {history['loss'][-1]:.4f}, Accuracy: {acc:.2f}%"
    )

    # Validation
    model.eval()
    val_loss, correct, total, y_true, y_pred, y_probs = 0.0, 0, 0, [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(1)
            probs = F.softmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    acc = 100 * correct / total
    history["val_loss"].append(val_loss / len(val_loader))
    history["val_accuracy"].append(acc)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
    print(
        f"Validation Loss: {history['val_loss'][-1]:.4f}, Accuracy: {acc:.2f}%, F1: {f1:.2f}, AUC: {auc:.3f}"
    )

    scheduler.step()
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

# Plot accuracy and loss
plt.figure()
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.savefig("accuracy.png")

plt.figure()
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig("loss.png")

# Evaluate on test set
test_csv = "test_data.csv"
test_img_dir = "bodyparts_classification/test"
test_df = pd.read_csv(test_csv)
test_image_paths = [
    os.path.join(test_img_dir, f"{uid}-c.png") for uid in test_df["SOPInstanceUID"]
]
test_labels = [label_mapping[label] for label in test_df["Target"]]
test_dataset = CustomDataset(test_image_paths, test_labels, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_loss, correct, total, y_true, y_pred, y_probs = 0.0, 0, 0, [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        preds = outputs.argmax(1)
        probs = F.softmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

acc = 100 * correct / total
f1 = f1_score(y_true, y_pred, average="weighted")
auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
print(
    f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {acc:.2f}%, F1: {f1:.2f}, AUC: {auc:.3f}"
)

# Confusion matrix
cm_test = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.savefig("confusion_matrix_test.png")
plt.show()
