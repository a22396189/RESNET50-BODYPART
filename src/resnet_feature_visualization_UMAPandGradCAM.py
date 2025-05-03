# ResNet Feature Visualization with UMAP and Grad-CAM
# Using after training a ResNet model for classification

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import umap
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set device to CPU for visualization
DEVICE = torch.device("cpu")


# Load pre-trained model checkpoint
def load_model(path, num_classes=8):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# Map 21 classes to 8 grouped categories
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


# Load dataset from CSV and image directory
def get_dataset(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    image_paths = [
        os.path.join(img_dir, f"{uid}-c.png") for uid in df["SOPInstanceUID"]
    ]
    labels = [label_mapping[label] for label in df["Target"]]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class CustomDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths, self.labels, self.transform = paths, labels, transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx], self.paths[idx]

    return CustomDataset(image_paths, labels, transform)


# Extract features from specific layer
@torch.no_grad()
def extract_features(model, layer_name, data_loader):
    features, labels = [], []
    activation = {}

    # Hook to capture output from the specified layer
    def hook_fn(module, input, output):
        activation[layer_name] = output

    # Register hook
    hook = dict(model.named_modules())[layer_name].register_forward_hook(hook_fn)

    for imgs, lbls, _ in data_loader:
        imgs = imgs.to(DEVICE)
        model(imgs)  # Trigger the forward hook

        feat = activation[layer_name]

        # Handle different feature shapes
        if feat.dim() == 4:  # shape: [B, C, H, W] -> apply global avg pooling
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
        elif feat.dim() == 2:  # shape: [B, F] - already flat
            feat = feat
        else:
            raise ValueError(f"Unexpected feature shape: {feat.shape}")

        features.append(feat.cpu())
        labels.append(lbls)

    hook.remove()

    return torch.cat(features), torch.cat(labels)


# UMAP projection
def plot_umap(features, labels, title):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(features.numpy())
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Class {i}", alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.show()


# t-SNE projection
def plot_tsne(features, labels, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(features.numpy())
    plt.figure(figsize=(10, 8))
    for i in np.unique(labels):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"Class {i}", alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.show()


# Grad-CAM computation
def grad_cam(model, img_tensor, layer_name):
    gradients, activations = {}, {}

    def fw_hook(m, i, o):
        activations[layer_name] = o

    def bw_hook(m, gi, go):
        gradients[layer_name] = go[0]

    h_fw = dict(model.named_modules())[layer_name].register_forward_hook(fw_hook)
    h_bw = dict(model.named_modules())[layer_name].register_full_backward_hook(bw_hook)

    output = model(img_tensor)
    pred_class = output.argmax().item()
    one_hot = torch.zeros_like(output)
    one_hot[0][pred_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot)

    pooled_grads = gradients[layer_name].mean(dim=[2, 3], keepdim=True)
    act = activations[layer_name]
    cam = torch.relu((act * pooled_grads).sum(dim=1)).squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    h_fw.remove(), h_bw.remove()
    return cam


# Overlay Grad-CAM heatmap
def show_cam_on_image(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam / np.max(cam)), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# Run UMAP and Grad-CAM visualizations
if __name__ == "__main__":
    MODEL_PATH = "model_epoch_30.pth"  # Path to your trained model
    CSV_PATH = "test_data.csv"
    IMG_DIR = "bodyparts_classification/test"

    model = load_model(MODEL_PATH)
    dataset = get_dataset(CSV_PATH, IMG_DIR)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Extract feature vectors from fc.0
    features, labels = extract_features(model, "fc.0", loader)
    plot_umap(features, labels.numpy(), "UMAP from fc.0")
    plot_tsne(features, labels.numpy(), "t-SNE from fc.0")

    img_tensor, _, img_path = dataset[0]
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    cam = grad_cam(model, img_tensor, "layer4.2.conv3")
    show_cam_on_image(img_path, cam)
