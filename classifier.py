# -*- coding: utf-8 -*-

import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim




#%%
def split_dataset(original_dir, output_dir, train_ratio=0.8):
    random.seed(42)

    # Remove previous folders if they exist
    for split in ['train', 'val']:
        split_path = os.path.join(output_dir, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)

    classes = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]

    for class_name in classes:
        class_path = os.path.join(original_dir, class_name)
        images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
        random.shuffle(images)

        train_size = int(len(images) * train_ratio)
        train_images = images[:train_size]
        val_images = images[train_size:]

        for split, split_images in zip(['train', 'val'], [train_images, val_images]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)


original_dataset_path = "data/dogs"         # origiinal data    
output_dataset_path = "data/dataset_split"  # splited data

split_dataset(original_dataset_path, output_dataset_path, train_ratio=0.8)
#%%

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Transforms para entrenamiento con augmentación
transform_augm = transforms.Compose([
    transforms.RandomResizedCrop(224),            
    transforms.RandomHorizontalFlip(),            
    transforms.RandomRotation(15),                
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),              
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                         std=[0.229, 0.224, 0.225])
])




train_dataset = datasets.ImageFolder(output_dataset_path + "/train", transform=transform_augm)
val_dataset = datasets.ImageFolder(output_dataset_path + "/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases

model = model.to(device)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


#%%

model.eval()
results = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Calcular probabilidades y predicción
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)

        # Contar aciertos
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)

        # Guardar resultados para cada imagen
        for i in range(images.size(0)):
            results.append({
                'true_class': labels[i].item(),
                'predicted_class': pred_classes[i].item(),
                'confidence': max_probs[i].item()
            })


accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")


#%%

import pandas as pd
import matplotlib.pyplot as plt

# pandas DataFrame
df = pd.DataFrame(results)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df['confidence'], bins=10, color='skyblue', alpha=0.6, edgecolor='black')

plt.xlabel('Confidence')
plt.ylabel('Count')
plt.title('Confidence distribution')
plt.show()

#%%

other_val_dataset = datasets.ImageFolder("data/val_with_other", transform=transform)
other_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model.eval()
results = []
correct = 0
total = 0

class_names = train_dataset.classes
threshold = 0.85
other_class = True


with torch.no_grad():
    for images, labels in other_val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)

        for i in range(images.size(0)):
            confidence = max_probs[i].item()
            true_class = class_names[labels[i].item()]

            # Clasificación como "other" si aplica
            if other_class and confidence < threshold:
                predicted_class = "other"
            else:
                predicted_class = class_names[pred_classes[i].item()]

            results.append({
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence
            })

            # count if the prediction matches the true class (including the "other" class).
            if predicted_class == true_class:
                correct += 1

            total += 1

accuracy = correct / total
print(f"Validation Accuracy (with 'other'): {accuracy:.4f}")

