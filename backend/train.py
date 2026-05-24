import os
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


EPP_CLASSES  = ['delantal', 'gorro', 'guantes', 'tapabocas']
IMG_SIZE     = 224         
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n{'='*60}")
print(f"  EPP Cafetería UAO — Entrenamiento CNN")
print(f"  Dispositivo: {DEVICE.upper()}")
print(f"{'='*60}\n")


def build_transforms():

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


def load_datasets(dataset_root: str):
    train_tf, eval_tf = build_transforms()

    train_ds = ImageFolder(root=os.path.join(dataset_root, 'train'),
                           transform=train_tf)
    valid_ds  = ImageFolder(root=os.path.join(dataset_root, 'valid'),
                            transform=eval_tf)
    test_ds   = ImageFolder(root=os.path.join(dataset_root, 'test'),
                            transform=eval_tf)

    print("📂 Dataset cargado:")
    print(f"   Train : {len(train_ds):>4} imágenes  |  clases: {train_ds.classes}")
    print(f"   Valid : {len(valid_ds):>4} imágenes")
    print(f"   Test  : {len(test_ds):>4} imágenes\n")

    return train_ds, valid_ds, test_ds


def build_loaders(train_ds, valid_ds, test_ds, batch_size: int):
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    valid_loader  = DataLoader(valid_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=pin)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, valid_loader, test_loader


class EPPClassifier(nn.Module):
    
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.features = backbone.features 
        self.pool     = nn.AdaptiveAvgPool2d(1)  

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)        
        x = self.pool(x).flatten(1)   
        return self.classifier(x)    

    def unfreeze_top(self, n_layers: int = 3):
        layers = list(self.features.children())
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"   ✓ Fine-tuning: {n_layers} últimas capas del backbone descongeladas")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, valid_loader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print(f"\n{'─'*50}")
    print(f"  Entrenando {epochs} épocas  |  lr={lr}  |  device={device}")
    print(f"{'─'*50}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ' ← mejor' if vl_acc > best_val_acc else ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)

        print(f"  Época {epoch:02d}/{epochs}  "
              f"loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc: {tr_acc:.3f}/{vl_acc:.3f}{marker}")

    print(f"\n  ✓ Mejor val_acc: {best_val_acc:.4f}")
    print(f"  ✓ Modelo guardado en: {save_path}\n")
    return history


def fine_tune(model, train_loader, valid_loader, epochs, device, save_path):
    print(f"\n{'─'*50}")
    print("  FASE 2 — Fine-tuning")
    print(f"{'─'*50}")
    model.unfreeze_top(n_layers=3)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        marker = ' ← mejor' if vl_acc > best_val_acc else ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
        print(f"  FT {epoch:02d}/{epochs}  "
              f"loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc: {tr_acc:.3f}/{vl_acc:.3f}{marker}")

    return history


def plot_history(history, ft_history=None, out_dir='model'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    all_tl = history['train_loss'] + (ft_history['train_loss'] if ft_history else [])
    all_vl = history['val_loss']   + (ft_history['val_loss']   if ft_history else [])
    all_ta = history['train_acc']  + (ft_history['train_acc']  if ft_history else [])
    all_va = history['val_acc']    + (ft_history['val_acc']    if ft_history else [])

    epochs = range(1, len(all_tl) + 1)
    split  = len(history['train_loss'])

    axes[0].plot(epochs, all_tl, color='#4dabf7', label='Train')
    axes[0].plot(epochs, all_vl, color='#00e5b0', label='Val')
    if ft_history:
        axes[0].axvline(x=split, color='#ffb347', linestyle='--', alpha=0.5, label='Fine-tune')
    axes[0].set_title('Loss', color='#e6edf3')
    axes[0].legend(facecolor='#1c2330', labelcolor='#e6edf3')

    axes[1].plot(epochs, all_ta, color='#4dabf7', label='Train')
    axes[1].plot(epochs, all_va, color='#00e5b0', label='Val')
    if ft_history:
        axes[1].axvline(x=split, color='#ffb347', linestyle='--', alpha=0.5, label='Fine-tune')
    axes[1].set_title('Accuracy', color='#e6edf3')
    axes[1].legend(facecolor='#1c2330', labelcolor='#e6edf3')

    plt.tight_layout()
    out = os.path.join(out_dir, 'training_history.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  ✓ Gráfica guardada: {out}")
    plt.close()


def plot_confusion_matrix(labels, preds, classes, out_dir='model'):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5, linecolor='#30363d')
    ax.set_title('Matriz de Confusión', color='#e6edf3', pad=12)
    ax.set_xlabel('Predicho',  color='#8b949e')
    ax.set_ylabel('Real',      color='#8b949e')
    ax.tick_params(colors='#8b949e')
    plt.tight_layout()
    out = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  ✓ Matriz de confusión guardada: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EPP Cafetería — Entrenamiento CNN')
    parser.add_argument('--dataset', default='../dataset', help='Ruta al dataset')
    parser.add_argument('--epochs',  type=int, default=30,  help='Épocas fase 1')
    parser.add_argument('--ft_epochs', type=int, default=10, help='Épocas fine-tuning')
    parser.add_argument('--batch',   type=int, default=32,  help='Batch size')
    parser.add_argument('--lr',      type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--out',     default='model',        help='Directorio salida')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    save_path = os.path.join(args.out, 'epp_classifier.pt')

    train_ds, valid_ds, test_ds = load_datasets(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(
        train_ds, valid_ds, test_ds, args.batch)

    model = EPPClassifier(num_classes=len(EPP_CLASSES), freeze_backbone=True)
    model = model.to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales   : {total_params:,}")
    print(f"  Parámetros trainable : {trainable_params:,}  ({trainable_params/total_params*100:.1f}%)")

    history = train(model, train_loader, valid_loader,
                    args.epochs, args.lr, DEVICE, save_path)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    ft_history = fine_tune(model, train_loader, valid_loader,
                           args.ft_epochs, DEVICE, save_path)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\n{'─'*50}")
    print(f"  EVALUACIÓN FINAL EN TEST")
    print(f"{'─'*50}")
    print(f"  Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)\n")
    print(classification_report(test_labels, test_preds,
                                 target_names=EPP_CLASSES))

    meta = {
        'classes': EPP_CLASSES,
        'img_size': IMG_SIZE,
        'test_accuracy': round(test_acc, 4),
        'architecture': 'MobileNetV2 + Custom Head',
    }
    with open(os.path.join(args.out, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    plot_history(history, ft_history, out_dir=args.out)
    plot_confusion_matrix(test_labels, test_preds, EPP_CLASSES, out_dir=args.out)

    print(f"\n  ✓ Entrenamiento completo. Archivos en '{args.out}/':")
    print(f"     • epp_classifier.pt   — pesos del modelo")
    print(f"     • model_meta.json     — metadata")
    print(f"     • training_history.png")
    print(f"     • confusion_matrix.png\n")


if __name__ == '__main__':
    main()
