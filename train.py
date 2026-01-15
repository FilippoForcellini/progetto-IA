"""
train.py - Script di addestramento con supporto per:
- Configurazione centralizzata (config.json)
- Early Stopping (con patience e target accuracy)
- Checkpoint e Resume del training
- TensorBoard per visualizzazione metriche
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
import argparse
from datetime import datetime

from models import get_model
from utils import (
    load_config, get_device, ensure_dirs,
    EarlyStopping, CheckpointManager, format_time
)


def get_transforms(config):
    """
    Crea le trasformazioni per training e test basate sulla configurazione.
    """
    img_size = config['data']['image_size']
    mean = tuple(config['data']['normalize_mean'])
    std = tuple(config['data']['normalize_std'])
    aug = config['augmentation']

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(aug['rotation_degrees']),
        transforms.ColorJitter(
            brightness=aug['brightness'],
            contrast=aug['contrast'],
            saturation=aug['saturation'],
            hue=aug['hue']
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, test_transform


def get_data_loaders(config):
    """
    Prepara i DataLoader con Data Augmentation.
    """
    train_transform, test_transform = get_transforms(config)
    data_root = config['paths']['data_root']
    batch_size = config['training']['batch_size']

    print("Caricamento dati...")
    train_set = torchvision.datasets.GTSRB(
        root=data_root, split='train', download=True, transform=train_transform
    )
    test_set = torchvision.datasets.GTSRB(
        root=data_root, split='test', download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Training samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")

    return train_loader, test_loader


def get_optimizer(model, config):
    """
    Crea l'optimizer basato sulla configurazione.
    """
    opt_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer non supportato: {opt_name}")


def get_scheduler(optimizer, config):
    """
    Crea lo scheduler del learning rate basato sulla configurazione.
    """
    sched_config = config['training']['scheduler']

    if not sched_config['enabled']:
        return None

    sched_type = sched_config['type'].lower()

    if sched_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif sched_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=sched_config['gamma'],
            patience=3
        )
    else:
        return None


def train_one_epoch(model, loader, criterion, optimizer, device, writer=None, epoch=0, log_interval=10):
    """
    Esegue un'epoca di training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log su TensorBoard ogni log_interval batch
        if writer is not None and batch_count % log_interval == 0:
            global_step = epoch * len(loader) + batch_count
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        batch_count += 1

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Valuta il modello sul dataset di test.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def plot_results(history, model_type, results_dir):
    """
    Genera i grafici di training e li salva.
    """
    epochs_range = range(1, len(history['train_acc']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico Accuratezza
    axes[0].plot(epochs_range, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs_range, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title(f'Accuratezza - {model_type.upper()}', fontsize=14)
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Accuratezza (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Grafico Loss
    axes[1].plot(epochs_range, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs_range, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title(f'Loss - {model_type.upper()}', fontsize=14)
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(results_dir, f'results_{model_type}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Grafici salvati: {save_path}")
    plt.show()


def train(config, resume=False, resume_path=None):
    """
    Funzione principale di training.

    Args:
        config: Dizionario di configurazione
        resume: Se True, riprende da un checkpoint precedente
        resume_path: Percorso specifico del checkpoint (opzionale)
    """
    # Setup
    device = get_device()
    ensure_dirs(config)

    model_type = config['training']['model_type']
    epochs = config['training']['epochs']

    print(f"\n{'='*60}")
    print(f"TRAINING - {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Dati
    train_loader, test_loader = get_data_loaders(config)

    # Modello
    model = get_model(model_type, config).to(device)
    print(f"Modello {model_type} inizializzato")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Checkpoint Manager
    checkpoint_mgr = CheckpointManager(
        models_dir=config['paths']['models_dir'],
        model_type=model_type,
        save_best=config['checkpoint']['save_best'],
        save_last=config['checkpoint']['save_last']
    )

    # Early Stopping
    es_config = config['early_stopping']
    early_stopping = None
    if es_config['enabled']:
        early_stopping = EarlyStopping(
            patience=es_config['patience'],
            min_delta=es_config['min_delta'],
            mode=es_config['mode'],
            target=es_config['target_accuracy']
        )
        print(f"Early Stopping attivo: patience={es_config['patience']}, target={es_config['target_accuracy']}%")

    # TensorBoard
    writer = None
    if config['tensorboard']['enabled']:
        log_dir = os.path.join(
            config['paths']['logs_dir'],
            f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard attivo: {log_dir}")
        print(f"Visualizza con: tensorboard --logdir={config['paths']['logs_dir']}")

    # Resume da checkpoint
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    if resume:
        checkpoint_data = checkpoint_mgr.load_checkpoint(model, optimizer, resume_path)
        if checkpoint_data is not None:
            start_epoch, history = checkpoint_data
            best_acc = max(history['val_acc']) if history['val_acc'] else 0.0

    # Training Loop
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        # Training
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            writer, epoch, config['tensorboard']['log_interval']
        )

        # Validation
        v_loss, v_acc = evaluate(model, test_loader, criterion, device)

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(v_acc)
            else:
                scheduler.step()

        # Salva history
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Log su TensorBoard
        if writer is not None:
            writer.add_scalars('Loss', {'train': t_loss, 'val': v_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': t_acc, 'val': v_acc}, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}%")
        print(f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f} | Tempo: {format_time(epoch_time)}")

        # Checkpoint
        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
        checkpoint_mgr.save_checkpoint(model, optimizer, epoch, v_acc, history, is_best)

        # Early Stopping
        if early_stopping is not None:
            if not early_stopping(v_acc):
                if early_stopping.target_reached:
                    print(f"\nTarget accuracy {es_config['target_accuracy']}% raggiunto!")
                else:
                    print("\nEarly stopping attivato!")
                break

    # Fine training
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETATO")
    print(f"{'='*60}")
    print(f"Tempo totale: {format_time(total_time)}")
    print(f"Miglior accuratezza: {best_acc:.2f}%")
    print(f"Modello salvato in: {checkpoint_mgr.get_best_model_path()}")

    # Chiudi TensorBoard
    if writer is not None:
        writer.close()

    # Plot risultati
    plot_results(history, model_type, config['paths']['results_dir'])

    return history, best_acc


def parse_args():
    """
    Parse degli argomenti da linea di comando.
    """
    parser = argparse.ArgumentParser(description='Training GTSRB')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Percorso file configurazione')
    parser.add_argument('--model', type=str, choices=['custom', 'resnet'],
                        help='Override del tipo modello')
    parser.add_argument('--epochs', type=int, help='Override numero epoche')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--resume', action='store_true',
                        help='Riprendi da ultimo checkpoint')
    parser.add_argument('--resume-from', type=str,
                        help='Percorso specifico checkpoint da cui riprendere')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Carica configurazione
    config = load_config(args.config)

    # Override da linea di comando
    if args.model:
        config['training']['model_type'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Check resume da config
    resume = args.resume
    resume_path = args.resume_from

    if config['checkpoint']['resume_from'] is not None and not resume:
        resume = True
        resume_path = config['checkpoint']['resume_from']

    # Avvia training
    train(config, resume=resume, resume_path=resume_path)
