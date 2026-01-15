"""
evaluate_detailed.py - Valutazione dettagliata del modello con metriche e matrice di confusione
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse

from models import get_model
from utils import load_config, get_device, get_class_names


def get_test_loader(config):
    """
    Crea il DataLoader per il test set.
    """
    img_size = config['data']['image_size']
    mean = tuple(config['data']['normalize_mean'])
    std = tuple(config['data']['normalize_std'])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_set = torchvision.datasets.GTSRB(
        root=config['paths']['data_root'],
        split='test',
        download=True,
        transform=transform
    )

    return DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )


def load_trained_model(config, model_type=None, checkpoint_path=None):
    """
    Carica il modello addestrato.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    device = get_device()

    # Crea modello
    model = get_model(model_type, config)

    # Carica pesi
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config['paths']['models_dir'],
            f'best_model_{model_type}.pth'
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Modello non trovato: {checkpoint_path}\nEsegui prima il training!")

    print(f"Caricamento modello da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Il checkpoint potrebbe contenere solo state_dict o essere un checkpoint completo
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint epoca: {checkpoint.get('epoch', 'N/A')}")
        print(f"Score salvato: {checkpoint.get('score', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


def evaluate_model(config, model_type=None, checkpoint_path=None, show_report=True):
    """
    Valuta il modello e genera metriche dettagliate.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    print(f"\n{'='*60}")
    print(f"VALUTAZIONE MODELLO: {model_type.upper()}")
    print(f"{'='*60}")

    # Carica modello e dati
    model, device = load_trained_model(config, model_type, checkpoint_path)
    test_loader = get_test_loader(config)

    all_preds = []
    all_labels = []
    all_probs = []

    # Predizioni
    print("\nCalcolo predizioni...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuratezza globale
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nAccuratezza globale: {accuracy:.2f}%")

    # Classification report
    if show_report:
        class_names = get_class_names(config)
        target_names = [f"{i}: {class_names.get(i, '?')}" for i in range(config['project']['num_classes'])]

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # Analisi errori per classe
    print("\n" + "="*60)
    print("ANALISI ERRORI PER CLASSE")
    print("="*60)

    errors_per_class = {}
    for true, pred in zip(all_labels, all_preds):
        if true != pred:
            if true not in errors_per_class:
                errors_per_class[true] = []
            errors_per_class[true].append(pred)

    # Top 5 classi con più errori
    error_counts = {k: len(v) for k, v in errors_per_class.items()}
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    class_names = get_class_names(config)
    print("\nTop 5 classi con più errori:")
    for class_id, count in top_errors:
        class_name = class_names.get(class_id, f"Classe {class_id}")
        total_samples = np.sum(all_labels == class_id)
        error_rate = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {class_id} ({class_name}): {count} errori ({error_rate:.1f}%)")

    return all_labels, all_preds, all_probs


def plot_confusion_matrix(all_labels, all_preds, config, model_type=None, save_path=None):
    """
    Genera e salva la matrice di confusione.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    num_classes = config['project']['num_classes']
    cm = confusion_matrix(all_labels, all_preds)

    # Normalizza per visualizzazione migliore
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Crea figura grande
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Matrice raw
    sns.heatmap(cm, ax=axes[0], cmap='Blues', fmt='d',
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    axes[0].set_title(f'Matrice di Confusione - {model_type.upper()} (Conteggi)', fontsize=14)
    axes[0].set_xlabel('Classe Predetta')
    axes[0].set_ylabel('Classe Reale')

    # Matrice normalizzata
    sns.heatmap(cm_normalized, ax=axes[1], cmap='Blues', fmt='.2f',
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1)
    axes[1].set_title(f'Matrice di Confusione - {model_type.upper()} (Normalizzata)', fontsize=14)
    axes[1].set_xlabel('Classe Predetta')
    axes[1].set_ylabel('Classe Reale')

    plt.tight_layout()

    # Salva
    if save_path is None:
        save_path = os.path.join(
            config['paths']['results_dir'],
            f'confusion_matrix_{model_type}.png'
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMatrice di confusione salvata: {save_path}")

    plt.show()

    return cm


def plot_top_confusions(all_labels, all_preds, config, model_type=None, top_n=10):
    """
    Mostra le coppie di classi più confuse.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    class_names = get_class_names(config)
    cm = confusion_matrix(all_labels, all_preds)

    # Trova le coppie più confuse (escludendo la diagonale)
    np.fill_diagonal(cm, 0)
    confusions = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confusions.append((i, j, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = confusions[:top_n]

    print("\n" + "="*60)
    print(f"TOP {top_n} CONFUSIONI TRA CLASSI")
    print("="*60)

    for true_class, pred_class, count in top_confusions:
        true_name = class_names.get(true_class, f"Classe {true_class}")
        pred_name = class_names.get(pred_class, f"Classe {pred_class}")
        print(f"  {true_class} ({true_name}) -> {pred_class} ({pred_name}): {count} volte")


def parse_args():
    """
    Parse degli argomenti da linea di comando.
    """
    parser = argparse.ArgumentParser(description='Valutazione modello GTSRB')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Percorso file configurazione')
    parser.add_argument('--model', type=str, choices=['custom', 'resnet'],
                        help='Tipo modello da valutare')
    parser.add_argument('--checkpoint', type=str,
                        help='Percorso specifico checkpoint')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Carica configurazione
    config = load_config(args.config)

    model_type = args.model if args.model else config['training']['model_type']

    # Valuta
    all_labels, all_preds, all_probs = evaluate_model(
        config, model_type, args.checkpoint
    )

    # Matrice di confusione
    cm = plot_confusion_matrix(all_labels, all_preds, config, model_type)

    # Top confusioni
    plot_top_confusions(all_labels, all_preds, config, model_type)
