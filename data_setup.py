"""
data_setup.py - Download e analisi del dataset GTSRB
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

from utils import load_config, get_class_names


def get_transform(config):
    """
    Crea la trasformazione base per visualizzazione.
    """
    img_size = config['data']['image_size']
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def get_datasets(config=None):
    """
    Scarica e prepara il dataset GTSRB.
    """
    if config is None:
        config = load_config()

    transform = get_transform(config)
    data_root = config['paths']['data_root']

    print("Download e preparazione dataset GTSRB...")

    train_set = torchvision.datasets.GTSRB(
        root=data_root, split='train', download=True, transform=transform
    )
    test_set = torchvision.datasets.GTSRB(
        root=data_root, split='test', download=True, transform=transform
    )

    print(f"Training samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")

    return train_set, test_set


def analyze_dataset(dataset, config=None, dataset_name="Train", save_dir=None):
    """
    Analisi statistica e visualizzazione del dataset.
    """
    if config is None:
        config = load_config()

    class_names = get_class_names(config)

    print(f"\n{'='*60}")
    print(f"ANALISI DATASET {dataset_name.upper()}")
    print(f"{'='*60}")

    # Estrai le etichette
    labels = [label for _, label in dataset._samples]

    # Conta occorrenze per classe
    counts = Counter(labels)
    classes = sorted(counts.keys())
    frequencies = [counts[c] for c in classes]

    # Statistiche
    print(f"\nTotale immagini: {len(dataset)}")
    print(f"Numero classi: {len(classes)}")

    max_class = max(counts, key=counts.get)
    min_class = min(counts, key=counts.get)
    print(f"Classe più frequente: {max_class} ({class_names.get(max_class, 'N/A')}) - {max(frequencies)} campioni")
    print(f"Classe meno frequente: {min_class} ({class_names.get(min_class, 'N/A')}) - {min(frequencies)} campioni")
    print(f"Media campioni per classe: {np.mean(frequencies):.1f}")
    print(f"Deviazione standard: {np.std(frequencies):.1f}")

    # Bilanciamento
    balance_ratio = min(frequencies) / max(frequencies)
    print(f"Rapporto bilanciamento (min/max): {balance_ratio:.3f}")

    if save_dir is None:
        save_dir = config['paths']['results_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Grafico 1: Distribuzione delle classi
    plt.figure(figsize=(14, 6))
    bars = plt.bar(classes, frequencies, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axhline(y=np.mean(frequencies), color='red', linestyle='--', label=f'Media: {np.mean(frequencies):.0f}')
    plt.title(f'Distribuzione Campioni per Classe ({dataset_name})', fontsize=14)
    plt.xlabel('ID Classe (0-42)', fontsize=12)
    plt.ylabel('Numero di Immagini', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    dist_path = os.path.join(save_dir, f'distribution_{dataset_name.lower()}.png')
    plt.savefig(dist_path, dpi=150)
    print(f"\nDistribuzione salvata: {dist_path}")
    plt.show()

    # Grafico 2: Esempi per classe (Grid 5x5)
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    fig.suptitle(f'Esempi di Segnali Stradali ({dataset_name})', fontsize=14)

    # Seleziona 25 classi casuali
    sample_classes = np.random.choice(classes, min(25, len(classes)), replace=False)

    for idx, (ax, class_id) in enumerate(zip(axes.flatten(), sample_classes)):
        # Trova un'immagine di questa classe
        class_indices = [i for i, (_, label) in enumerate(dataset._samples) if label == class_id]
        if class_indices:
            sample_idx = np.random.choice(class_indices)
            img, label = dataset[sample_idx]
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(f'{class_id}: {class_names.get(class_id, "?")}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()

    samples_path = os.path.join(save_dir, f'samples_{dataset_name.lower()}.png')
    plt.savefig(samples_path, dpi=150)
    print(f"Esempi salvati: {samples_path}")
    plt.show()

    return counts


def show_class_examples(dataset, class_id, config=None, num_examples=16):
    """
    Mostra esempi di una specifica classe.
    """
    if config is None:
        config = load_config()

    class_names = get_class_names(config)
    class_name = class_names.get(class_id, f"Classe {class_id}")

    # Trova tutte le immagini di questa classe
    class_indices = [i for i, (_, label) in enumerate(dataset._samples) if label == class_id]

    if not class_indices:
        print(f"Nessuna immagine trovata per la classe {class_id}")
        return

    # Seleziona esempi casuali
    num_show = min(num_examples, len(class_indices))
    selected = np.random.choice(class_indices, num_show, replace=False)

    # Crea grid
    cols = 4
    rows = (num_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    fig.suptitle(f'Classe {class_id}: {class_name}', fontsize=14)

    for idx, ax in enumerate(axes.flatten()):
        if idx < num_show:
            img, _ = dataset[selected[idx]]
            ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = load_config()
    train_data, test_data = get_datasets(config)

    # Analisi training set
    train_counts = analyze_dataset(train_data, config, "Train")

    # Analisi test set
    test_counts = analyze_dataset(test_data, config, "Test")
