import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Definizione delle trasformazioni di base (Ridimensioniamo tutto a 32x32 per uniformità)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def get_datasets(root='./data'):
    print("Download e preparazione dataset GTSRB...")
    # Scarichiamo il train set
    train_set = torchvision.datasets.GTSRB(
        root=root, split='train', download=True, transform=transform
    )
    # Scarichiamo il test set
    test_set = torchvision.datasets.GTSRB(
        root=root, split='test', download=True, transform=transform
    )
    return train_set, test_set

def analyze_dataset(dataset, dataset_name="Train"):
    """
    Funzione per soddisfare i requisiti di analisi statistica e bilanciamento [cite: 72, 73]
    """
    print(f"Analisi del dataset {dataset_name}...")
    
    # Estraiamo le etichette (labels)
    # Nota: In GTSRB le labels sono dentro dataset._samples (lista di tuple path, label)
    labels = [label for _, label in dataset._samples]
    
    # Contiamo le occorrenze per classe
    counts = Counter(labels)
    classes = sorted(counts.keys())
    frequencies = [counts[c] for c in classes]
    
    # Grafico 1: Distribuzione delle classi (Istogramma)
    plt.figure(figsize=(12, 6))
    plt.bar(classes, frequencies, color='skyblue', edgecolor='black')
    plt.title(f'Distribuzione Campioni per Classe ({dataset_name})')
    plt.xlabel('ID Classe (0-42)')
    plt.ylabel('Numero di Immagini')
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    
    print(f"Totale immagini: {len(dataset)}")
    print(f"Classe più frequente: {max(counts, key=counts.get)} con {max(frequencies)} campioni")
    print(f"Classe meno frequente: {min(counts, key=counts.get)} con {min(frequencies)} campioni")
    
    # Visualizzazione Esempi (Grid)
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 4, 4
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Class: {label}")
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0)) # Da (C,H,W) a (H,W,C) per matplotlib
    plt.show()

if __name__ == "__main__":
    train_data, _ = get_datasets()
    analyze_dataset(train_data)