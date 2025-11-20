import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # Barra di caricamento
import os

# Importiamo i nostri modelli (assicurati di aver creato models.py come discusso)
from models import CustomCNN, TransferResNet

# --- 1. CONFIGURAZIONE IPER-PARAMETRI [cite: 77] ---
# Modificando questi valori fai la "sperimentazione" richiesta
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "custom" # Scegli: "custom" o "resnet"

def get_data_loaders():
    """
    Prepara i DataLoader con Data Augmentation [cite: 77]
    """
    # Trasformazioni per il Training (Augmentation)
    # Ruotiamo e cambiamo leggermente i colori per rendere la rete robusta
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalizzazione
    ])

    # Trasformazioni per il Test (Nessuna alterazione, solo resize)
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Scarichiamo il dataset (GTSRB)
    print("Caricamento dati...")
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)

    # Creiamo i DataLoader (necessari per iterare sui batch)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Imposta il modello in modalità training
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop attraverso i batch
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # 1. Reset dei gradienti
        optimizer.zero_grad()

        # 2. Forward pass (previsione)
        outputs = model(images)
        
        # 3. Calcolo della Loss
        loss = criterion(outputs, labels)

        # 4. Backward pass (calcolo gradienti)
        loss.backward()

        # 5. Ottimizzazione (aggiornamento pesi)
        optimizer.step()

        # Statistiche
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval() # Imposta il modello in modalità valutazione (spegne dropout, ecc.)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # Disabilita il calcolo dei gradienti (risparmia memoria)
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

def plot_results(train_acc, val_acc, train_loss, val_loss):
    """Genera i grafici da inserire nel Report [cite: 8]"""
    epochs_range = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Grafico Accuratezza
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Grafico Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'results_{MODEL_TYPE}.png') # Salva l'immagine
    print(f"Grafici salvati come results_{MODEL_TYPE}.png")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Preparazione Dati
    train_loader, test_loader = get_data_loaders()

    # 2. Selezione Modello [cite: 22]
    print(f"Inizializzazione modello: {MODEL_TYPE} su {DEVICE}")
    if MODEL_TYPE == "resnet":
        model = CustomCNN(num_classes=43).to(DEVICE)
    else:
        model = TransferResNet(num_classes=43).to(DEVICE)

    # 3. Loss e Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Liste per salvare i risultati per i grafici
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    # 4. Loop di Addestramento 
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validation
        v_loss, v_acc = evaluate(model, test_loader, criterion, DEVICE)
        
        print(f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}%")
        print(f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%")

        # Salvataggio storico
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Salvataggio Miglior Modello 
        if v_acc > best_acc:
            best_acc = v_acc
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), f'models/best_model_{MODEL_TYPE}.pth')
            print("Miglior modello salvato!")

    # 5. Plotting finale
    plot_results(history['train_acc'], history['val_acc'], 
                 history['train_loss'], history['val_loss'])