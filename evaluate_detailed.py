import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Se non ce l'hai: pip install seaborn
from models import CustomCNN, TransferResNet

# CONFIGURAZIONE
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "resnet"  # DEVE essere lo stesso usato in train.py
MODEL_PATH = f'models/best_model_{MODEL_TYPE}.pth'

def load_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_model():
    # 1. Carica Dati
    test_loader = load_data()
    
    # 2. Carica Architettura
    print(f"Caricamento modello {MODEL_TYPE}...")
    if MODEL_TYPE == "custom":
        model = CustomCNN(num_classes=43)
    else:
        model = TransferResNet(num_classes=43)
        
    # 3. Carica i Pesi Addestrati
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    # 4. Predizioni
    print("Calcolo predizioni...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Metriche e Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds))

    # 6. Matrice di Confusione Grafica
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') # annot=False perché 43 classi sono troppe per i numeri
    plt.title(f'Confusion Matrix - {MODEL_TYPE}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{MODEL_TYPE}.png')
    print(f"Matrice salvata come confusion_matrix_{MODEL_TYPE}.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()