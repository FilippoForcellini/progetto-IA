import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from models import TransferResNet, CustomCNN
import random

# Dizionario delle classi GTSRB (Semplificato)
CLASSES = {
    0: 'Limite 20 km/h', 1: 'Limite 30 km/h', 2: 'Limite 50 km/h', 3: 'Limite 60 km/h',
    4: 'Limite 70 km/h', 5: 'Limite 80 km/h', 6: 'Fine limite 80', 7: 'Limite 100 km/h',
    8: 'Limite 120 km/h', 9: 'Divieto sorpasso', 10: 'Divieto sorpasso camioni',
    11: 'Precedenza incrocio', 12: 'Strada con priorità', 13: 'Dare precedenza', 14: 'STOP',
    15: 'Divieto transito', 16: 'Divieto camion', 17: 'Divieto accesso', 18: 'Pericolo generico',
    19: 'Curva a sinistra', 20: 'Curva a destra', 21: 'Doppia curva', 22: 'Strada dissestata',
    23: 'Strada sdrucciolevole', 24: 'Restringimento destra', 25: 'Lavori in corso',
    26: 'Semaforo', 27: 'Attraversamento pedoni', 28: 'Bambini', 29: 'Biciclette',
    30: 'Ghiaccio/Neve', 31: 'Animali selvatici', 32: 'Fine limiti', 33: 'Obbligo destra',
    34: 'Obbligo sinistra', 35: 'Dritto', 36: 'Dritto o destra', 37: 'Dritto o sinistra',
    38: 'Passaggio a destra', 39: 'Passaggio a sinistra', 40: 'Rotatoria',
    41: 'Divieto sorpasso (fine)', 42: 'Divieto camion (fine)'
}

def predict_random_image():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "resnet" # O "custom"
    
    # 1. Carica Modello
    print(f"Caricamento modello {MODEL_TYPE}...")
    if MODEL_TYPE == "custom":
        model = CustomCNN(num_classes=43)
    else:
        model = TransferResNet(num_classes=43)
    
    try:
        model.load_state_dict(torch.load(f'models/best_model_{MODEL_TYPE}.pth', map_location=DEVICE))
    except FileNotFoundError:
        print("ERRORE: Devi prima addestrare il modello! Esegui train.py")
        return

    model.to(DEVICE)
    model.eval()

    # 2. Carica Dataset Test
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)

    # 3. Prendi un'immagine a caso
    idx = random.randint(0, len(test_set))
    img_tensor, label_idx = test_set[idx]
    
    # 4. Fai la predizione
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(DEVICE) # Aggiungi dimensione batch
        outputs = model(img_input)
        _, predicted_idx = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item()

    # 5. Mostra Risultato
    label_real = CLASSES.get(label_idx, "Sconosciuto")
    label_pred = CLASSES.get(predicted_idx.item(), "Sconosciuto")
    
    color = 'green' if label_idx == predicted_idx.item() else 'red'
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_tensor.permute(1, 2, 0) * 0.5 + 0.5) # Denormalizza per visualizzare
    plt.title(f"Reale: {label_real}\nPredetto: {label_pred} ({confidence*100:.1f}%)", color=color, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    print(f"Immagine #{idx}")
    print(f"Reale:    {label_real}")
    print(f"Predetto: {label_pred}")

if __name__ == "__main__":
    predict_random_image()