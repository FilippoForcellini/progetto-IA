import torch.nn as nn
import torchvision.models as models

# --- MODELLO 1: Architettura Ad-Hoc (Semplice) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(CustomCNN, self).__init__()
        # Primo blocco convoluzionale
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Input: 3 canali (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Dimezza la dimensione: 32x32 -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )
        
        # Classificatore lineare (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Per evitare overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- MODELLO 2: Architettura Nota (ResNet18 con Transfer Learning) ---
class TransferResNet(nn.Module):
    def __init__(self, num_classes=43, pretrained=True):
        super(TransferResNet, self).__init__()
        # Carichiamo una ResNet18 pre-addestrata su ImageNet
        # Nota: 'weights' sostituisce 'pretrained' nelle nuove versioni di torch
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Sostituiamo l'ultimo livello (che ha 1000 classi di ImageNet)
        # con uno che ne ha 43 (i nostri segnali)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)