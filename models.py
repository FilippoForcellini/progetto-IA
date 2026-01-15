"""
models.py - Definizione delle architetture neurali per classificazione GTSRB
"""
import torch.nn as nn
import torchvision.models as models
from utils import load_config


def get_model(model_type, config=None):
    """
    Factory function per creare il modello appropriato.

    Args:
        model_type: 'custom' o 'resnet'
        config: Configurazione (se None, viene caricata automaticamente)

    Returns:
        nn.Module: Il modello richiesto
    """
    if config is None:
        config = load_config()

    num_classes = config['project']['num_classes']

    if model_type == 'custom':
        model_config = config['model']['custom_cnn']
        return CustomCNN(
            num_classes=num_classes,
            conv_filters=model_config['conv_filters'],
            fc_hidden=model_config['fc_hidden'],
            dropout=model_config['dropout']
        )
    elif model_type == 'resnet':
        model_config = config['model']['resnet']
        return TransferResNet(
            num_classes=num_classes,
            pretrained=model_config['pretrained'],
            freeze_backbone=model_config['freeze_backbone']
        )
    else:
        raise ValueError(f"Tipo modello non valido: {model_type}. Usa 'custom' o 'resnet'")


class CustomCNN(nn.Module):
    """
    CNN personalizzata per classificazione segnali stradali.
    Architettura: 3 blocchi convoluzionali + classificatore fully connected.
    """
    def __init__(self, num_classes=43, conv_filters=None, fc_hidden=512, dropout=0.5):
        """
        Args:
            num_classes: Numero di classi da classificare
            conv_filters: Lista dei filtri convoluzionali [32, 64, 128]
            fc_hidden: Numero di neuroni nel layer nascosto FC
            dropout: Tasso di dropout per regolarizzazione
        """
        super(CustomCNN, self).__init__()

        if conv_filters is None:
            conv_filters = [32, 64, 128]

        # Costruzione dinamica dei layer convoluzionali
        layers = []
        in_channels = 3  # RGB input

        for out_channels in conv_filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Aggiunto BatchNorm per stabilità
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Calcolo dimensione dopo i pooling: 32 -> 16 -> 8 -> 4
        final_size = 32 // (2 ** len(conv_filters))
        flatten_size = conv_filters[-1] * final_size * final_size

        # Classificatore lineare (Fully Connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TransferResNet(nn.Module):
    """
    ResNet18 con Transfer Learning da ImageNet.
    Usa pesi pre-addestrati e sostituisce l'ultimo layer per le nostre classi.
    """
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Numero di classi da classificare
            pretrained: Se usare pesi pre-addestrati su ImageNet
            freeze_backbone: Se congelare i layer convoluzionali (solo fine-tuning del classificatore)
        """
        super(TransferResNet, self).__init__()

        # Carica ResNet18 pre-addestrata
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Congela i layer se richiesto
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Sostituisci l'ultimo layer per le nostre classi
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_backbone(self):
        """Sblocca il backbone per fine-tuning completo"""
        for param in self.resnet.parameters():
            param.requires_grad = True


# Per compatibilità con import precedenti
__all__ = ['CustomCNN', 'TransferResNet', 'get_model']
