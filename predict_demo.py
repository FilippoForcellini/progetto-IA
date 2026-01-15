"""
predict_demo.py - Demo interattiva per predizioni su immagini casuali
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse

from models import get_model
from utils import load_config, get_device, get_class_names


def get_test_dataset(config):
    """
    Carica il dataset di test.
    """
    img_size = config['data']['image_size']
    mean = tuple(config['data']['normalize_mean'])
    std = tuple(config['data']['normalize_std'])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return torchvision.datasets.GTSRB(
        root=config['paths']['data_root'],
        split='test',
        download=True,
        transform=transform
    )


def load_model(config, model_type=None):
    """
    Carica il modello addestrato.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    device = get_device()

    # Crea modello
    model = get_model(model_type, config)

    # Carica pesi
    model_path = os.path.join(
        config['paths']['models_dir'],
        f'best_model_{model_type}.pth'
    )

    if not os.path.exists(model_path):
        print(f"ERRORE: Modello non trovato: {model_path}")
        print("Devi prima addestrare il modello con: python train.py")
        return None, None

    print(f"Caricamento modello {model_type}...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


def denormalize(tensor, mean, std):
    """
    Denormalizza un tensore per la visualizzazione.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def predict_single_image(model, image_tensor, device, config):
    """
    Esegue la predizione su una singola immagine.
    """
    with torch.no_grad():
        img_input = image_tensor.unsqueeze(0).to(device)
        outputs = model(img_input)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    return predicted_idx.item(), confidence.item()


def predict_random_image(config, model_type=None, num_predictions=1):
    """
    Esegue predizioni su immagini casuali dal test set.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    # Carica modello e dati
    model, device = load_model(config, model_type)
    if model is None:
        return

    test_set = get_test_dataset(config)
    class_names = get_class_names(config)

    mean = config['data']['normalize_mean']
    std = config['data']['normalize_std']

    print(f"\nMostrando {num_predictions} predizione(i) casuale(i)...\n")

    # Crea figura
    cols = min(num_predictions, 4)
    rows = (num_predictions + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    if num_predictions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    correct = 0
    for i in range(num_predictions):
        # Seleziona immagine casuale (FIX: usa len-1 per evitare index out of range)
        idx = random.randint(0, len(test_set) - 1)
        img_tensor, true_label = test_set[idx]

        # Predizione
        pred_label, confidence = predict_single_image(model, img_tensor, device, config)

        # Nomi classi
        true_name = class_names.get(true_label, f"Classe {true_label}")
        pred_name = class_names.get(pred_label, f"Classe {pred_label}")

        # Verifica correttezza
        is_correct = true_label == pred_label
        if is_correct:
            correct += 1

        # Denormalizza per visualizzazione
        img_display = denormalize(img_tensor, mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        # Plot
        ax = axes[i] if num_predictions > 1 else axes[0]
        ax.imshow(img_display)

        color = 'green' if is_correct else 'red'
        symbol = 'V' if is_correct else 'X'

        ax.set_title(
            f"[{symbol}] Immagine #{idx}\n"
            f"Reale: {true_name}\n"
            f"Predetto: {pred_name}\n"
            f"Confidenza: {confidence*100:.1f}%",
            color=color, fontsize=10, fontweight='bold'
        )
        ax.axis('off')

    # Nascondi assi vuoti
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Demo Predizioni - {model_type.upper()} ({correct}/{num_predictions} corrette)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\nRisultato: {correct}/{num_predictions} predizioni corrette ({100*correct/num_predictions:.1f}%)")


def predict_by_class(config, class_id, model_type=None, num_samples=4):
    """
    Mostra predizioni per una specifica classe.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    model, device = load_model(config, model_type)
    if model is None:
        return

    test_set = get_test_dataset(config)
    class_names = get_class_names(config)

    mean = config['data']['normalize_mean']
    std = config['data']['normalize_std']

    # Trova immagini di questa classe
    class_indices = [i for i, (_, label) in enumerate(test_set._samples) if label == class_id]

    if not class_indices:
        print(f"Nessuna immagine trovata per la classe {class_id}")
        return

    # Seleziona campioni
    num_show = min(num_samples, len(class_indices))
    selected = random.sample(class_indices, num_show)

    class_name = class_names.get(class_id, f"Classe {class_id}")
    print(f"\nPredizioni per classe {class_id}: {class_name}")

    fig, axes = plt.subplots(1, num_show, figsize=(4*num_show, 4))
    if num_show == 1:
        axes = [axes]

    correct = 0
    for i, idx in enumerate(selected):
        img_tensor, true_label = test_set[idx]
        pred_label, confidence = predict_single_image(model, img_tensor, device, config)

        is_correct = true_label == pred_label
        if is_correct:
            correct += 1

        img_display = denormalize(img_tensor, mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)

        axes[i].imshow(img_display)
        color = 'green' if is_correct else 'red'
        pred_name = class_names.get(pred_label, f"Classe {pred_label}")
        axes[i].set_title(f"Pred: {pred_name}\n{confidence*100:.1f}%", color=color, fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Classe {class_id}: {class_name} ({correct}/{num_show} corrette)', fontsize=12)
    plt.tight_layout()
    plt.show()


def interactive_demo(config, model_type=None):
    """
    Demo interattiva con menu.
    """
    if model_type is None:
        model_type = config['training']['model_type']

    class_names = get_class_names(config)

    while True:
        print("\n" + "="*50)
        print("DEMO PREDIZIONI GTSRB")
        print("="*50)
        print(f"Modello: {model_type.upper()}")
        print("-"*50)
        print("1. Predici immagine casuale")
        print("2. Predici multiple immagini casuali")
        print("3. Predici per classe specifica")
        print("4. Mostra elenco classi")
        print("0. Esci")

        choice = input("\nScelta: ").strip()

        if choice == '1':
            predict_random_image(config, model_type, num_predictions=1)
        elif choice == '2':
            try:
                n = int(input("Quante immagini? [1-16]: "))
                n = max(1, min(16, n))
                predict_random_image(config, model_type, num_predictions=n)
            except ValueError:
                print("Numero non valido")
        elif choice == '3':
            try:
                class_id = int(input("ID classe [0-42]: "))
                if 0 <= class_id <= 42:
                    predict_by_class(config, class_id, model_type)
                else:
                    print("ID classe non valido")
            except ValueError:
                print("ID non valido")
        elif choice == '4':
            print("\nELENCO CLASSI:")
            for i in range(43):
                print(f"  {i:2d}: {class_names.get(i, 'N/A')}")
        elif choice == '0':
            break
        else:
            print("Scelta non valida")


def parse_args():
    parser = argparse.ArgumentParser(description='Demo predizioni GTSRB')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Percorso file configurazione')
    parser.add_argument('--model', type=str, choices=['custom', 'resnet'],
                        help='Tipo modello')
    parser.add_argument('--num', type=int, default=1,
                        help='Numero di predizioni (default: 1)')
    parser.add_argument('--interactive', action='store_true',
                        help='Avvia demo interattiva')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    model_type = args.model if args.model else config['training']['model_type']

    if args.interactive:
        interactive_demo(config, model_type)
    else:
        predict_random_image(config, model_type, num_predictions=args.num)
