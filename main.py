"""
main.py - Menu principale del progetto GTSRB
Gestisce l'interazione con l'utente per tutte le operazioni disponibili.
"""
import os
import sys
import json

from utils import load_config


def print_header():
    """Stampa l'intestazione del programma."""
    print("\n" + "="*60)
    print("   PROGETTO GTSRB - Classificazione Segnali Stradali")
    print("   German Traffic Sign Recognition Benchmark")
    print("="*60)


def print_config_summary(config):
    """Mostra un riepilogo della configurazione attuale."""
    print("\nConfigurazione attuale:")
    print(f"  Modello: {config['training']['model_type'].upper()}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Early Stopping: {'Attivo' if config['early_stopping']['enabled'] else 'Disattivo'}")
    print(f"  TensorBoard: {'Attivo' if config['tensorboard']['enabled'] else 'Disattivo'}")


def menu_training(config):
    """Sottomenu per il training."""
    print("\n--- OPZIONI TRAINING ---")
    print(f"Modello selezionato: {config['training']['model_type'].upper()}")
    print("-"*40)
    print("1. Avvia nuovo training")
    print("2. Riprendi da checkpoint")
    print("3. Cambia modello (custom/resnet)")
    print("4. Modifica parametri training")
    print("0. Torna al menu principale")

    choice = input("\nScelta: ").strip()

    if choice == '1':
        print("\nAvvio training...")
        os.system(f"{sys.executable} train.py")
    elif choice == '2':
        print("\nRipresa training da checkpoint...")
        os.system(f"{sys.executable} train.py --resume")
    elif choice == '3':
        current = config['training']['model_type']
        new_model = 'resnet' if current == 'custom' else 'custom'
        print(f"\nCambio modello da {current} a {new_model}")
        os.system(f"{sys.executable} train.py --model {new_model}")
    elif choice == '4':
        modify_training_params(config)
    elif choice == '0':
        return
    else:
        print("Scelta non valida")


def modify_training_params(config):
    """Permette di modificare i parametri di training."""
    print("\n--- MODIFICA PARAMETRI ---")
    print("(Premi INVIO per mantenere il valore attuale)")

    try:
        # Epochs
        epochs = input(f"Epochs [{config['training']['epochs']}]: ").strip()
        if epochs:
            config['training']['epochs'] = int(epochs)

        # Learning rate
        lr = input(f"Learning Rate [{config['training']['learning_rate']}]: ").strip()
        if lr:
            config['training']['learning_rate'] = float(lr)

        # Batch size
        bs = input(f"Batch Size [{config['training']['batch_size']}]: ").strip()
        if bs:
            config['training']['batch_size'] = int(bs)

        # Model type
        model = input(f"Modello (custom/resnet) [{config['training']['model_type']}]: ").strip()
        if model in ['custom', 'resnet']:
            config['training']['model_type'] = model

        # Salva configurazione
        save = input("\nSalvare le modifiche? (s/n): ").strip().lower()
        if save == 's':
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print("Configurazione salvata!")
        else:
            print("Modifiche annullate.")

    except ValueError as e:
        print(f"Errore nel valore inserito: {e}")


def menu_experiments():
    """Sottomenu per confronto esperimenti."""
    print("\n--- CONFRONTO ESPERIMENTI ---")
    print("1. Addestra modello Custom CNN")
    print("2. Addestra modello ResNet18")
    print("3. Confronta risultati (richiede entrambi i modelli)")
    print("4. Apri TensorBoard")
    print("0. Torna al menu principale")

    choice = input("\nScelta: ").strip()

    if choice == '1':
        os.system(f"{sys.executable} train.py --model custom")
    elif choice == '2':
        os.system(f"{sys.executable} train.py --model resnet")
    elif choice == '3':
        compare_models()
    elif choice == '4':
        print("\nAvvio TensorBoard...")
        print("Apri il browser su: http://localhost:6006")
        os.system("tensorboard --logdir=./runs")
    elif choice == '0':
        return
    else:
        print("Scelta non valida")


def compare_models():
    """Confronta i risultati dei due modelli."""
    import matplotlib.pyplot as plt

    config = load_config()
    results_dir = config['paths']['results_dir']

    custom_results = os.path.join(results_dir, 'results_custom.png')
    resnet_results = os.path.join(results_dir, 'results_resnet.png')

    if not os.path.exists(custom_results) or not os.path.exists(resnet_results):
        print("\nPer confrontare i modelli, devi prima addestrarli entrambi!")
        return

    print("\nConfrontando i risultati dei due modelli...")

    # Valuta entrambi i modelli
    print("\n--- Valutazione Custom CNN ---")
    os.system(f"{sys.executable} evaluate_detailed.py --model custom")

    print("\n--- Valutazione ResNet18 ---")
    os.system(f"{sys.executable} evaluate_detailed.py --model resnet")


def main():
    """Funzione principale con menu interattivo."""
    config = load_config()

    while True:
        print_header()
        print_config_summary(config)
        print("\n--- MENU PRINCIPALE ---")
        print("1. Scarica dati e mostra statistiche")
        print("2. Training modello")
        print("3. Valutazione dettagliata (matrice confusione)")
        print("4. Demo: predici immagine casuale")
        print("5. Confronto esperimenti")
        print("6. Avvia TensorBoard")
        print("7. Modifica configurazione")
        print("0. Esci")

        choice = input("\nScelta: ").strip()

        if choice == '1':
            print("\nDownload e analisi dataset...")
            os.system(f"{sys.executable} data_setup.py")

        elif choice == '2':
            menu_training(config)
            config = load_config()  # Ricarica config in caso di modifiche

        elif choice == '3':
            print("\nValutazione dettagliata del modello...")
            os.system(f"{sys.executable} evaluate_detailed.py")

        elif choice == '4':
            print("\nAvvio demo predizioni...")
            os.system(f"{sys.executable} predict_demo.py --interactive")

        elif choice == '5':
            menu_experiments()

        elif choice == '6':
            print("\nAvvio TensorBoard...")
            print("Visualizza su: http://localhost:6006")
            print("Premi Ctrl+C per terminare TensorBoard")
            os.system("tensorboard --logdir=./runs")

        elif choice == '7':
            modify_training_params(config)
            config = load_config()  # Ricarica config

        elif choice == '0':
            print("\nArrivederci!")
            break

        else:
            print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()
