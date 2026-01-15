"""
run_experiments.py - Script per eseguire esperimenti multipli con diverse configurazioni.
Utile per confrontare iperparametri e generare dati per il report.
"""
import os
import json
import time
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, ensure_dirs
from train import train


# ============================================================================
# ESPERIMENTI RICHIESTI DAL PROFESSORE
# ============================================================================
# Tabella esperimenti:
# | modello | batch_size | learning_rate | epoche |
# |---------|------------|---------------|--------|
# | custom  | 64         | 0.001         | 2      |
# | custom  | 64         | 0.001         | 5      |
# | custom  | 64         | 0.001         | 10     |
# | custom  | 8          | 0.001         | 5      |
# | custom  | 16         | 0.001         | 5      |
# | custom  | 32         | 0.001         | 5      |
# | custom  | 64         | 0.0001        | 5      |
# | custom  | 64         | 0.005         | 5      |
# | custom  | 64         | 0.01          | 5      |
# ============================================================================

EXPERIMENTS_PROFESSOR = [
    # =========================================
    # GRUPPO 1: Confronto numero di epoche
    # (batch_size=64, lr=0.001, epoche variabili)
    # =========================================
    {
        "name": "exp1_epochs2",
        "description": "Custom CNN - 2 epoche (bs=64, lr=0.001)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 2
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}  # Disabilitato per vedere tutte le epoche
        }
    },
    {
        "name": "exp2_epochs5",
        "description": "Custom CNN - 5 epoche (bs=64, lr=0.001)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    {
        "name": "exp3_epochs10",
        "description": "Custom CNN - 10 epoche (bs=64, lr=0.001)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 10
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },

    # =========================================
    # GRUPPO 2: Confronto batch size
    # (lr=0.001, epoche=5, batch_size variabile)
    # =========================================
    {
        "name": "exp4_bs8",
        "description": "Custom CNN - batch_size=8 (lr=0.001, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    {
        "name": "exp5_bs16",
        "description": "Custom CNN - batch_size=16 (lr=0.001, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 16,
                "learning_rate": 0.001,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    {
        "name": "exp6_bs32",
        "description": "Custom CNN - batch_size=32 (lr=0.001, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    # Nota: exp2_epochs5 già copre bs=64, lr=0.001, ep=5

    # =========================================
    # GRUPPO 3: Confronto learning rate
    # (batch_size=64, epoche=5, lr variabile)
    # =========================================
    {
        "name": "exp7_lr0001",
        "description": "Custom CNN - lr=0.0001 (bs=64, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.0001,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    {
        "name": "exp8_lr005",
        "description": "Custom CNN - lr=0.005 (bs=64, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.005,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
    {
        "name": "exp9_lr01",
        "description": "Custom CNN - lr=0.01 (bs=64, ep=5)",
        "config_overrides": {
            "training": {
                "model_type": "custom",
                "batch_size": 64,
                "learning_rate": 0.01,
                "epochs": 5
            },
            "tensorboard": {"enabled": True},
            "early_stopping": {"enabled": False}
        }
    },
]

# Esperimenti originali (mantenuti per compatibilita)
EXPERIMENTS = [
    # Esperimento 1: Confronto modelli con configurazione base
    {
        "name": "baseline_custom",
        "description": "Custom CNN con configurazione base",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 20, "learning_rate": 0.001}
        }
    },
    {
        "name": "baseline_resnet",
        "description": "ResNet18 con configurazione base",
        "config_overrides": {
            "training": {"model_type": "resnet", "epochs": 20, "learning_rate": 0.001}
        }
    },

    # Esperimento 2: Confronto learning rate
    {
        "name": "custom_lr_0001",
        "description": "Custom CNN con LR=0.0001",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "learning_rate": 0.0001}
        }
    },
    {
        "name": "custom_lr_001",
        "description": "Custom CNN con LR=0.001",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "learning_rate": 0.001}
        }
    },
    {
        "name": "custom_lr_01",
        "description": "Custom CNN con LR=0.01",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "learning_rate": 0.01}
        }
    },

    # Esperimento 3: Confronto batch size
    {
        "name": "custom_bs32",
        "description": "Custom CNN con batch_size=32",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "batch_size": 32}
        }
    },
    {
        "name": "custom_bs64",
        "description": "Custom CNN con batch_size=64",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "batch_size": 64}
        }
    },
    {
        "name": "custom_bs128",
        "description": "Custom CNN con batch_size=128",
        "config_overrides": {
            "training": {"model_type": "custom", "epochs": 15, "batch_size": 128}
        }
    },
]


def merge_config(base_config, overrides):
    """Unisce ricorsivamente le configurazioni."""
    result = copy.deepcopy(base_config)

    for key, value in overrides.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value

    return result


def run_single_experiment(exp_config, base_config, results_base_dir):
    """Esegue un singolo esperimento."""
    exp_name = exp_config['name']
    description = exp_config['description']

    print(f"\n{'='*70}")
    print(f"ESPERIMENTO: {exp_name}")
    print(f"Descrizione: {description}")
    print(f"{'='*70}")

    # Crea configurazione specifica
    config = merge_config(base_config, exp_config.get('config_overrides', {}))

    # Directory dedicate per questo esperimento (usa percorsi assoluti)
    results_base_abs = os.path.abspath(results_base_dir)
    config['paths']['results_dir'] = os.path.join(results_base_abs, exp_name)
    config['paths']['logs_dir'] = os.path.join(results_base_abs, 'runs', exp_name)

    # Crea le directory manualmente per evitare problemi con TensorBoard
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)

    ensure_dirs(config)

    # Esegui training
    start_time = time.time()
    try:
        history, best_acc = train(config, resume=False)
        elapsed = time.time() - start_time

        return {
            'name': exp_name,
            'description': description,
            'config': exp_config.get('config_overrides', {}),
            'best_accuracy': best_acc,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'epochs_run': len(history['train_acc']),
            'elapsed_time': elapsed,
            'history': history,
            'status': 'completed'
        }
    except Exception as e:
        return {
            'name': exp_name,
            'description': description,
            'status': 'failed',
            'error': str(e)
        }


def plot_comparison(results, save_path):
    """Crea grafici di confronto tra esperimenti."""
    completed = [r for r in results if r.get('status') == 'completed']

    if not completed:
        print("Nessun esperimento completato da confrontare.")
        return

    # Grafico 1: Confronto accuratezza finale
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r['name'] for r in completed]
    accuracies = [r['best_accuracy'] for r in completed]
    times = [r['elapsed_time'] / 60 for r in completed]  # in minuti

    # Bar plot accuratezza
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), accuracies, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Accuratezza (%)')
    ax.set_title('Confronto Accuratezza Migliore')
    ax.set_ylim([min(accuracies) - 5, 100])

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    # Bar plot tempo
    ax = axes[0, 1]
    ax.bar(range(len(names)), times, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Tempo (minuti)')
    ax.set_title('Tempo di Training')

    # Curve di learning (solo primi 3 esperimenti)
    ax = axes[1, 0]
    for r in completed[:5]:  # Max 5 per leggibilità
        if 'history' in r:
            ax.plot(r['history']['val_acc'], label=r['name'])
    ax.set_xlabel('Epoca')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Curve di Apprendimento')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Curve di loss
    ax = axes[1, 1]
    for r in completed[:5]:
        if 'history' in r:
            ax.plot(r['history']['val_loss'], label=r['name'])
    ax.set_xlabel('Epoca')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Curve di Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nGrafico confronto salvato: {save_path}")
    plt.show()


def print_summary(results):
    """Stampa un riepilogo degli esperimenti."""
    print("\n" + "="*80)
    print("RIEPILOGO ESPERIMENTI")
    print("="*80)

    completed = [r for r in results if r.get('status') == 'completed']
    failed = [r for r in results if r.get('status') == 'failed']

    print(f"\nCompletati: {len(completed)}/{len(results)}")

    if completed:
        # Ordina per accuratezza
        sorted_results = sorted(completed, key=lambda x: x['best_accuracy'], reverse=True)

        print("\nClassifica per accuratezza:")
        print("-"*70)
        print(f"{'#':<3} {'Esperimento':<25} {'Acc. Best':>10} {'Acc. Final':>10} {'Tempo':>10}")
        print("-"*70)

        for i, r in enumerate(sorted_results, 1):
            time_str = f"{r['elapsed_time']/60:.1f}min"
            print(f"{i:<3} {r['name']:<25} {r['best_accuracy']:>9.2f}% {r['final_val_acc']:>9.2f}% {time_str:>10}")

        # Miglior esperimento
        best = sorted_results[0]
        print(f"\nMIGLIOR RISULTATO: {best['name']} con {best['best_accuracy']:.2f}%")

    if failed:
        print("\nEsperimenti falliti:")
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', 'Errore sconosciuto')}")


def run_all_experiments(experiments=None, save_results=True):
    """Esegue tutti gli esperimenti definiti."""
    if experiments is None:
        experiments = EXPERIMENTS

    base_config = load_config(validate=False)

    # Directory per i risultati degli esperimenti
    # Usa un percorso senza caratteri speciali per evitare problemi con TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Usa la home directory dell'utente per evitare problemi con caratteri speciali
    home_dir = os.path.expanduser('~')
    results_base_dir = os.path.join(home_dir, 'gtsrb_experiments', timestamp)
    os.makedirs(results_base_dir, exist_ok=True)

    # Crea anche un link simbolico nella directory del progetto
    local_link = os.path.join('./experiments', timestamp)
    os.makedirs('./experiments', exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# AVVIO BATTERIA ESPERIMENTI")
    print(f"# Numero esperimenti: {len(experiments)}")
    print(f"# Output directory: {results_base_dir}")
    print(f"{'#'*70}")

    results = []

    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Avvio esperimento: {exp['name']}")
        result = run_single_experiment(exp, base_config, results_base_dir)
        results.append(result)

    # Salva risultati JSON
    if save_results:
        results_file = os.path.join(results_base_dir, 'results.json')
        # Rimuovi history per JSON (troppo grande)
        results_for_json = []
        for r in results:
            r_copy = copy.deepcopy(r)
            if 'history' in r_copy:
                # Salva solo statistiche riassuntive
                r_copy['history_summary'] = {
                    'epochs': len(r_copy['history']['train_acc']),
                    'final_train_acc': r_copy['history']['train_acc'][-1] if r_copy['history']['train_acc'] else None,
                    'final_val_acc': r_copy['history']['val_acc'][-1] if r_copy['history']['val_acc'] else None,
                }
                del r_copy['history']
            results_for_json.append(r_copy)

        with open(results_file, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        print(f"\nRisultati salvati: {results_file}")

    # Grafici di confronto
    comparison_path = os.path.join(results_base_dir, 'comparison.png')
    plot_comparison(results, comparison_path)

    # Riepilogo
    print_summary(results)

    return results


def run_quick_test():
    """Esegue un test rapido con poche epoche per verificare il funzionamento."""
    quick_experiments = [
        {
            "name": "quick_custom",
            "description": "Test rapido Custom CNN",
            "config_overrides": {
                "training": {"model_type": "custom", "epochs": 2},
                "early_stopping": {"enabled": False}
            }
        },
        {
            "name": "quick_resnet",
            "description": "Test rapido ResNet18",
            "config_overrides": {
                "training": {"model_type": "resnet", "epochs": 2},
                "early_stopping": {"enabled": False}
            }
        }
    ]
    return run_all_experiments(quick_experiments)


def run_professor_experiments():
    """
    Esegue gli esperimenti richiesti dal professore per il report.
    Questi esperimenti sono progettati per analizzare l'impatto di:
    - Numero di epoche (2, 5, 10)
    - Batch size (8, 16, 32, 64)
    - Learning rate (0.0001, 0.001, 0.005, 0.01)
    """
    print("\n" + "#"*70)
    print("# ESPERIMENTI RICHIESTI DAL PROFESSORE")
    print("# TensorBoard abilitato per tutti gli esperimenti")
    print("#"*70)
    print("\nTabella esperimenti:")
    print("-"*50)
    print(f"{'Nome':<20} {'BS':<6} {'LR':<10} {'Epochs':<8}")
    print("-"*50)
    for exp in EXPERIMENTS_PROFESSOR:
        cfg = exp['config_overrides']['training']
        print(f"{exp['name']:<20} {cfg['batch_size']:<6} {cfg['learning_rate']:<10} {cfg['epochs']:<8}")
    print("-"*50)
    print(f"\nTotale: {len(EXPERIMENTS_PROFESSOR)} esperimenti")
    print("\nAvvio in corso...\n")

    return run_all_experiments(EXPERIMENTS_PROFESSOR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Esegui esperimenti multipli')
    parser.add_argument('--quick', action='store_true',
                        help='Esegue un test rapido (2 epoche)')
    parser.add_argument('--all', action='store_true',
                        help='Esegue tutti gli esperimenti predefiniti')
    parser.add_argument('--professor', action='store_true',
                        help='Esegue gli esperimenti richiesti dal professore (9 esperimenti con TensorBoard)')
    args = parser.parse_args()

    if args.quick:
        print("Modalità test rapido (2 epoche per modello)")
        run_quick_test()
    elif args.professor:
        print("Esecuzione esperimenti richiesti dal professore")
        run_professor_experiments()
    elif args.all:
        print("Esecuzione di tutti gli esperimenti predefiniti")
        run_all_experiments()
    else:
        print("Uso: python run_experiments.py [--quick | --all | --professor]")
        print("\n--quick:     Test rapido (2 epoche)")
        print("--all:       Tutti gli esperimenti predefiniti")
        print("--professor: Esperimenti richiesti dal professore (9 esperimenti)")
