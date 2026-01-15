# GTSRB - Traffic Sign Classification

Progetto per il corso di **Laboratorio di Ottimizzazione** - Classificazione di segnali stradali tedeschi mediante reti neurali convoluzionali.

## Descrizione

Il progetto implementa un sistema di classificazione automatica dei segnali stradali utilizzando il dataset **GTSRB** (German Traffic Sign Recognition Benchmark). Sono disponibili due architetture:

- **Custom CNN**: Rete convoluzionale personalizzata con 3 blocchi conv + classificatore FC
- **ResNet18**: Transfer learning da ImageNet con fine-tuning

## Dataset

- **Training set**: 39,209 immagini
- **Test set**: 12,630 immagini
- **Classi**: 43 tipi di segnali stradali
- **Dimensione**: 32x32 pixel (dopo resize)

Il dataset viene scaricato automaticamente al primo avvio.

## Requisiti

- Python 3.8+
- PyTorch 2.0+
- CUDA (opzionale, per accelerazione GPU)

## Installazione

```bash
# Clona o scarica il progetto
cd PROGETTO

# Installa le dipendenze
pip install -r requirements.txt
```

## Utilizzo

### Menu Interattivo

```bash
python main.py
```

Il menu principale permette di:
1. Scaricare i dati e visualizzare statistiche
2. Addestrare il modello (custom o resnet)
3. Valutare il modello con matrice di confusione
4. Demo interattiva con predizioni casuali
5. Confrontare esperimenti
6. Visualizzare metriche su TensorBoard

### Training Diretto

```bash
# Training con configurazione di default
python train.py

# Training con parametri personalizzati
python train.py --model custom --epochs 10 --lr 0.001 --batch-size 64

# Riprendere da checkpoint
python train.py --resume
```

### Valutazione

```bash
# Valutazione dettagliata con matrice di confusione
python evaluate_detailed.py --model custom
```

### Demo Predizioni

```bash
# Demo interattiva
python predict_demo.py --interactive

# Predizioni multiple
python predict_demo.py --num 8
```

### Esperimenti

```bash
# Esperimenti richiesti dal professore (9 configurazioni)
python run_experiments.py --professor

# Test rapido (2 epoche)
python run_experiments.py --quick
```

### TensorBoard

```bash
tensorboard --logdir=./runs
# Apri il browser su http://localhost:6006
```

## Struttura del Progetto

```
PROGETTO/
├── config.json              # Configurazione centralizzata
├── config.schema.json       # Schema di validazione JSON
├── requirements.txt         # Dipendenze Python
├── main.py                  # Menu principale interattivo
├── train.py                 # Script di training
├── models.py                # Definizione architetture (CustomCNN, ResNet18)
├── utils.py                 # Utilities (EarlyStopping, CheckpointManager)
├── data_setup.py            # Download e analisi dataset
├── evaluate_detailed.py     # Valutazione con metriche dettagliate
├── predict_demo.py          # Demo predizioni interattive
├── run_experiments.py       # Esecuzione esperimenti multipli
├── REPORT_TEMPLATE.md       # Template per il report
├── data/                    # Dataset (scaricato automaticamente)
├── models/                  # Checkpoint e modelli salvati
├── results/                 # Grafici e risultati
├── runs/                    # Log TensorBoard
└── experiments/             # Output esperimenti
```

## Configurazione

Il file `config.json` contiene tutti i parametri configurabili:

| Sezione | Parametri |
|---------|-----------|
| `training` | model_type, batch_size, learning_rate, epochs, optimizer |
| `early_stopping` | enabled, patience, target_accuracy |
| `augmentation` | rotation, brightness, contrast |
| `tensorboard` | enabled, log_interval |
| `checkpoint` | save_best, save_last, resume_from |

## Esperimenti del Professore

Il progetto include 9 esperimenti pre-configurati per analizzare:

| Gruppo | Parametro Variabile | Valori |
|--------|---------------------|--------|
| 1 | Epoche | 2, 5, 10 |
| 2 | Batch Size | 8, 16, 32, 64 |
| 3 | Learning Rate | 0.0001, 0.001, 0.005, 0.01 |

Esegui con: `python run_experiments.py --professor`

## Architettura Custom CNN

```
Input: 32x32x3 (RGB)
    ↓
Conv2d(3→32) → BatchNorm → ReLU → MaxPool(2x2)
    ↓
Conv2d(32→64) → BatchNorm → ReLU → MaxPool(2x2)
    ↓
Conv2d(64→128) → BatchNorm → ReLU → MaxPool(2x2)
    ↓
Flatten: 128 × 4 × 4 = 2048
    ↓
Linear(2048→512) → ReLU → Dropout(0.5)
    ↓
Linear(512→43) → Output
```

## Autori

- **Filippo Forcellini** - filippo.forcellini@studio.unibo.it
- **Manuel Ragazzini** - manuel.ragazzini3@studio.unibo.it

## Licenza

Progetto universitario - Laboratorio di Ottimizzazione
