# Report Esperimenti - GTSRB Traffic Sign Classification

**Corso:** Laboratorio di Ottimizzazione
**Studenti:** Filippo Forcellini, Manuel Ragazzini
**Data:** [Data]

---

## 1. Introduzione

### 1.1 Obiettivo del Progetto
Classificazione di segnali stradali tedeschi (GTSRB - German Traffic Sign Recognition Benchmark) utilizzando reti neurali convoluzionali.

### 1.2 Dataset
- **Training set:** 39,209 immagini
- **Test set:** 12,630 immagini
- **Classi:** 43 tipi di segnali stradali
- **Dimensione immagini:** 32x32 pixel (dopo resize)

### 1.3 Architettura del Modello (Custom CNN)
```
Input: 32x32x3 (RGB)
  |
Block 1: Conv2d(3->32) -> BatchNorm -> ReLU -> MaxPool(2x2)
  |
Block 2: Conv2d(32->64) -> BatchNorm -> ReLU -> MaxPool(2x2)
  |
Block 3: Conv2d(64->128) -> BatchNorm -> ReLU -> MaxPool(2x2)
  |
Flatten: 128 * 4 * 4 = 2048
  |
FC: Linear(2048, 512) -> ReLU -> Dropout(0.5)
  |
Output: Linear(512, 43)
```

---

## 2. Configurazione degli Esperimenti

### 2.1 Tabella Esperimenti

| # | Nome Esperimento | Batch Size | Learning Rate | Epoche |
|---|------------------|------------|---------------|--------|
| 1 | exp1_epochs2     | 64         | 0.001         | 2      |
| 2 | exp2_epochs5     | 64         | 0.001         | 5      |
| 3 | exp3_epochs10    | 64         | 0.001         | 10     |
| 4 | exp4_bs8         | 8          | 0.001         | 5      |
| 5 | exp5_bs16        | 16         | 0.001         | 5      |
| 6 | exp6_bs32        | 32         | 0.001         | 5      |
| 7 | exp7_lr0001      | 64         | 0.0001        | 5      |
| 8 | exp8_lr005       | 64         | 0.005         | 5      |
| 9 | exp9_lr01        | 64         | 0.01          | 5      |

### 2.2 Parametri Comuni
- **Optimizer:** Adam
- **Weight Decay:** 0.0001
- **Loss Function:** CrossEntropyLoss
- **Data Augmentation:** Rotazione (10°), ColorJitter (brightness=0.2, contrast=0.2)
- **Early Stopping:** Disabilitato (per completare tutte le epoche)
- **TensorBoard:** Abilitato

---

## 3. Risultati degli Esperimenti

### 3.1 Gruppo 1: Confronto Numero di Epoche
*(batch_size=64, learning_rate=0.001)*

| Esperimento    | Epoche | Train Acc (%) | Val Acc (%) | Tempo (min) |
|----------------|--------|---------------|-------------|-------------|
| exp1_epochs2   | 2      | [___]         | [___]       | [___]       |
| exp2_epochs5   | 5      | [___]         | [___]       | [___]       |
| exp3_epochs10  | 10     | [___]         | [___]       | [___]       |

**Osservazioni:**
[Inserire osservazioni sull'impatto del numero di epoche sulla convergenza e sull'accuratezza]

**Screenshot TensorBoard - Curve di apprendimento (epoche):**
[Inserire screenshot]

---

### 3.2 Gruppo 2: Confronto Batch Size
*(learning_rate=0.001, epochs=5)*

| Esperimento | Batch Size | Train Acc (%) | Val Acc (%) | Tempo (min) |
|-------------|------------|---------------|-------------|-------------|
| exp4_bs8    | 8          | [___]         | [___]       | [___]       |
| exp5_bs16   | 16         | [___]         | [___]       | [___]       |
| exp6_bs32   | 32         | [___]         | [___]       | [___]       |
| exp2_epochs5| 64         | [___]         | [___]       | [___]       |

**Osservazioni:**
[Inserire osservazioni sull'impatto del batch size sulla velocita di training e sulla generalizzazione]

**Screenshot TensorBoard - Curve di apprendimento (batch size):**
[Inserire screenshot]

---

### 3.3 Gruppo 3: Confronto Learning Rate
*(batch_size=64, epochs=5)*

| Esperimento  | Learning Rate | Train Acc (%) | Val Acc (%) | Tempo (min) |
|--------------|---------------|---------------|-------------|-------------|
| exp7_lr0001  | 0.0001        | [___]         | [___]       | [___]       |
| exp2_epochs5 | 0.001         | [___]         | [___]       | [___]       |
| exp8_lr005   | 0.005         | [___]         | [___]       | [___]       |
| exp9_lr01    | 0.01          | [___]         | [___]       | [___]       |

**Osservazioni:**
[Inserire osservazioni sull'impatto del learning rate sulla convergenza e stabilita del training]

**Screenshot TensorBoard - Curve di apprendimento (learning rate):**
[Inserire screenshot]

---

## 4. Analisi TensorBoard

### 4.1 Come Visualizzare i Grafici
```bash
# Avviare TensorBoard dalla cartella del progetto
tensorboard --logdir=./experiments/[TIMESTAMP]/runs

# Oppure dal menu principale
python main.py -> Opzione 6 (Avvia TensorBoard)
```

### 4.2 Metriche Visualizzate
- **Loss/train e Loss/val:** Andamento della loss durante il training
- **Accuracy/train e Accuracy/val:** Accuratezza su training e validation set
- **LearningRate:** Variazione del learning rate (se scheduler attivo)

### 4.3 Screenshot Comparativi

**Confronto Loss:**
[Inserire screenshot comparativo della loss per i diversi esperimenti]

**Confronto Accuracy:**
[Inserire screenshot comparativo dell'accuratezza per i diversi esperimenti]

---

## 5. Discussione e Conclusioni

### 5.1 Impatto del Numero di Epoche
[Discutere come il numero di epoche influenza:
- Convergenza del modello
- Rischio di overfitting
- Trade-off tra tempo di training e accuratezza]

### 5.2 Impatto del Batch Size
[Discutere come il batch size influenza:
- Velocita di convergenza
- Stabilita del gradiente
- Utilizzo della memoria GPU
- Capacita di generalizzazione]

### 5.3 Impatto del Learning Rate
[Discutere come il learning rate influenza:
- Velocita di apprendimento
- Stabilita del training (oscillazioni della loss)
- Convergenza verso minimi locali/globali]

### 5.4 Configurazione Ottimale
Basandoci sui risultati degli esperimenti, la configurazione ottimale risulta essere:
- **Batch Size:** [___]
- **Learning Rate:** [___]
- **Epoche:** [___]
- **Accuratezza raggiunta:** [___]%

### 5.5 Limitazioni e Lavori Futuri
[Discutere eventuali limitazioni e possibili miglioramenti:
- Data augmentation piu aggressiva
- Architetture piu profonde
- Transfer learning (ResNet18)
- Tecniche di regolarizzazione aggiuntive]

---

## 6. Appendice

### 6.1 Comandi Utilizzati

```bash
# Installazione dipendenze
pip install -r requirements.txt

# Esecuzione esperimenti del professore
python run_experiments.py --professor

# Visualizzazione TensorBoard
tensorboard --logdir=./experiments/[TIMESTAMP]/runs

# Training singolo con parametri personalizzati
python train.py --model custom --epochs 10 --lr 0.001 --batch-size 64

# Valutazione dettagliata
python evaluate_detailed.py --model custom
```

### 6.2 Struttura dei File di Output

```
experiments/
└── [TIMESTAMP]/
    ├── results.json              # Riepilogo risultati
    ├── comparison.png            # Grafico comparativo
    ├── exp1_epochs2/
    │   └── results_custom.png    # Curve training
    ├── exp2_epochs5/
    │   └── results_custom.png
    ├── ...
    └── runs/                     # Log TensorBoard
        ├── exp1_epochs2/
        ├── exp2_epochs5/
        └── ...
```

### 6.3 Configurazione Hardware
- **CPU:** [Modello]
- **GPU:** [Modello] / CPU only
- **RAM:** [GB]
- **Sistema Operativo:** [Windows/Linux/MacOS]

---

*Report generato per il corso di Laboratorio di Ottimizzazione*
