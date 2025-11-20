Progetto d'Esame: Classificazione Segnali Stradali (GTSRB)

Studente: [Tuo Nome Cognome]

Matricola: [Tua Matricola]

Corso: [Nome del Corso]

Anno Accademico: 2024/2025

📌 Descrizione del Progetto

Questo progetto implementa una pipeline di Deep Learning per la classificazione di segnali stradali utilizzando il dataset GTSRB (German Traffic Sign Recognition Benchmark).

L'obiettivo principale è il confronto statistico tra due architetture diverse per valutare l'impatto del Transfer Learning rispetto a un addestramento da zero:

Custom CNN: Una rete convoluzionale leggera sviluppata ad-hoc (Baseline).

ResNet18: Una rete profonda pre-addestrata su ImageNet e adattata tramite Fine-Tuning.

Il progetto soddisfa i requisiti d'esame includendo:

Analisi statistica del dataset (distribuzione classi).

Data Augmentation per mitigare lo sbilanciamento.

Valutazione tramite Matrice di Confusione e Report di Classificazione.

Interfaccia demo per testare il modello su singole immagini.

📂 Struttura dei File

Progetto_Esame/
│
├── data/                   # Cartella di destinazione per il dataset (scaricato automaticamente)
├── models/                 # Cartella dove vengono salvati i pesi addestrati (.pth)
│   ├── best_model_resnet.pth
│   └── best_model_custom.pth
│
├── src/                    # Codice sorgente
│   ├── data_setup.py       # Download dataset e analisi statistica (grafici)
│   ├── models.py           # Definizione architetture (CustomCNN, TransferResNet)
│   ├── train.py            # Script di addestramento e validazione
│   ├── evaluate_detailed.py# Calcolo metriche avanzate e matrice di confusione
│   ├── predict_demo.py     # Demo inferenza su immagine casuale
│   └── make_graphs.py      # Script di utilità per rigenerare i grafici
│
├── main.py                 # Menu principale per eseguire il progetto
├── requirements.txt        # Dipendenze Python necessarie
├── Relazione_Progetto.pdf  # Documentazione dettagliata
└── Presentazione.html      # Slide interattive per la discussione


🚀 Installazione e Setup

Prerequisiti: Assicurarsi di avere Python installato (versione 3.8 o superiore consigliata).

Installazione Librerie: Eseguire il seguente comando nel terminale per installare tutte le dipendenze:

pip install -r requirements.txt


Nota: Il progetto utilizza torch, torchvision, matplotlib, pandas, scikit-learn e tqdm.

🖥️ Istruzioni per l'Esecuzione

Per avviare il progetto, utilizzare l'orchestratore principale che fornisce un menu interattivo:

python main.py


Funzionalità del Menu:

Analisi Dataset: Scarica il dataset GTSRB (se non presente) e mostra i grafici di distribuzione delle classi e una griglia di esempi.

Addestramento (Train): Avvia il training del modello selezionato per 10 epoche. Salva il modello migliore in models/.

Nota: Per cambiare il modello da addestrare, modificare la variabile MODEL_TYPE all'interno di src/train.py ("custom" o "resnet").

Valutazione: Carica il modello addestrato e genera la Matrice di Confusione sul test set.

Demo: Preleva un'immagine casuale dal test set, esegue la predizione e mostra il confronto visivo tra etichetta reale e predetta.

📊 Risultati Attesi

Architettura

Accuratezza Test

Note

Custom CNN

~85%

Modello leggero, soffre di leggero overfitting.

ResNet18

~97%

Convergenza rapida, elevata robustezza grazie al Transfer Learning.

I grafici di training (Loss/Accuracy) vengono salvati automaticamente come file .png nella cartella principale al termine dell'addestramento.

⚠️ Requisiti di Sicurezza

Si certifica di aver completato i moduli 1 e 2 sulla sicurezza nei luoghi di studio in modalità e-learning, come richiesto dal regolamento d'esame.

📚 Riferimenti

Dataset: GTSRB - German Traffic Sign Recognition Benchmark

PyTorch Documentation: https://pytorch.org/docs/

ResNet Paper: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
