"""
utils.py - Funzioni di utilità e gestione configurazione centralizzata
"""
import json
import os
import torch

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.schema.json')


def load_config(config_path=None, validate=True):
    """
    Carica la configurazione dal file JSON.

    Args:
        config_path: Percorso del file di configurazione (opzionale)
        validate: Se True, valida la configurazione contro lo schema JSON

    Returns:
        dict: Configurazione caricata
    """
    path = config_path or CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if validate:
        validate_config(config)

    return config


def validate_config(config, schema_path=None):
    """
    Valida la configurazione contro lo schema JSON.

    Args:
        config: Dizionario di configurazione da validare
        schema_path: Percorso dello schema JSON (opzionale)

    Raises:
        jsonschema.ValidationError: Se la configurazione non è valida
        FileNotFoundError: Se lo schema non esiste
    """
    try:
        import jsonschema
    except ImportError:
        print("Warning: jsonschema non installato, validazione saltata. Installa con: pip install jsonschema")
        return

    path = schema_path or SCHEMA_PATH

    if not os.path.exists(path):
        print(f"Warning: Schema non trovato in {path}, validazione saltata")
        return

    with open(path, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=config, schema=schema)
        print("Configurazione valida secondo lo schema JSON")
    except jsonschema.ValidationError as e:
        print(f"ERRORE: Configurazione non valida!")
        print(f"  Campo: {'.'.join(str(p) for p in e.absolute_path)}")
        print(f"  Messaggio: {e.message}")
        raise


def get_device():
    """Ritorna il device disponibile (CUDA se presente, altrimenti CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Usando CPU")
    return device


def ensure_dirs(config):
    """Crea le directory necessarie se non esistono"""
    dirs = [
        config['paths']['models_dir'],
        config['paths']['results_dir'],
        config['paths']['logs_dir']
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_class_names(config):
    """Ritorna il dizionario delle classi"""
    return {int(k): v for k, v in config['classes'].items()}


class EarlyStopping:
    """
    Early stopping per terminare l'addestramento quando non ci sono miglioramenti.
    Supporta anche lo stop per raggiungimento di un target di accuratezza.
    """
    def __init__(self, patience=7, min_delta=0.001, mode='max', target=None, verbose=True):
        """
        Args:
            patience: Numero di epoche senza miglioramento prima di fermarsi
            min_delta: Miglioramento minimo per considerare un'epoca come miglioramento
            mode: 'max' per massimizzare (es. accuratezza), 'min' per minimizzare (es. loss)
            target: Valore target opzionale. Se raggiunto, termina subito
            verbose: Se True, stampa messaggi sullo stato
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target = target
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False

    def __call__(self, score):
        """
        Aggiorna lo stato dell'early stopping.

        Args:
            score: Valore corrente della metrica monitorata

        Returns:
            bool: True se bisogna continuare, False se bisogna fermarsi
        """
        # Controlla se il target è stato raggiunto
        if self.target is not None:
            if (self.mode == 'max' and score >= self.target) or \
               (self.mode == 'min' and score <= self.target):
                self.target_reached = True
                if self.verbose:
                    print(f"Target raggiunto! Score: {score:.4f} >= Target: {self.target}")
                return False

        if self.best_score is None:
            self.best_score = score
            return True

        # Calcola se c'è miglioramento
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epoche senza miglioramento")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping attivato! Miglior score: {self.best_score:.4f}")
                return False

        return True

    def reset(self):
        """Reset dello stato per nuovo training"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False


class CheckpointManager:
    """
    Gestisce il salvataggio e caricamento dei checkpoint per il training.
    """
    def __init__(self, models_dir, model_type, save_best=True, save_last=True):
        """
        Args:
            models_dir: Directory dove salvare i checkpoint
            model_type: Tipo di modello ('custom' o 'resnet')
            save_best: Se salvare il miglior modello
            save_last: Se salvare l'ultimo checkpoint
        """
        self.models_dir = models_dir
        self.model_type = model_type
        self.save_best = save_best
        self.save_last = save_last
        self.best_score = 0.0

        os.makedirs(models_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, score, history, is_best=False):
        """
        Salva un checkpoint completo.

        Args:
            model: Il modello da salvare
            optimizer: L'optimizer con il suo stato
            epoch: Epoca corrente
            score: Score corrente (accuratezza)
            history: Storico del training
            is_best: Se questo è il miglior modello finora
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'history': history
        }

        # Salva sempre l'ultimo checkpoint
        if self.save_last:
            last_path = os.path.join(self.models_dir, f'last_checkpoint_{self.model_type}.pth')
            torch.save(checkpoint, last_path)

        # Salva il miglior modello
        if is_best and self.save_best:
            self.best_score = score
            best_path = os.path.join(self.models_dir, f'best_model_{self.model_type}.pth')
            torch.save(checkpoint, best_path)
            print(f"Nuovo miglior modello salvato! Accuratezza: {score:.2f}%")

    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """
        Carica un checkpoint.

        Args:
            model: Modello in cui caricare i pesi
            optimizer: Optimizer in cui caricare lo stato (opzionale)
            checkpoint_path: Percorso specifico del checkpoint (opzionale)

        Returns:
            tuple: (start_epoch, history) o None se non esiste checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.models_dir, f'last_checkpoint_{self.model_type}.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Nessun checkpoint trovato in: {checkpoint_path}")
            return None

        print(f"Caricamento checkpoint da: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        })

        print(f"Checkpoint caricato! Ripresa da epoca {start_epoch}")
        print(f"Miglior score precedente: {checkpoint.get('score', 'N/A')}")

        return start_epoch, history

    def get_best_model_path(self):
        """Ritorna il percorso del miglior modello"""
        return os.path.join(self.models_dir, f'best_model_{self.model_type}.pth')

    def get_last_checkpoint_path(self):
        """Ritorna il percorso dell'ultimo checkpoint"""
        return os.path.join(self.models_dir, f'last_checkpoint_{self.model_type}.pth')


def format_time(seconds):
    """Formatta i secondi in ore:minuti:secondi"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
