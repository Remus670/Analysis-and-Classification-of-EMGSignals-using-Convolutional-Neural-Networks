from pathlib import Path

TRAIN_CSV = Path("data/EMG_train_80.csv")
VALIDATION_CSV = Path("data/EMG_validation_10.csv")
TEST_SIGNALS_DIR = Path("data/test")

SAVED_MODELS_DIR = Path("saved_models")
RESULTS_DIR = Path("results")


CHANNEL_COLUMNS = [f"channel{i}" for i in range(1, 9)]
CLASS_COLUMN = "class"
LABEL_COLUMN = "label"
TIME_COLUMN = "time"

VALID_CLASSES = [1, 2, 3, 4, 5, 6]
CLASS_TO_INDEX = {class_id: idx for idx, class_id in enumerate(VALID_CLASSES)}
INDEX_TO_CLASS = {idx: class_id for class_id, idx in CLASS_TO_INDEX.items()}

N_CHANNELS = 8
N_CLASSES = 6

WINDOW_SIZE = 150
STRIDE = 30 #50 pentru antrenare cu raw

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
RANDOM_SEED = 42

PREPROCESSING_FILE = RESULTS_DIR / "preprocessing_emg.json"

CLASS_NAMES = {
    1: "Gest 1",
    2: "Gest 2",
    3: "Gest 3",
    4: "Gest 4",
    5: "Gest 5",
    6: "Gest 6",
}