import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fft

import config


REQUIRED_COLUMNS = (
    [config.TIME_COLUMN]
    + config.CHANNEL_COLUMNS
    + [config.CLASS_COLUMN, config.LABEL_COLUMN]
)


def validate_csv_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} columns missing: {missing}")


def read_emg_csv(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    df = pd.read_csv(path)
    validate_csv_columns(df, path)

    df = df[df[config.CLASS_COLUMN].isin(config.VALID_CLASSES)].copy()
    if df.empty:
        raise ValueError(f"{path} does not contain valid classes {config.VALID_CLASSES}")

    return df


def fit_zscore(df: pd.DataFrame) -> dict:
    values = df[config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    if std_val == 0:
        raise ValueError(
            "Cannot standardize: standard deviation is 0 (constant signal)."
        )

    return {"mean": mean_val, "std": std_val}


def save_preprocessing(stats: dict, path=None) -> None:
    if path is None:
        path = config.PREPROCESSING_FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_preprocessing(path=None) -> dict:
    if path is None:
        path = config.PREPROCESSING_FILE
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}. Run train.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_channels(values: np.ndarray, stats: dict) -> np.ndarray:
    mean_val = float(stats["mean"])
    std_val = float(stats["std"])

    if std_val == 0:
        raise ValueError("Invalid normalization statistics: std == 0.")

    return ((values - mean_val) / std_val).astype(np.float32)


def _segment_ids(df: pd.DataFrame) -> np.ndarray:
    class_change = df[config.CLASS_COLUMN].ne(df[config.CLASS_COLUMN].shift())
    label_change = df[config.LABEL_COLUMN].ne(df[config.LABEL_COLUMN].shift())
    return (class_change | label_change).cumsum().to_numpy()


def dataframe_to_windows(
    df: pd.DataFrame,
    stats: dict,
    window_size: int = None,
    stride: int = None,
    keep_segments_complete: bool = True,
):
    
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if stride is None:
        stride = config.STRIDE

    X_list, y_list, meta = [], [], []
    segment_ids = _segment_ids(df)

    for seg_no, (_, seg) in enumerate(df.groupby(segment_ids, sort=False), start=1):
        class_values = seg[config.CLASS_COLUMN].unique()
        label_values = seg[config.LABEL_COLUMN].unique()

        if len(class_values) != 1 or len(label_values) != 1:
            if keep_segments_complete:
                raise ValueError(
                    "Segment invalid: contains multiple classes or labels."
                )
            continue

        class_id = int(class_values[0])
        if class_id not in config.CLASS_TO_INDEX:
            continue

        values = seg[config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
        values = normalize_channels(values, stats)

        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start : start + window_size]
            X_list.append(window.T)  # PyTorch Conv1d format: channels x time
            y_list.append(config.CLASS_TO_INDEX[class_id])
            meta.append(
                {
                    "segment_no": seg_no,
                    "class_original": class_id,
                    "label": int(label_values[0]),
                    "window_start_in_segment": start,
                }
            )

    if not X_list:
        raise ValueError(
            "No windows obtained. Check WINDOW_SIZE/STRIDE and CSV file."
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, meta


def load_csv_dataset(csv_path, feature="temporal", stats=None, fit_stats=False):
    df = read_emg_csv(csv_path)

    if fit_stats:
        stats = fit_zscore(df)
    elif stats is None:
        stats = load_preprocessing()

    X, y, meta = dataframe_to_windows(df, stats)
    X = apply_feature(X, feature)
    return X, y, meta, stats


def apply_feature(X: np.ndarray, feature: str) -> np.ndarray:
    if feature == "temporal":
        return X.astype(np.float32)
    if feature == "features":
        return compute_rms(X)
    if feature == "spectral":
        return compute_fft(X)
    raise ValueError("feature must be: temporal, features or spectral")


def compute_rms(X: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(X), axis=2, keepdims=True))
    mav = np.mean(np.abs(X), axis=2, keepdims=True)
    var = np.var(X, axis=2, keepdims=True)
    zcr = np.mean(np.diff(np.sign(X), axis=2) != 0, axis=2, keepdims=True)
    return np.concatenate([rms, mav, var, zcr], axis=2).astype(np.float32)


def compute_fft(X: np.ndarray) -> np.ndarray:
    half = X.shape[2] // 2
    fft_vals = np.abs(fft(X, axis=2))[:, :, :half]
    return fft_vals.astype(np.float32)


def class_distribution(y: np.ndarray) -> dict:
    out = {}
    for idx, count in zip(*np.unique(y, return_counts=True)):
        original_class = config.INDEX_TO_CLASS[int(idx)]
        out[int(original_class)] = int(count)
    return out
