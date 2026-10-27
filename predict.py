import json
import threading
import traceback
from pathlib import Path
from tkinter import Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.fft import fft
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import config

try:
    from models.models import get_model
except ImportError:
    from models import get_model


CHANNEL_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
]

BG = "#ffffff"
GRID_C = "#e0e0e0"
TEXT_C = "#1a1a1a"
SUB_C = "#555555"
FS = 1000

APP_BG = "#f4f6f8"
PANEL_BG = "#ffffff"
ACCENT = "#2f5d9f"
ACCENT_DARK = "#234578"
TEXT_MAIN = "#1a1a1a"
TEXT_SUB = "#5a5a5a"
OK_COLOR = "#1e8e3e"
ERR_COLOR = "#c5221f"

METHOD_INFO = {
    "temporal": {
        "label": "EMGConvRecurrent",
        "color": "#2f5d9f",
    },
    "features": {
        "label": "EMGFeatureMLP",
        "color": "#1f9d6b",
    },
    "spectral": {
        "label": "EMGSpectralCNN",
        "color": "#b5651d",
    },
}


 

def load_preprocessing(path=None) -> dict:
    if path is None:
        path = config.PREPROCESSING_FILE

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Lipsește fișierul de preprocesare: {path}. Rulează mai întâi train.py."
        )

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"Fișierul de preprocesare este invalid: {path}")

    return stats


def normalize_channels(values: np.ndarray, stats: dict) -> np.ndarray:
    mean_val = float(stats["mean"])
    std_val = float(stats["std"])

    if std_val == 0:
        raise ValueError("Statistici de normalizare invalide: std == 0.")

    return ((values - mean_val) / std_val).astype(np.float32)


def read_signal_csv_for_prediction(csv_path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fișierul nu există: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"Fișierul CSV este gol: {csv_path}")

    missing = [col for col in config.CHANNEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} nu are coloanele de semnal necesare: {missing}. "
            f"Sunt necesare: {config.CHANNEL_COLUMNS}"
        )

    for col in config.CHANNEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def infer_true_class_optional(df: pd.DataFrame):
    """Folosit doar pentru afișare, nu pentru predicție."""
    if config.CLASS_COLUMN not in df.columns:
        return None

    values = pd.to_numeric(df[config.CLASS_COLUMN], errors="coerce")
    values = values.dropna().astype(int)

    valid = values[values.isin(config.VALID_CLASSES)].unique()
    valid = sorted(int(v) for v in valid)

    if len(valid) == 1:
        return valid[0]

    return None


def dataframe_to_prediction_windows(
    df: pd.DataFrame,
    stats: dict,
    window_size: int = None,
    stride: int = None,
):
    if window_size is None:
        window_size = config.WINDOW_SIZE
    if stride is None:
        stride = config.STRIDE

    values = df[config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
    values = normalize_channels(values, stats)

    if len(values) < window_size:
        raise ValueError(
            f"Nu se pot obține ferestre: fișierul are {len(values)} eșantioane, "
            f"iar WINDOW_SIZE este {window_size}."
        )

    X_list = []
    meta = []

    for start in range(0, len(values) - window_size + 1, stride):
        window = values[start : start + window_size]
        X_list.append(window.T)  # shape: canale x timp
        meta.append({"window_start": int(start)})

    if not X_list:
        raise ValueError(
            "Nu s-au obținut ferestre. Verifică WINDOW_SIZE, STRIDE și fișierul CSV."
        )

    X = np.asarray(X_list, dtype=np.float32)
    return X, meta


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


def apply_feature(X: np.ndarray, feature: str) -> np.ndarray:
    if feature == "temporal":
        return X.astype(np.float32)

    if feature == "features":
        return compute_rms(X)

    if feature == "spectral":
        return compute_fft(X)

    raise ValueError("feature trebuie să fie: temporal, features sau spectral")


def load_csv_for_prediction(csv_path, feature="temporal", stats=None):
    df = read_signal_csv_for_prediction(csv_path)

    if stats is None:
        stats = load_preprocessing()

    true_class = infer_true_class_optional(df)

    X, meta = dataframe_to_prediction_windows(df, stats)
    X = apply_feature(X, feature)

    return X, meta, true_class

def _fig_style(fig):
    fig.patch.set_facecolor(BG)


def _ax_style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors=SUB_C, labelsize=9)

    for spine in ax.spines.values():
        spine.set_color(GRID_C)

    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--")

    if title:
        ax.set_title(title, color=TEXT_C, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=SUB_C)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=SUB_C)


def _suptitle(fig, l1, l2=None):
    fig.suptitle(
        f"{l1}\n{l2}" if l2 else l1,
        color=TEXT_C,
        fontsize=11,
        fontweight="bold",
        y=0.96,
    )


def _legend(ax, n_ch):
    ax.legend(
        ncol=n_ch,
        fontsize=8,
        loc="upper right",
        facecolor=BG,
        labelcolor=TEXT_C,
        edgecolor=GRID_C,
        framealpha=1,
    )


def _load_csv(csv_path, fs=1000):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    missing = [col for col in config.CHANNEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV file is missing channel columns: {missing}")

    for col in config.CHANNEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    data_mv = df[config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32) * 1000.0
    time_s = np.arange(len(df), dtype=np.float32) / float(fs)

    return data_mv, time_s, config.CHANNEL_COLUMNS


def plot_rms(csv_path: Path, pred_name: str, fs=1000, win_ms: int = 50) -> None:
    data_mv, time_s, ch_cols = _load_csv(csv_path, fs=fs)
    n_ch = len(ch_cols)

    window = max(1, int(round(win_ms / 1000.0 * fs)))

    fig, ax = plt.subplots(figsize=(14, 5))
    _fig_style(fig)
    _ax_style(ax, xlabel="Timp (s)", ylabel="RMS (mV)")

    for i in range(n_ch):
        color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
        sig = data_mv[:, i]
        rms = np.sqrt(uniform_filter1d(sig**2, size=window, mode="nearest"))

        ax.fill_between(time_s, rms, alpha=0.15, color=color)
        ax.plot(
            time_s,
            rms,
            color=color,
            linewidth=1.3,
            alpha=0.9,
            label=f"Canal {i + 1}",
        )

    ax.set_ylim(bottom=0)
    _legend(ax, n_ch)
    _suptitle(fig, "Grafic RMS", pred_name)
    fig.tight_layout(rect=[0, 0, 1, 0.88])


def plot_fft(csv_path: Path, pred_name: str, max_hz: int = 250) -> None:
    data_mv, _, ch_cols = _load_csv(csv_path)
    n_ch = len(ch_cols)
    N = len(data_mv)

    freqs = np.fft.rfftfreq(N, d=1 / FS)
    mask = freqs <= max_hz

    fig, ax = plt.subplots(figsize=(14, 5))
    _fig_style(fig)
    _ax_style(ax, xlabel="Frecvență (Hz)", ylabel="Amplitudine (mV)")

    for i in range(n_ch):
        color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]

        sig = data_mv[:, i] - np.mean(data_mv[:, i])
        mag = np.abs(np.fft.rfft(sig)) / N

        if len(mag) > 2:
            mag[1:-1] *= 2

        sm = uniform_filter1d(mag, size=9)

        ax.plot(
            freqs[mask],
            sm[mask],
            color=color,
            linewidth=1.1,
            alpha=0.85,
            label=f"Canal {i + 1}",
        )

        dom_idx = np.argmax(sm[mask])
        dom_f = freqs[mask][dom_idx]
        dom_amp = sm[mask][dom_idx]

        ax.axvline(dom_f, color=color, linewidth=0.5, alpha=0.4, linestyle=":")
        ax.annotate(
            f"{dom_f:.0f}Hz",
            xy=(dom_f, dom_amp),
            xytext=(dom_f + 3, dom_amp * 1.05),
            fontsize=6.5,
            color=color,
            alpha=0.9,
        )

    _legend(ax, n_ch)
    _suptitle(fig, "Grafic FFT", pred_name)
    fig.tight_layout(rect=[0, 0, 1, 0.88])


def load_model(feature, model_path, device):
    model = get_model(feature, device)

    if model_path is None:
        model_path = config.SAVED_MODELS_DIR / f"emgnet_{feature}_best.pt"

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelul nu există: {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, model_path


@torch.no_grad()
def predict_windows(model, X, device):
    x = torch.tensor(X, dtype=torch.float32).to(device)
    out = model(x)
    probs = nn.Softmax(dim=1)(out).cpu().numpy()

    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    return pred_idx, conf


def predict_file(csv_path, feature, model, device):
    original_csv_path = Path(csv_path)
    train_stats = load_preprocessing()

    X, meta, true_class = load_csv_for_prediction(
        original_csv_path,
        feature=feature,
        stats=train_stats,
    )

    pred_idx, conf = predict_windows(model, X, device)

    vote_counts = np.bincount(pred_idx, minlength=config.N_CLASSES)
    final_idx = int(vote_counts.argmax())
    final_class = int(config.INDEX_TO_CLASS[final_idx])

    majority_confidence = float(np.mean(pred_idx == final_idx))
    mean_softmax_confidence = float(np.mean(conf))

    return {
        "file": str(original_csv_path),
        "windows": int(len(X)),
        "true_class_original": true_class,
        "pred_class_original": final_class,
        "pred_name": config.CLASS_NAMES.get(final_class, str(final_class)),
        "majority_confidence": majority_confidence,
        "mean_softmax_confidence": mean_softmax_confidence,
        "correct": None if true_class is None else bool(true_class == final_class),
        "votes_by_original_class": {
            str(config.INDEX_TO_CLASS[i]): int(vote_counts[i])
            for i in range(config.N_CLASSES)
        },
    }


def show_result_window(parent, result: dict, feature: str):
    win = Toplevel(parent)
    win.title("Rezultat predicție")
    win.configure(bg=PANEL_BG)
    win.resizable(False, False)
    win.transient(parent)

    method = METHOD_INFO[feature]

    header = ttk.Frame(win, style="Header.TFrame")
    header.pack(fill="x")

    ttk.Label(
        header,
        text=f"Predicție metodă {method['label']}",
        style="HeaderTitle.TLabel",
    ).pack(anchor="w", padx=18, pady=(16, 2))

    ttk.Label(
        header,
        text=Path(result["file"]).name,
        style="HeaderSub.TLabel",
    ).pack(anchor="w", padx=18, pady=(0, 14))

    body = ttk.Frame(win, style="Body.TFrame", padding=18)
    body.pack(fill="both", expand=True)

    def row(label, value, value_color=TEXT_MAIN, bold=False):
        r = ttk.Frame(body, style="Body.TFrame")
        r.pack(fill="x", pady=3)

        ttk.Label(
            r,
            text=label,
            style="RowLabel.TLabel",
            width=22,
            anchor="w",
        ).pack(side="left")

        val_style = "RowValueBold.TLabel" if bold else "RowValue.TLabel"

        lbl = ttk.Label(
            r,
            text=str(value),
            style=val_style,
            foreground=value_color,
        )
        lbl.pack(side="left")

    true_text = (
        "necunoscută"
        if result["true_class_original"] is None
        else str(result["true_class_original"])
    )

    row("Clasă reală:", true_text)
    row(
        "Predicție:",
        f"clasa {result['pred_class_original']} · {result['pred_name']}",
        value_color=method["color"],
        bold=True,
    )
    row("Ferestre analizate:", result["windows"])
    row("Încredere vot majoritar:", f"{result['majority_confidence']:.2%}")
    row("Încredere softmax mediu:", f"{result['mean_softmax_confidence']:.2%}")

    if result["correct"] is not None:
        row(
            "Corect:",
            "DA" if result["correct"] else "NU",
            value_color=OK_COLOR if result["correct"] else ERR_COLOR,
            bold=True,
        )

    ttk.Separator(body).pack(fill="x", pady=(10, 8))

    ttk.Label(body, text="Voturi pe clase", style="SectionLabel.TLabel").pack(
        anchor="w", pady=(0, 6)
    )

    votes_frame = ttk.Frame(body, style="Body.TFrame")
    votes_frame.pack(fill="x")

    votes = result["votes_by_original_class"]
    max_votes = max(votes.values()) if votes else 1

    for cls, count in votes.items():
        vrow = ttk.Frame(votes_frame, style="Body.TFrame")
        vrow.pack(fill="x", pady=2)

        ttk.Label(
            vrow,
            text=f"Clasa {cls}",
            style="RowLabel.TLabel",
            width=10,
            anchor="w",
        ).pack(side="left")

        bar_bg = ttk.Frame(vrow, style="BarBg.TFrame", height=14, width=180)
        bar_bg.pack(side="left", padx=(4, 8))
        bar_bg.pack_propagate(False)

        ratio = count / max_votes if max_votes else 0
        fill_w = max(2, int(180 * ratio))
        is_pred = cls == str(result["pred_class_original"])

        bar = ttk.Frame(
            bar_bg,
            style="BarFillPred.TFrame" if is_pred else "BarFill.TFrame",
            height=14,
            width=fill_w,
        )
        bar.place(x=0, y=0)

        ttk.Label(vrow, text=str(count), style="RowValue.TLabel").pack(side="left")

    ttk.Separator(body).pack(fill="x", pady=(14, 10))

    btn_row = ttk.Frame(body, style="Body.TFrame")
    btn_row.pack(fill="x")

    ttk.Button(
        btn_row,
        text="Închide",
        style="Secondary.TButton",
        command=win.destroy,
    ).pack(side="right")

    win.update_idletasks()

    w, h = win.winfo_reqwidth(), win.winfo_reqheight()
    x = parent.winfo_x() + (parent.winfo_width() - w) // 2
    y = parent.winfo_y() + (parent.winfo_height() - h) // 2

    win.geometry(f"{w}x{h}+{max(x, 0)}+{max(y, 0)}")
    win.grab_set()


class EMGPredictApp:
    def __init__(self, root: Tk):
        self.root = root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_cache = {}
        self._busy = False

        self._setup_style()
        self._build_ui()

    def _get_model(self, feature: str):
        if feature not in self._models_cache:
            model, _ = load_model(feature, None, self.device)
            self._models_cache[feature] = model

        return self._models_cache[feature]

    def _setup_style(self):
        self.root.configure(bg=APP_BG)

        style = ttk.Style(self.root)

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Header.TFrame", background=ACCENT_DARK)
        style.configure(
            "HeaderTitle.TLabel",
            background=ACCENT_DARK,
            foreground="#ffffff",
            font=("Segoe UI", 13, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=ACCENT_DARK,
            foreground="#cfe0f5",
            font=("Segoe UI", 9),
        )

        style.configure("Body.TFrame", background=PANEL_BG)
        style.configure(
            "RowLabel.TLabel",
            background=PANEL_BG,
            foreground=TEXT_SUB,
            font=("Segoe UI", 10),
        )
        style.configure(
            "RowValue.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MAIN,
            font=("Segoe UI", 10),
        )
        style.configure(
            "RowValueBold.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "SectionLabel.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MAIN,
            font=("Segoe UI", 10, "bold"),
        )

        style.configure("BarBg.TFrame", background="#e9edf2")
        style.configure("BarFill.TFrame", background="#9fb6d4")
        style.configure("BarFillPred.TFrame", background=ACCENT)

        style.configure("Secondary.TButton", font=("Segoe UI", 9), padding=6)

        style.configure("Main.TFrame", background=APP_BG)
        style.configure("AppHeader.TFrame", background=ACCENT_DARK)
        style.configure(
            "AppTitle.TLabel",
            background=ACCENT_DARK,
            foreground="#ffffff",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "AppSub.TLabel",
            background=ACCENT_DARK,
            foreground="#cfe0f5",
            font=("Segoe UI", 9),
        )

    def _build_ui(self):
        self.root.title("Predicție")
        self.root.resizable(False, False)

        header = ttk.Frame(self.root, style="AppHeader.TFrame")
        header.pack(fill="x")

        ttk.Label(
            header,
            text="Clasificare Gesturi EMG",
            style="AppTitle.TLabel",
        ).pack(anchor="w", padx=20, pady=(18, 2))

        ttk.Label(
            header,
            text="Alege metoda de procesare a semnalului",
            style="AppSub.TLabel",
        ).pack(anchor="w", padx=20, pady=(0, 16))

        body = ttk.Frame(self.root, style="Main.TFrame", padding=20)
        body.pack(fill="both", expand=True)

        for feature in ("temporal", "features", "spectral"):
            self._build_method_card(body, feature)

        ttk.Frame(self.root, style="Main.TFrame", height=10).pack(fill="x")

    def _build_method_card(self, parent, feature):
        info = METHOD_INFO[feature]

        card = ttk.Frame(parent, style="Body.TFrame", padding=14)
        card.pack(fill="x", pady=6)
        card.configure(borderwidth=1, relief="solid")

        top = ttk.Frame(card, style="Body.TFrame")
        top.pack(fill="x")

        ttk.Label(
            top,
            text=info["label"],
            style="SectionLabel.TLabel",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")

        btn = ttk.Button(
            top,
            text="Alege fișier",
            command=lambda f=feature: self.on_method_clicked(f),
        )
        btn.pack(side="right")

        setattr(self, f"_btn_{feature}", btn)

    def on_method_clicked(self, feature: str):
        if self._busy:
            return

        csv_path = filedialog.askopenfilename(
            parent=self.root,
            title=f"Alege fișierul CSV pentru predicție ({feature.upper()})",
            filetypes=[("Fișiere CSV", "*.csv"), ("Toate fișierele", "*.*")],
        )

        if not csv_path:
            return

        self._set_busy(True)

        thread = threading.Thread(
            target=self._run_prediction,
            args=(feature, csv_path),
            daemon=True,
        )
        thread.start()

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = "disabled" if busy else "normal"

        for f in ("temporal", "features", "spectral"):
            getattr(self, f"_btn_{f}").configure(state=state)

    def _run_prediction(self, feature: str, csv_path: str):
        try:
            model = self._get_model(feature)
            result = predict_file(csv_path, feature, model, self.device)

        except Exception as exc:
            traceback.print_exc()
            error_message = str(exc)
            self.root.after(0, lambda msg=error_message: self._on_prediction_error(msg))
            return

        self.root.after(
            0,
            lambda: self._on_prediction_done(feature, csv_path, result),
        )

    def _on_prediction_done(self, feature: str, csv_path: str, result: dict):
        self._set_busy(False)

        if feature in ("features", "spectral"):
            try:
                if feature == "features":
                    plot_rms(Path(csv_path), result["pred_name"])
                else:
                    plot_fft(Path(csv_path), result["pred_name"])

                plt.show(block=False)

            except Exception:
                traceback.print_exc()

        show_result_window(self.root, result, feature)

    def _on_prediction_error(self, error_message: str):
        self._set_busy(False)
        messagebox.showerror("Error", f"Prediction failed:\n\n{error_message}")


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)

    root = Tk()
    EMGPredictApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
