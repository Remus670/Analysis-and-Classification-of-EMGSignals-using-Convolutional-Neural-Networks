# sEMG Gesture Recognition via Deep Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic-green.svg)]()

A Bachelor's Thesis project for hand gesture classification from 8-channel surface electromyography (sEMG) signals. The framework compares three signal processing paradigms (time-domain raw signals, engineered statistical features, and frequency-domain spectral analysis) and provides an interactive desktop GUI for real-time inference and signal inspection.

---

## Architecture Overview

The system classifies 6 distinct hand gestures using three specialized PyTorch architectures:

| Approach | Model | Input Representation | Architecture Summary |
|---|---|---|---|
| **`temporal`** | `EMGConvRecurrent` | Raw window `(8 × 150)` | 1D CNN (spatial-temporal features) + 2-layer Bidirectional LSTM (sequential dynamics) + Linear classifier |
| **`features`** | `EMGFeatureMLP` | Feature vector `(32)` | Time-domain features (RMS, MAV, VAR, ZCR per channel) + 2-layer MLP with BatchNorm & Dropout |
| **`spectral`** | `EMGSpectralCNN` | FFT magnitude `(8 × 75)` | Fast Fourier Transform ($|\text{FFT}|$) + 3-stage 1D CNN with Adaptive Average Pooling |

---

## Dataset & Preprocessing

- **Signal Source**: 8-channel sEMG sensor array ($f_s = 1000\text{ Hz}$).
- **Classes**: 6 hand gestures (`Gest 1` – `Gest 6`).
- **Sliding Window**: Window size $W = 150$ samples ($150\text{ ms}$), stride $S = 30$ samples ($80\%$ overlap). Window extraction strictly preserves segment boundaries.
- **Normalization**: Global Z-Score standardization calculated on the training partition and saved to [`results/preprocessing_emg.json`](src/results/preprocessing_emg.json).

---

## Benchmark Results

Evaluated on an independent validation set ($4,738$ windows):

| Model | Input Feature | Validation Accuracy | Key Strengths |
|---|---|:---:|---|
| **`EMGSpectralCNN`** | Frequency Magnitude ($|\text{FFT}|$) | **92.21%** | **Top accuracy**: cleanly separates gestures via muscle activation harmonics. |
| **`EMGFeatureMLP`** | Engineered Features (RMS, MAV, VAR, ZCR) | **90.50%** | **Lightweight & fast**: minimal latency, well-suited for embedded/edge devices. |
| **`EMGConvRecurrent`** | Raw Normalized Signals | **89.47%** | **End-to-end learning**: captures raw waveform patterns without manual feature engineering. |

Confusion matrices, training curves, and per-class metrics are automatically exported to [`src/results/`](src/results/).

---

## Project Structure

```text
proiect_licenta/
├── pyproject.toml              # Dependencies and project metadata
├── uv.lock                     # UV dependency lockfile
├── README.md                   # Project documentation
└── src/
    ├── config.py               # Hyperparameters and path configurations
    ├── dataset.py              # Data loading, Z-score, sliding window, FFT & RMS
    ├── train.py                # Model training, scheduler, checkpointing & plots
    ├── evaluate.py             # Validation evaluation & confusion matrix generator
    ├── predict.py              # Interactive Tkinter desktop GUI for inference
    ├── models/
    │   ├── models.py           # Model factory selector
    │   ├── conv_recurrent.py   # EMGConvRecurrent (CNN + BiLSTM)
    │   ├── feature_mlp.py      # EMGFeatureMLP (Statistical Features + MLP)
    │   └── spectral_cnn.py     # EMGSpectralCNN (Spectral 1D CNN)
    ├── data/                   # Train (80%), validation (10%), and test signals
    ├── saved_models/           # Saved PyTorch checkpoints (.pt)
    └── results/                # Normalization stats, evaluation CSVs, and plots
```

---

## Installation

### Prerequisites
- Python $\ge$ 3.10 (3.12 recommended)
- CUDA-enabled GPU (optional; falls back to CPU automatically)

### Using `uv` (Recommended)
```bash
uv sync
```

### Using standard `pip`
```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib seaborn scikit-learn pandas
```

---

## Usage

> **Note**: Execute commands from the `src/` directory so relative paths in [`config.py`](src/config.py) resolve correctly.

### 1. Training
Train any of the three models with learning rate scheduling (`ReduceLROnPlateau`) and checkpointing:
```bash
cd src
python train.py --feature spectral   # Options: temporal | features | spectral
```

### 2. Evaluation
Generate confusion matrices (CSV and PNG heatmap) and a complete classification report:
```bash
cd src
python evaluate.py --feature spectral
```

### 3. Interactive Desktop Application (GUI)
Launch the graphical interface for file prediction and signal analysis:
```bash
cd src
python predict.py
```

**GUI Features:**
- **Asynchronous Inference**: Runs non-blocking predictions on background threads.
- **Majority Voting**: Aggregates sliding window probabilities to predict the full signal's gesture with confidence scores.
- **Signal Visualizations**:
  - **RMS Plot**: Multi-channel envelope display over time.
  - **FFT Spectrum**: Frequency response ($0\text{–}250\text{ Hz}$) with automated dominant frequency peak detection per channel.

---

## Hyperparameters

Configured in [`src/config.py`](src/config.py):

| Parameter | Value | Description |
|---|---|---|
| `N_CHANNELS` | `8` | Number of sEMG channels |
| `N_CLASSES` | `6` | Target gesture classes |
| `WINDOW_SIZE` | `150` | Analysis window length in samples ($150\text{ ms}$) |
| `STRIDE` | `30` | Sliding window stride ($80\%$ overlap) |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `LR` | `1e-3` | Initial Adam learning rate |
| `EPOCHS` | `50` | Maximum training epochs |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |

---

## Authors & Academic Context

- **Project**: Bachelor's Thesis (*Lucrare de Licență*)
- **Core Stack**: PyTorch, NumPy, SciPy, Pandas, Scikit-learn, Matplotlib, Tkinter
