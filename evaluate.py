import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

import config
from dataset import load_csv_dataset
from train import EMGDataset, get_model

def plot_confusion_matrix(cm, classes, feature, output_path):
    
    plt.figure(figsize=(10, 8))
    
     
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Număr de ferestre'})
    
    plt.title(f'Matrice de Confuzie (Metodă: {feature})', fontsize=14, pad=15)
    plt.ylabel('Clasă Reală', fontsize=12, labelpad=10)
    plt.xlabel('Clasă Prezisă', fontsize=12, labelpad=10)
    
     
    plt.tight_layout()
    
     
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Evaluare model EMG și generare Matrice de Confuzie")
    ap.add_argument("--val_csv", default=str(config.VALIDATION_CSV), help="Calea către fișierul de validare")
    ap.add_argument("--feature", default="temporal", choices=["temporal", "features", "spectral"], help="Tipul de feature folosit")
    ap.add_argument("--model", default=None, help="Calea către model")
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE, help="Dimensiunea batch-ului")
    args = ap.parse_args()

   
    config.RESULTS_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device} | Feature: {args.feature}")

     
    X_val, y_val, _, stats = load_csv_dataset(args.val_csv, feature=args.feature)
    val_loader = DataLoader(EMGDataset(X_val, y_val), batch_size=args.batch, shuffle=False)

     
    model = get_model(args.feature, device)
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = config.SAVED_MODELS_DIR / f"emgnet_{args.feature}_best.pt"
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

     
    cm = confusion_matrix(all_labels, all_preds)
     
    class_names = [config.CLASS_NAMES[config.INDEX_TO_CLASS[i]] for i in range(config.N_CLASSES)]

     
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    csv_path = config.RESULTS_DIR / f"validation_{args.feature}_confusion_matrix.csv"
    df_cm.to_csv(csv_path)

     
    png_path = config.RESULTS_DIR / f"confusion_emgnet_{args.feature}.png"
    plot_confusion_matrix(cm, class_names, args.feature.upper(), png_path)

    # 7. Afișare Raport
    print("\n" + "="*50)
    print("CLASIFICATION REPORT:")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n[DONE] Files generated in the 'results' folder:")
    print(f"  -> CSV Table: {csv_path}")
    print(f"  -> PNG Image: {png_path}")

if __name__ == "__main__":
    main()