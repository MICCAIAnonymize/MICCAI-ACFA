import os
import argparse
from typing import Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model import FusionMultiHeadAttentionClassifier, FusionDataset
from dataset_handling import prepare_eval_dataframe
from metrics import evaluate_with_labels
from utils import apply_standardizer, ensure_dir


def run_test(
    ckpt_path: str,
    emb_csv: str,
    morph_csv: str,
    label_csv: str,
    label_col: Optional[str] = None,
    batch_size: int = 64,
    impute: str = "drop",
    out_dir: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    emb_cols = ckpt["emb_cols"]
    morph_cols = ckpt["morph_cols"]
    morph_mean = ckpt["morph_mean"]
    morph_std = ckpt["morph_std"]

    ckpt_label_col = ckpt.get("label_col", "label")
    if label_col is None or str(label_col).strip() == "":
        label_col = ckpt_label_col

    label_classes = ckpt["label_classes"]
    num_classes = int(ckpt["num_classes"])
    class_names = [str(x) for x in label_classes]

    label_to_id: Dict[str, int] = {str(lbl): i for i, lbl in enumerate(label_classes)}

    model = FusionMultiHeadAttentionClassifier(
        emb_dim=int(ckpt["emb_dim"]),
        morph_dim=int(ckpt["morph_dim"]),
        num_classes=num_classes,
        d_model=int(ckpt["d_model"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)

    df = prepare_eval_dataframe(
        emb_csv=emb_csv,
        morph_csv=morph_csv,
        label_csv=label_csv,
        label_col=label_col,
        emb_cols_expected=emb_cols,
        morph_cols_expected=morph_cols,
        impute=impute,
    )

    y_str = df[label_col].astype(str).tolist()
    unknown = sorted(list({v for v in y_str if v not in label_to_id}))
    if unknown:
        raise ValueError(
            "Found labels in test set that were not present in training.\n"
            f"Unknown labels examples: {unknown[:10]}"
        )

    y = np.array([label_to_id[v] for v in y_str], dtype=np.int64)

    x_emb = df[emb_cols].to_numpy(dtype=np.float32)
    x_morph = df[morph_cols].to_numpy(dtype=np.float32)
    x_morph = apply_standardizer(x_morph, np.array(morph_mean), np.array(morph_std))

    loader = DataLoader(
        FusionDataset(x_emb, x_morph, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    acc, macro_f1, auc, y_true, y_pred = evaluate_with_labels(model, loader, device, num_classes=num_classes)

    print("\nEvaluation results")
    if auc is None:
        print(f"acc {acc:.4f}  macro_f1 {macro_f1:.4f}")
    else:
        print(f"acc {acc:.4f}  macro_f1 {macro_f1:.4f}  auc {auc:.4f}")

    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))

    if out_dir:
        ensure_dir(out_dir)

        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

        rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
        pd.DataFrame(rep).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))

        summ = {"acc": acc, "macro_f1": macro_f1, "auc": auc if auc is not None else ""}
        pd.DataFrame([summ]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

        pred_df = pd.DataFrame(
            {
                "patient_id": df["patient_id"].astype(str).tolist(),
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "true_name": [class_names[i] for i in y_true.tolist()],
                "pred_name": [class_names[i] for i in y_pred.tolist()],
            }
        )
        pred_df.to_csv(os.path.join(out_dir, "per_patient_predictions.csv"), index=False)

        print("\nSaved outputs to:", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained fusion attention model on a new dataset.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best checkpoint .pt file")
    parser.add_argument("--DINO_EMB", type=str, required=True, help="Embeddings CSV with patient_id column")
    parser.add_argument("--tumor_morphology", type=str, required=True, help="Morphology CSV with patient_id column")
    parser.add_argument("--labels", type=str, required=True, help="Labels CSV with patient_id column")
    parser.add_argument("--label_col", type=str, default=None, help="Label column name in labels CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--impute", type=str, default="drop", choices=["drop", "mean", "zero"])
    parser.add_argument("--out_dir", type=str, default="test_reports")

    args = parser.parse_args()

    run_test(
        ckpt_path=args.ckpt,
        emb_csv=args.DINO_EMB,
        morph_csv=args.tumor_morphology,
        label_csv=args.labels,
        label_col=args.label_col,
        batch_size=args.batch_size,
        impute=args.impute,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()