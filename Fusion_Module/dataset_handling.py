import os
from typing import List
import pandas as pd
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    def __init__(self, x_emb: np.ndarray, x_morph: np.ndarray, y: Optional[np.ndarray] = None):
        self.x_emb = torch.tensor(x_emb, dtype=torch.float32)
        self.x_morph = torch.tensor(x_morph, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x_emb.shape[0])

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_emb[idx], self.x_morph[idx]
        return self.x_emb[idx], self.x_morph[idx], self.y[idx]
    
def smart_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep=";")
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep="\t")

    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if "patient_id" not in df.columns:
        raise ValueError(f"patient_id column not found in {path}. Columns: {list(df.columns)[:15]}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def drop_duplicate_patients(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df["patient_id"].duplicated().any():
        ex = df.loc[df["patient_id"].duplicated(), "patient_id"].astype(str).unique()[:5].tolist()
        print(f"{name}: duplicates found, examples: {ex}. Keeping first occurrence per patient_id.")
        df = df.drop_duplicates(subset=["patient_id"], keep="first")
    return df


def prepare_eval_dataframe(
    emb_csv: str,
    morph_csv: str,
    label_csv: str,
    label_col: str,
    emb_cols_expected: List[str],
    morph_cols_expected: List[str],
    impute: str = "drop",
) -> pd.DataFrame:
    emb = drop_duplicate_patients(smart_read_csv(emb_csv), "eval_embeddings")
    morph = drop_duplicate_patients(smart_read_csv(morph_csv), "eval_morphology")
    lab = drop_duplicate_patients(smart_read_csv(label_csv), "eval_labels")

    if label_col not in lab.columns:
        raise ValueError(f"Label column '{label_col}' not found in labels. Columns: {list(lab.columns)}")

    emb_feat_cols = [c for c in emb.columns if c != "patient_id"]
    morph_feat_cols = [c for c in morph.columns if c != "patient_id"]

    emb = emb.rename(columns={c: f"emb_{c}" for c in emb_feat_cols})
    morph = morph.rename(columns={c: f"morph_{c}" for c in morph_feat_cols})

    df = emb.merge(morph, on="patient_id", how="inner")
    df = df.merge(lab[["patient_id", label_col]], on="patient_id", how="inner")

    for c in df.columns:
        if c.startswith("emb_") or c.startswith("morph_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing_emb = [c for c in emb_cols_expected if c not in df.columns]
    missing_morph = [c for c in morph_cols_expected if c not in df.columns]
    if missing_emb or missing_morph:
        raise ValueError(
            "Feature mismatch versus checkpoint.\n"
            f"Missing emb cols: {missing_emb[:10]}{' ...' if len(missing_emb) > 10 else ''}\n"
            f"Missing morph cols: {missing_morph[:10]}{' ...' if len(missing_morph) > 10 else ''}"
        )

    keep_cols = ["patient_id"] + emb_cols_expected + morph_cols_expected + [label_col]
    df = df[keep_cols].copy()

    if impute == "drop":
        df = df.dropna(subset=emb_cols_expected + morph_cols_expected + [label_col]).reset_index(drop=True)
    elif impute == "mean":
        for c in emb_cols_expected + morph_cols_expected:
            df[c] = df[c].fillna(df[c].mean())
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
    elif impute == "zero":
        df[emb_cols_expected + morph_cols_expected] = df[emb_cols_expected + morph_cols_expected].fillna(0.0)
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
    else:
        raise ValueError("impute must be one of: drop, mean, zero")

    if len(df) == 0:
        raise ValueError("No rows remain after merge and cleaning.")

    return df