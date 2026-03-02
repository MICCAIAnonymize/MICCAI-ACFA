# Fusion Multi Head Attention Classifier

A trained fusion model that combines two feature sources per patient:

1. DINO embeddings from images, stored in a CSV
2. Tumor morphology features, stored in a CSV

The evaluator loads a saved checkpoint, rebuilds the exact same model architecture, restores the exact feature column order used during training, applies the same morphology normalization statistics saved inside the checkpoint, then reports standard classification metrics.

## Module overview

### `model/model.py`

Defines the model architecture used during training:

- `AttnBlock`: a Transformer style block with Multi Head Attention, residual connections, LayerNorm, and an MLP.
- `FusionMultiHeadAttentionClassifier`: projects embeddings and morphology into a shared `d_model` space, builds a 3 token sequence:
  - `[CLS]` token
  - one token for embeddings
  - one token for morphology
  then applies `n_layers` attention blocks and uses the `[CLS]` output token for classification.

### `model/dataset.py`

- `FusionDataset`: PyTorch Dataset that provides `(x_emb, x_morph, y)` for evaluation.

### `model/metrics.py`

- `evaluate_with_labels`: runs inference, collects predictions and probabilities, computes accuracy, macro F1, and AUC where possible.

### `scripts/testing.py`

This is the command line entry point. It:

- Loads the checkpoint
- Recreates the model with the saved hyperparameters
- Loads the weights
- Loads and prepares the evaluation dataframe
- Standardizes morphology features using checkpoint statistics
- Evaluates and prints results
- Saves CSV reports to `out_dir`

## Input CSV format

All CSV files must include a `patient_id` column.

### Embeddings CSV

Example columns:

- `patient_id`
- `0`, `1`, `2`, ... or any other embedding feature column names

All non `patient_id` columns are treated as embedding features and will be renamed to `emb_<original_name>`.

### Morphology CSV

Example columns:

- `patient_id`
- `area`, `perimeter`, `volume`, ...

All non `patient_id` columns are treated as morphology features and will be renamed to `morph_<original_name>`.

### Labels CSV

Must contain:

- `patient_id`
- label column, for example, `subtypes`

The label column values must match the class names seen during training, because the mapping is taken from the checkpoint.

## Installation

Create a Python environment and install dependencies:

Example `requirements.txt`:

- torch
- numpy
- pandas
- scikit-learn

## Run evaluation

From the repository root:
```bash

  python scripts/test_trained_fusion_model.py
--ckpt best_Splited_model.pt
--DINO_EMB DINO_EMB_Subtypes.csv
--tumor_morphology tumor_morphology.csv
--labels labels.csv
--label_col subtypes
--batch_size 64
--impute drop
--out_dir test_reports

```



