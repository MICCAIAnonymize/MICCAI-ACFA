#####import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Paths
MODEL_DIR  = r""
IMAGES_TR  = r""
LABELS_TR  = r""
OUT_ROOT   = r""

# Folds and checkpoint
FOLDS      = [0, 1, 2, 3, 4]
CHKPT_NAME = "checkpoint_final.pth"   # or "checkpoint_best.pth"

TUMOR_LABEL_IDS = None

NUM_PREPROC = 3
NUM_EXPORT  = 3
NUM_EVAL    = 4



# helpers
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def predict_one_fold(model_dir, images_dir, out_dir, fold, chkpt_name, npp=3, nexp=3):
    ensure_dir(out_dir)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, verbose=True)
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold,),
        checkpoint_name=chkpt_name
    )
    predictor.predict_from_files(
        images_dir,
        out_dir,
        save_probabilities=False,     # save hard label maps like the CLI
        overwrite=True,
        num_processes_preprocessing=npp,
        num_processes_segmentation_export=nexp
    )
    return out_dir

# run all folds
raw_dirs = {}
for f in FOLDS:
    out_raw = os.path.join(OUT_ROOT, f"fold_{f}", "raw_pred")
    print(f"Predicting fold {f} into {out_raw}")
    predict_one_fold(MODEL_DIR, IMAGES_TR, out_raw, f, CHKPT_NAME, NUM_PREPROC, NUM_EXPORT)
    raw_dirs[f] = out_raw

print("Predictions done")
