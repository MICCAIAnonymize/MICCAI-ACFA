
import os
from typing import Dict, Any

import pandas as pd

from utils import read_image
from utils import find_mask_case_insensitive, preprocess_mask_for_image, safe_float
from radiomic_extractor import make_extractor


def extract_one_patient(pdir: str, masks_root: str, mode: str = "shape_only") -> Dict[str, Any]:
    patient_id = os.path.basename(pdir)

    extractor = make_extractor(mode=mode)

    # 1) Locate pre and post scans inside the patient folder
    pre_path, post_path = find_pre_post_in_patient_folder(pdir)

    # 2) Locate the segmentation mask file for this patient
    mask_path = find_mask_case_insensitive(masks_root, patient_id)

    # 3) Load images
    pre_img = read_image(pre_path)
    post_img = read_image(post_path)

    # 4) Preprocess mask independently for each scan so it matches the scan grid
    mask_pre = preprocess_mask_for_image(mask_path, pre_img)
    mask_post = preprocess_mask_for_image(mask_path, post_img)

    # 5) Extract radiomics features
    pre_feats = extractor.execute(pre_img, mask_pre, label=1)
    post_feats = extractor.execute(post_img, mask_post, label=1)

    # 6) Build one row, diagnostics keys are removed
    row: Dict[str, Any] = {"patient_id": patient_id}

    for k, v in pre_feats.items():
        if str(k).startswith("diagnostics"):
            continue
        row["pre_" + str(k)] = safe_float(v)

    for k, v in post_feats.items():
        if str(k).startswith("diagnostics"):
            continue
        row["post_" + str(k)] = safe_float(v)

    return row


if __name__ == "__main__":
    # Update these paths to your data locations
    patient_dir = "/path/to/patients_root/Patient_001"
    masks_root = "/path/to/masks_root"

    row = extract_one_patient(patient_dir, masks_root, mode="shape_only")

    # Print the number of extracted features and show a few keys
    keys = [k for k in row.keys() if k != "patient_id"]
    print("patient_id:", row["patient_id"])
    print("num_features:", len(keys))
    print("example_keys:", keys[:10])

    # Optional, save as a one row CSV
    pd.DataFrame([row]).to_csv("one_patient_features.csv", index=False)

    print("Saved one_patient_features.csv")

