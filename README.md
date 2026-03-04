# ACFA: Tumor-guided Attention Fusion for Breast Cancer Characterization from DCE MRI

This repository implements **ACFA**, a fully automated pipeline for breast cancer **molecular subtype** and **tumor aggressiveness** classification from **dynamic contrast enhanced (DCE) MRI**. ACFA combines anatomy-aware segmentation priors, a tumor-focused multi-parametric representation, self-supervised Vision Transformer embeddings, and attention-based fusion with tumor morphology features.

## Key idea

Given a multi-parametric DCE MRI study  
`\[
\mathcal{X}=\{X^{pre},\,X^{post},\,X^{sub}\}, \quad X^{sub}=X^{post}-X^{pre},
\]`
ACFA predicts a label `\(y\in\{1,\dots,C\}\)` by learning complementary signals from:
- a **VISUAL branch** based on **DINOv2** embeddings computed from a tumor-focused 3-channel aggregated image, and
- an **explicit morphology branch** computed from the 3D tumor mask,  
then combining them with a lightweight **attention-based fusion module**.

## Repository structure

Each pipeline stage is implemented and documented in its own folder:

- `Breast Delineation/`  
  Unilateral breast segmentation and laterality selection (left vs right breast ROI).
- `Tumor Segmentation/`  
  Tumor mask inference using pretrained nnU-Net models, plus post-processing.
- `Morphological Extraction/`  
  3D tumor mask–based geometry and radiomics-style morphology features.
- `VISUAL Branch/`  
  Tumor-focused multiparametric representation (slice selection + aggregation + 3-channel stack) and DINOv2 embedding/fine-tuning.
- `Fusion_Module/`  
  Token-based attention fusion of (visual embedding, morphology vector) and final classification.

Each folder contains its own `README.md` with detailed instructions for running that stage.

## End-to-end pipeline

### Step 0: Inputs

**Required per case**
- Pre-contrast DCE MRI: `X_pre`
- Post-contrast DCE MRI: `X_post`
- Subtraction volume: `X_sub = X_post - X_pre`

**Assumption**
- Volumes are spatially aligned within each study.

---

### Step 1: Breast delineation (anatomical focus)

Run unilateral breast segmentation to obtain:
- `B_hat ∈ {0,1}^{H×W×S}`

Used to suppress non-breast anatomy and define unilateral ROI.

Go to: `Breast Delineation/`

---

### Step 2: Tumor segmentation (lesion localization)

Run tumor segmentation to obtain:
- `T_hat ∈ {0,1}^{H×W×S}`

Used for laterality selection, tumor-slice selection, and morphology extraction.

Go to: `Tumor Segmentation/`

---

### Step 3: Tumor side localization and tumor-slice selection

Define tumor-containing slices:
\[
\mathcal{S}=\{s:\sum_{x,y}T\_hat(x,y,s) > 0\}.
\]

**Design choice:** we keep full unilateral breast context and do not crop a tight tumor bounding box.

---

### Step 4: Morphological feature extraction

From `T_hat`, extract an interpretable vector including:
- volume, surface area, max diameter
- principal axis lengths, elongation, flatness
- surface-to-volume ratio, sphericity

Output: `r ∈ ℝ^{d_r}`

Go to: `Morphological Extraction/`

---

### Step 5: VISUAL branch (tumor-focused representation + DINOv2)

Construct a compact 3-channel representation:
\[
I = concat(I^{pre}, I^{post}, I^{sub}) \in \mathbb{R}^{H\times W\times 3}.
\]

Extract global visual embedding:
\[
v = g_{\theta}(I),
\]
where \(g_{\theta}\) is a DINOv2 backbone.

**Fine-tuning**
- default: end-to-end fine-tuning  
- small-data: freeze early transformer blocks, fine-tune later blocks + head

Output: `v ∈ ℝ^{d_v}`

Go to: `VISUAL Branch/`

---

### Step 6: Fusion module (attention-based integration)

Project and fuse:
- `h_v = ψ_v(v)`
- `h_r = ψ_r(r)`

Form tokens `[CLS], h_v, h_r`, apply self-attention blocks, classify from the final CLS token.

Go to: `Fusion_Module/`

---

## Limitations

- Excludes multi-centric disease (multiple malignant lesions in different quadrants).
- Scanner variability and low-quality scans may still affect appearance-driven features.

## Citation
......Under-review.......
```bibtex
@inproceedings{tugaf2026,
  title={ACFA: Tumor-guided Attention Fusion for Breast Cancer Characterization from DCE MRI},
  author={...},
  booktitle={...},
  year={2026}
}
