
#### Input files

Need two NIfTI files:

1. An image, `image.nii.gz`
2. A binary segmentation mask, `mask.nii.gz`

If the mask is not perfectly binary, the preprocessing will binarize it.

#### Example code

```bash
Python `extractor.py`:
```

Output: image_features.csv containing the extracted features for the image

By default, the extractor uses shape_only, which extracts only the Shape feature class.

To include intensity and texture features, change mode to shape_firstorder_glcm.

The mask is automatically resampled to match the image grid, so spacing differences between the image and mask are handled.
