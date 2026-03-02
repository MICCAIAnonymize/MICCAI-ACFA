from typing import Tuple

import SimpleITK as sitk
from radiomics import featureextractor


def make_extractor(
    mode: str = "shape_firstorder_glcm",
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 2.0),
    bin_width: int = 25,
) -> featureextractor.RadiomicsFeatureExtractor:
    """
    mode:
      - shape_only
      - shape_firstorder_glcm
    """
    settings = {
        "correctMask": True,
        "label": 1,
        "normalize": False,
        "removeOutliers": None,
        "binWidth": bin_width,
        "resampledPixelSpacing": list(target_spacing),
        "interpolator": sitk.sitkBSpline,
        "force2D": False,
    }

    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()

    if mode == "shape_only":
        ext.enableFeatureClassByName("shape")
    elif mode == "shape_firstorder_glcm":
        ext.enableFeatureClassByName("shape")
        ext.enableFeatureClassByName("firstorder")
        ext.enableFeatureClassByName("glcm")
    else:
        raise ValueError("mode must be 'shape_only' or 'shape_firstorder_glcm'")

    return ext