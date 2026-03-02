import os
import glob
from typing import Any
import numpy as np
import SimpleITK as sitk



def binarize_mask(mask: sitk.Image, threshold: float = 0.5) -> sitk.Image:
    mask_f = sitk.Cast(mask, sitk.sitkFloat32)
    mask_b = sitk.Cast(mask_f > threshold, sitk.sitkUInt8)
    return mask_b


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_mask: bool) -> sitk.Image:
    """
    Resample moving image into reference grid.
    Mask uses nearest neighbor, image uses BSpline.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    out = resampler.Execute(moving)
    if is_mask:
        out = sitk.Cast(out > 0, sitk.sitkUInt8)
    return out


def keep_largest_component(mask_bin: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask_bin)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    labels = list(stats.GetLabels())
    if len(labels) == 0:
        return mask_bin

    largest = max(labels, key=lambda l: stats.GetNumberOfPixels(l))
    out = sitk.Cast(cc == largest, sitk.sitkUInt8)
    return out


def fill_holes_3d(mask_bin: sitk.Image) -> sitk.Image:
    filled = sitk.BinaryFillhole(mask_bin)
    return sitk.Cast(filled > 0, sitk.sitkUInt8)


def preprocess_mask_for_image(
    mask_path: str,
    ref_img: sitk.Image,
    do_largest_cc: bool = True,
    do_fill_holes: bool = True,
) -> sitk.Image:
    """
    1) read mask
    2) binarize
    3) resample to reference image grid
    4) optional: keep largest connected component
    5) optional: fill holes
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    mask = sitk.ReadImage(mask_path)
    mask = binarize_mask(mask)
    mask = resample_to_reference(mask, ref_img, is_mask=True)

    if do_largest_cc:
        mask = keep_largest_component(mask)

    if do_fill_holes:
        mask = fill_holes_3d(mask)

    if int(sitk.GetArrayViewFromImage(mask).sum()) == 0:
        raise ValueError("Mask became empty after preprocessing")

    return mask


def safe_float(v: Any) -> float:
    """
    Pyradiomics sometimes returns numpy scalars. Convert safely.
    """
    if isinstance(v, (np.generic,)):
        return float(v)
    return float(v)