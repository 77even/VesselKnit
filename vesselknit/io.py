# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
I/O utilities for loading reference images and saving NIfTI files.

Supports:
- DICOM series loading (GDCM + pydicom fallback)
- NIfTI file loading
- NIfTI saving with spatial metadata preservation
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

logger = logging.getLogger(__name__)


def load_reference_image(
    path: Union[str, Path],
) -> sitk.Image:
    """
    Load a reference image from a DICOM directory or NIfTI file.

    Automatically detects the input type:
    - Directory → DICOM series (uses read_dicom_series)
    - .nii / .nii.gz file → NIfTI (uses SimpleITK)

    Args:
        path: Path to DICOM directory or NIfTI file

    Returns:
        SimpleITK Image with full spatial metadata

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If path type is not recognized or loading fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_dir():
        return _load_dicom(path)
    elif path.is_file() and (
        path.suffix == ".nii" or path.name.lower().endswith(".nii.gz")
    ):
        return _load_nifti(path)
    else:
        raise ValueError(
            f"Unrecognized path type: {path}. "
            "Expected a DICOM directory or a .nii/.nii.gz file."
        )


def save_nifti(
    mask_array: np.ndarray,
    reference_image: sitk.Image,
    output_path: Union[str, Path],
) -> None:
    """
    Save a numpy mask array as NIfTI with spatial metadata from a reference image.

    Args:
        mask_array: numpy array with shape (z, y, x). This matches the
                    convention returned by ``stl_to_mask`` and SimpleITK's
                    ``GetArrayFromImage``.
        reference_image: SimpleITK image providing spatial metadata
                        (origin, spacing, direction)
        output_path: Output file path (.nii or .nii.gz)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create SimpleITK image from array
    # numpy (z, y, x) → SimpleITK (x, y, z) via GetImageFromArray
    img = sitk.GetImageFromArray(mask_array)

    # Copy spatial metadata from reference
    img.SetOrigin(reference_image.GetOrigin())
    img.SetSpacing(reference_image.GetSpacing())
    img.SetDirection(reference_image.GetDirection())

    # Use compression for .nii.gz
    use_compression = str(output_path).endswith(".gz")

    sitk.WriteImage(img, str(output_path), use_compression)
    logger.info(f"Saved NIfTI: {output_path} (shape={mask_array.shape}, "
                f"dtype={mask_array.dtype})")


def _load_nifti(path: Path) -> sitk.Image:
    """Load a NIfTI file via SimpleITK."""
    image = sitk.ReadImage(str(path))
    logger.info(
        f"Loaded NIfTI: {path.name}, size={image.GetSize()}, "
        f"spacing=[{image.GetSpacing()[0]:.4f}, "
        f"{image.GetSpacing()[1]:.4f}, {image.GetSpacing()[2]:.4f}]"
    )
    return image


def _load_dicom(dicom_dir: Path) -> sitk.Image:
    """
    Read a DICOM series from a directory, returning a SimpleITK image.

    Preserves:
    - Original HU (Hounsfield Unit) values via RescaleSlope/Intercept
    - Full spatial information (origin, spacing, direction)
    - Original bit depth and data range

    Handles common issues:
    - GDCM series detection failures (falls back to pydicom-based sorting)
    - Non-standard file naming (numeric filenames → alphabetical misordering)
    - Nested directory structures
    - Duplicate slices at the same Z position
    - Incorrect Z-spacing from mis-detected series

    Args:
        dicom_dir: Path to directory containing DICOM files

    Returns:
        SimpleITK Image

    Raises:
        ValueError: If no valid DICOM series is found
    """
    reader = sitk.ImageSeriesReader()

    # Step 1: Find all directories containing .dcm files
    dcm_containing_dirs = []
    for root, dirs, files in os.walk(str(dicom_dir)):
        if any(f.lower().endswith(".dcm") for f in files):
            dcm_containing_dirs.append(root)

    if not dcm_containing_dirs:
        raise ValueError(f"No DICOM files found in: {dicom_dir}")

    # Step 2: Try GDCM series detection on each directory
    dicom_files = None
    for dcm_dir_path in dcm_containing_dirs:
        try:
            series_ids = reader.GetGDCMSeriesIDs(dcm_dir_path)
            if series_ids:
                best_series = max(series_ids, key=lambda sid: len(
                    reader.GetGDCMSeriesFileNames(dcm_dir_path, sid)
                ))
                candidate_files = reader.GetGDCMSeriesFileNames(dcm_dir_path, best_series)
                if len(candidate_files) > 1:
                    dicom_files = candidate_files
                    logger.debug(f"  GDCM found series with {len(dicom_files)} files")
                    break
        except RuntimeError:
            continue

    # Step 3: If GDCM failed, manually sort using pydicom
    if not dicom_files:
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM reading. "
                "Install with: pip install vesselknit[dicom]"
            )

        logger.debug("  GDCM series detection failed, falling back to pydicom sorting...")
        all_dcm_with_pos = []

        for dcm_dir_path in dcm_containing_dirs:
            for fname in os.listdir(dcm_dir_path):
                if not fname.lower().endswith(".dcm"):
                    continue
                fpath = os.path.join(dcm_dir_path, fname)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                    pos = getattr(ds, "ImagePositionPatient", None)
                    if pos is not None:
                        z_pos = float(pos[2])
                        all_dcm_with_pos.append((z_pos, fpath))
                    else:
                        inst = getattr(ds, "InstanceNumber", None)
                        sloc = getattr(ds, "SliceLocation", None)
                        if inst is not None:
                            sort_key = float(inst)
                        elif sloc is not None:
                            sort_key = float(sloc)
                        else:
                            sort_key = 0.0
                        all_dcm_with_pos.append((sort_key, fpath))
                except Exception:
                    continue

        if len(all_dcm_with_pos) < 2:
            raise ValueError(f"Not enough DICOM slices found in: {dicom_dir}")

        # Sort by Z position (ascending)
        all_dcm_with_pos.sort(key=lambda x: x[0])
        dicom_files = [fpath for _, fpath in all_dcm_with_pos]
        logger.debug(f"  pydicom sorted {len(dicom_files)} files")

    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {dicom_dir}")

    # Step 4: Deduplicate slices at the same Z position
    try:
        import pydicom
        dedup_map = {}
        files_without_pos = []
        files_with_pos_total = 0
        for fpath in dicom_files:
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                pos = getattr(ds, "ImagePositionPatient", None)
                if pos is not None:
                    files_with_pos_total += 1
                    z_key = round(float(pos[2]), 4)
                    if z_key not in dedup_map:
                        dedup_map[z_key] = fpath
                else:
                    files_without_pos.append(fpath)
            except Exception:
                # Unreadable — preserve original ordering by keeping the file
                files_without_pos.append(fpath)

        # Only apply dedup if it actually shrinks the IPP-bearing list
        if dedup_map and len(dedup_map) < files_with_pos_total:
            n_before = len(dicom_files)
            sorted_with_pos = [fpath for _, fpath in sorted(dedup_map.items())]
            dicom_files = sorted_with_pos + files_without_pos
            logger.info(f"  Deduplicated DICOM slices: {n_before} → {len(dicom_files)}")
    except ImportError:
        pass  # pydicom not available, skip dedup

    # Step 5: Read the series with SimpleITK
    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        image = reader.Execute()
    except RuntimeError as e:
        raise ValueError(f"Failed to read DICOM series from {dicom_dir}: {e}")

    # Step 6: Validate and fix spacing if needed
    spacing = list(image.GetSpacing())
    size = image.GetSize()

    if spacing[2] < 0.3 or spacing[2] > 10.0:
        logger.warning(f"  Suspicious Z spacing: {spacing[2]:.4f}mm, attempting recalculation...")
        try:
            import pydicom
            ds_first = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
            ds_last = pydicom.dcmread(dicom_files[-1], stop_before_pixels=True)
            z_first = float(ds_first.ImagePositionPatient[2])
            z_last = float(ds_last.ImagePositionPatient[2])
            n_slices = size[2]
            if n_slices > 1:
                corrected_z_spacing = abs(z_last - z_first) / (n_slices - 1)
                if 0.3 <= corrected_z_spacing <= 10.0:
                    logger.info(f"  Corrected Z spacing: {spacing[2]:.4f} → {corrected_z_spacing:.4f}mm")
                    spacing[2] = corrected_z_spacing
                    image.SetSpacing(spacing)
                    origin = list(image.GetOrigin())
                    origin[2] = min(z_first, z_last)
                    image.SetOrigin(origin)
        except Exception as e:
            logger.warning(f"  Could not recalculate spacing: {e}")

    # Convert to Int16 (standard for HU values)
    if image.GetPixelID() != sitk.sitkInt16:
        image = sitk.Cast(image, sitk.sitkInt16)

    logger.info(
        f"  DICOM loaded: size={image.GetSize()}, "
        f"spacing=[{image.GetSpacing()[0]:.4f}, "
        f"{image.GetSpacing()[1]:.4f}, {image.GetSpacing()[2]:.4f}], "
        f"origin=[{image.GetOrigin()[0]:.2f}, "
        f"{image.GetOrigin()[1]:.2f}, {image.GetOrigin()[2]:.2f}]"
    )

    return image
