# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""VesselKnit -- voxelize vessel STL meshes into NIfTI masks without breakage.

Rasterizing a thin vessel STL onto a CT/MR grid usually fractures it into
disconnected blobs at narrow junctions. VesselKnit keeps the vessel a single
connected structure -- matching the original mesh in morphology and calibre --
via progressive gap-bridging during voxelization plus 26-/6-connectivity repair.

Quick start::

    from vesselknit import load_reference_image, stl_to_mask, save_nifti

    ref = load_reference_image("ct_scan/")          # DICOM dir or .nii.gz
    mask = stl_to_mask("vessel.stl", ref)           # (z, y, x) uint8
    save_nifti(mask, ref, "vessel.nii.gz")
"""

from ._version import __version__
from .core import stl_to_mask, combine_masks
from .bridge import bridge_vessel_26conn, bridge_vessel_6conn
from .io import load_reference_image, save_nifti

__all__ = [
    "stl_to_mask",
    "combine_masks",
    "bridge_vessel_26conn",
    "bridge_vessel_6conn",
    "load_reference_image",
    "save_nifti",
    "__version__",
]
