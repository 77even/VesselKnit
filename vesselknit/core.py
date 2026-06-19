# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Core voxelization: STL mesh → binary volumetric mask conversion.

This module provides the main `stl_to_mask` function that converts an STL
surface mesh to a binary volumetric mask aligned with a reference image.

Key features:
- Automatic coordinate space alignment (LPS/RAS, Z-reflection, direction matrix)
- VTK stencil-based voxelization with volumetric accuracy
- Progressive gap-bridging for thin tubular junctions
- Multi-label mask combination with priority ordering
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    raise ImportError("VTK is required. Install with: pip install vtk")

logger = logging.getLogger(__name__)


def stl_to_mask(
    stl_path: Union[str, Path],
    reference_image: sitk.Image,
    check_coordinate_space: bool = True,
    keep_largest_component: bool = False,
    bridge_radii: Tuple[int, ...] = (3, 5, 8, 12, 18),
    min_fragment_size: int = 50,
) -> Optional[np.ndarray]:
    """
    Convert an STL surface mesh to a binary volumetric mask.

    Uses VTK's vtkPolyDataToImageStencil for robust mesh-to-volume
    conversion, with automatic coordinate space alignment between the
    STL mesh and the reference image.

    Pipeline:
    1. Read STL mesh
    2. Optionally keep only the largest connected mesh component
    3. Check coordinate alignment (bounding box overlap)
    4. Handle non-identity direction matrices
    5. Stencil voxelization (interior filling)
    6. Gap-bridging for thin tubular junctions (if fragmented)
    7. Fragment cleanup

    Args:
        stl_path: Path to STL file
        reference_image: SimpleITK image defining the output volume geometry
                         (size, spacing, origin, direction)
        check_coordinate_space: Whether to verify coordinate overlap and
                                try fallbacks if needed (default: True)
        keep_largest_component: Keep only the largest connected mesh
                                component, filtering out annotation noise
                                (default: False)
        bridge_radii: Progressive dilation radii for gap-bridging.
                     Larger values bridge wider gaps but add more voxels.
                     Set to empty tuple to disable bridging.
                     (default: (3, 5, 8, 12, 18))
        min_fragment_size: Minimum component size (in voxels) to keep
                          after closing. Fragments smaller than this are
                          removed as stencil artifacts. (default: 50)

    Returns:
        Binary numpy mask (z, y, x) uint8, or None if conversion fails
    """
    stl_path = Path(stl_path)

    # ---- Read STL ----
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    polydata = reader.GetOutput()

    if polydata is None or polydata.GetNumberOfPoints() == 0:
        logger.warning(f"Empty or invalid STL: {stl_path.name}")
        return None

    # ---- Keep largest mesh component (optional) ----
    if keep_largest_component:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(polydata)
        conn.SetExtractionModeToAllRegions()
        conn.Update()
        n_regions = conn.GetNumberOfExtractedRegions()
        if n_regions > 1:
            conn.SetExtractionModeToLargestRegion()
            conn.Update()
            polydata = conn.GetOutput()
            logger.info(
                f"STL {stl_path.name}: filtered {n_regions} mesh "
                f"components → kept largest"
            )

    # ---- Get reference image geometry ----
    size = list(reference_image.GetSize())        # (x, y, z)
    spacing = list(reference_image.GetSpacing())   # (sx, sy, sz)
    origin = list(reference_image.GetOrigin())     # (ox, oy, oz)
    direction = np.array(reference_image.GetDirection()).reshape(3, 3)

    # ---- Compute physical bounds of the reference volume ----
    corners_idx = [
        [0, 0, 0],
        [size[0] - 1, 0, 0],
        [0, size[1] - 1, 0],
        [0, 0, size[2] - 1],
        [size[0] - 1, size[1] - 1, size[2] - 1],
    ]
    corners_phys = np.array([
        np.array(reference_image.TransformIndexToPhysicalPoint(
            [int(v) for v in c]
        ))
        for c in corners_idx
    ])
    vol_min = corners_phys.min(axis=0)
    vol_max = corners_phys.max(axis=0)

    # ---- Get STL bounding box ----
    stl_bounds = np.array(polydata.GetBounds())
    stl_min = np.array([stl_bounds[0], stl_bounds[2], stl_bounds[4]])
    stl_max = np.array([stl_bounds[1], stl_bounds[3], stl_bounds[5]])

    # ---- Coordinate space alignment ----
    if check_coordinate_space:
        polydata = _align_coordinate_space(
            polydata, stl_min, stl_max, vol_min, vol_max, origin, stl_path.name
        )
        if polydata is None:
            return None

    # ---- Handle non-identity direction matrix ----
    is_identity = np.allclose(direction, np.eye(3), atol=1e-4)

    if not is_identity:
        # Transform mesh from physical space to axis-aligned space
        inv_direction = np.linalg.inv(direction)

        mat4x4 = vtk.vtkMatrix4x4()
        mat4x4.Identity()
        for i in range(3):
            for j in range(3):
                mat4x4.SetElement(i, j, inv_direction[i, j])
            t = sum(inv_direction[i, j] * origin[j] for j in range(3))
            mat4x4.SetElement(i, 3, -t)

        transform = vtk.vtkTransform()
        transform.SetMatrix(mat4x4)

        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetInputData(polydata)
        transformer.SetTransform(transform)
        transformer.Update()
        polydata = transformer.GetOutput()

        stencil_origin = [0.0, 0.0, 0.0]
        stencil_spacing = list(spacing)
    else:
        stencil_origin = origin
        stencil_spacing = spacing

    # ---- Clean mesh (fill holes) ----
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(polydata)
    cleaner.Update()
    polydata = cleaner.GetOutput()

    # ---- VTK voxelization ----
    try:
        from scipy import ndimage as _ndi
        _struct26 = _ndi.generate_binary_structure(3, 3)
    except ImportError:
        raise ImportError("scipy is required for voxelization. Install with: pip install scipy")

    # Pass 1: Stencil (interior voxels)
    white_image = vtk.vtkImageData()
    white_image.SetDimensions(size)
    white_image.SetSpacing(stencil_spacing)
    white_image.SetOrigin(stencil_origin)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    n_points = white_image.GetNumberOfPoints()
    arr = vtk.vtkUnsignedCharArray()
    arr.SetNumberOfTuples(n_points)
    arr.FillComponent(0, 1)
    white_image.GetPointData().SetScalars(arr)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin(stencil_origin)
    pol2stenc.SetOutputSpacing(stencil_spacing)
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(white_image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    vtk_array = imgstenc.GetOutput().GetPointData().GetScalars()
    np_array = vtk_to_numpy(vtk_array)
    mask = np_array.reshape(size[2], size[1], size[0]).astype(np.uint8)

    # Pass 2: Targeted gap-bridging for thin junctions
    stencil_voxels = mask.sum()
    comp_labels, n_components = _ndi.label(mask, structure=_struct26)

    if stencil_voxels > 0 and n_components > 1 and bridge_radii:
        # Surface voxels from the subdivided mesh (mark candidate bridge sites)
        vox_z, vox_y, vox_x = _surface_voxels(
            polydata, size, stencil_origin, stencil_spacing
        )
        surf_mask = np.zeros(mask.shape, dtype=np.uint8)
        surf_mask[vox_z, vox_y, vox_x] = 1

        # Progressive bridge: increase radius until connected
        struct_dilate = _ndi.generate_binary_structure(3, 3)
        current_labels = comp_labels
        current_n = n_components
        total_bridge = 0

        for bridge_radius in bridge_radii:
            if current_n <= 1:
                break
            reach_count = np.zeros(mask.shape, dtype=np.int16)
            for cid in range(1, current_n + 1):
                dilated = _ndi.binary_dilation(
                    current_labels == cid,
                    structure=struct_dilate,
                    iterations=bridge_radius,
                )
                reach_count += dilated.astype(np.int16)
            interface_zone = reach_count >= 2

            bridge_mask = surf_mask & interface_zone & (~mask.astype(bool))
            bridge_added = int(bridge_mask.sum())
            total_bridge += bridge_added
            mask = np.maximum(mask, bridge_mask.astype(np.uint8))
            current_labels, current_n = _ndi.label(mask, structure=_struct26)

        logger.debug(
            f"STL {stl_path.name}: bridge {n_components} → {current_n} "
            f"components (+{total_bridge} voxels)"
        )

        # If still fragmented: close tiny gaps, remove small fragments
        if current_n > 1:
            closed = _ndi.binary_closing(mask, structure=_struct26, iterations=1)
            mask = closed.astype(np.uint8)
            frag_labels, frag_n = _ndi.label(mask, structure=_struct26)
            if frag_n > 1:
                comp_sizes = _ndi.sum(mask, frag_labels, range(1, frag_n + 1))
                keep = np.zeros(mask.shape, dtype=np.uint8)
                for cid, csz in enumerate(comp_sizes, 1):
                    if csz > min_fragment_size:
                        keep[frag_labels == cid] = 1
                if keep.sum() > 0:
                    mask = keep
                    _, frag_n = _ndi.label(mask, structure=_struct26)
            logger.debug(
                f"STL {stl_path.name}: closing → {frag_n} components"
            )

    elif stencil_voxels == 0:
        # Stencil missed entirely — fall back to surface vertex mapping
        vox_z, vox_y, vox_x = _surface_voxels(
            polydata, size, stencil_origin, stencil_spacing
        )
        mask[vox_z, vox_y, vox_x] = 1
        logger.debug(f"STL {stl_path.name}: stencil empty, used surface mapping")

    # ---- Final check ----
    voxel_count = np.sum(mask > 0)
    if voxel_count == 0:
        logger.warning(f"STL {stl_path.name}: mask is empty after voxelization.")
        return None

    logger.debug(f"STL {stl_path.name}: {voxel_count} voxels")
    return mask


def combine_masks(
    masks: Dict[str, np.ndarray],
    label_map: Dict[str, int],
    priority: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Combine multiple binary masks into a single multi-label volume.

    Lower priority labels are written first; higher priority labels
    overwrite them. This is useful when structures overlap (e.g., vessel
    segments overlap liver segments).

    Args:
        masks: Dict mapping label_name → binary mask array (z, y, x)
        label_map: Dict mapping label_name → integer label index
        priority: Write order (later overwrites earlier).
                  If None, uses dict insertion order.

    Returns:
        Multi-label numpy array (z, y, x) with uint8 values

    Example:
        >>> combined = combine_masks(
        ...     masks={"liver": liver_mask, "vessel": vessel_mask},
        ...     label_map={"liver": 1, "vessel": 2},
        ...     priority=["liver", "vessel"],  # vessel overwrites liver
        ... )
    """
    if not masks:
        raise ValueError("No masks provided")

    # Determine volume shape from first mask
    first_mask = next(iter(masks.values()))
    volume_shape = first_mask.shape
    combined = np.zeros(volume_shape, dtype=np.uint8)

    # Build write order
    if priority is not None:
        write_order = [name for name in priority if name in masks and name in label_map]
        # Add any masks not in priority list
        for name in masks:
            if name not in write_order and name in label_map:
                write_order.append(name)
    else:
        write_order = [name for name in masks if name in label_map]

    for label_name in write_order:
        mask = masks[label_name]
        label_idx = label_map[label_name]
        combined[mask > 0] = label_idx

    return combined


def _align_coordinate_space(
    polydata,
    stl_min: np.ndarray,
    stl_max: np.ndarray,
    vol_min: np.ndarray,
    vol_max: np.ndarray,
    origin: List[float],
    name: str,
):
    """
    Align an STL mesh to the reference volume's coordinate space.

    Returns the mesh as-is if it already overlaps the volume. Otherwise tries
    three coordinate-convention fallbacks — each applied to the *original* mesh
    — and returns the first that produces substantial overlap:
      1. LPS↔RAS flip (negate X, Y)
      2. Z-reflection through the DICOM origin
      3. RAS flip + Z-reflection combined

    Returns None if none of them overlap (caller should skip the STL).
    """
    if _check_substantial_overlap(stl_min, stl_max, vol_min, vol_max):
        return polydata

    logger.info(f"STL {name}: no substantial overlap, trying coordinate fallbacks...")

    def _ras_flip() -> "vtk.vtkTransform":
        t = vtk.vtkTransform()
        t.Scale(-1, -1, 1)
        return t

    def _z_reflect() -> "vtk.vtkTransform":
        t = vtk.vtkTransform()
        t.Translate(0, 0, 2 * origin[2])
        t.Scale(1, 1, -1)
        return t

    def _ras_and_z() -> "vtk.vtkTransform":
        t = vtk.vtkTransform()
        t.Translate(0, 0, 2 * origin[2])
        t.Scale(-1, -1, -1)
        return t

    original = polydata
    for desc, make_transform in (
        ("RAS flip", _ras_flip),
        ("Z-reflection", _z_reflect),
        ("RAS flip + Z-reflection", _ras_and_z),
    ):
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetInputData(original)
        transformer.SetTransform(make_transform())
        transformer.Update()
        candidate = transformer.GetOutput()

        bounds = np.array(candidate.GetBounds())
        c_min = np.array([bounds[0], bounds[2], bounds[4]])
        c_max = np.array([bounds[1], bounds[3], bounds[5]])
        if _check_substantial_overlap(c_min, c_max, vol_min, vol_max):
            logger.info(f"STL {name}: {desc} resolved coordinate mismatch.")
            return candidate

    logger.warning(
        f"STL {name}: no overlap after all fallbacks. "
        f"Vol bounds: {vol_min} - {vol_max}. Skipping."
    )
    return None


def _surface_voxels(
    polydata,
    size: List[int],
    stencil_origin: List[float],
    stencil_spacing: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map mesh surface vertices to clipped voxel indices ``(vz, vy, vx)``.

    The mesh is linearly subdivided twice first (denser vertex sampling) so the
    returned voxels trace the surface without holes. Used both to seed bridge
    candidates and to fall back to surface mapping when the stencil is empty.
    """
    surface_mesh = polydata
    subdivider = vtk.vtkLinearSubdivisionFilter()
    subdivider.SetInputData(polydata)
    subdivider.SetNumberOfSubdivisions(2)
    subdivider.Update()
    subdiv_out = subdivider.GetOutput()
    if (subdiv_out is not None
            and subdiv_out.GetNumberOfPoints() > polydata.GetNumberOfPoints()):
        surface_mesh = subdiv_out

    mesh_pts = vtk_to_numpy(surface_mesh.GetPoints().GetData())
    org = np.array(stencil_origin)
    sp = np.array(stencil_spacing)
    vox_x = np.clip(
        np.round((mesh_pts[:, 0] - org[0]) / sp[0]).astype(np.intp), 0, size[0] - 1
    )
    vox_y = np.clip(
        np.round((mesh_pts[:, 1] - org[1]) / sp[1]).astype(np.intp), 0, size[1] - 1
    )
    vox_z = np.clip(
        np.round((mesh_pts[:, 2] - org[2]) / sp[2]).astype(np.intp), 0, size[2] - 1
    )
    return vox_z, vox_y, vox_x


def _check_substantial_overlap(
    s_min: np.ndarray,
    s_max: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    margin: float = 10.0,
    min_overlap_ratio: float = 0.05,
) -> bool:
    """
    Check if STL bounding box substantially overlaps the volume bounding box.

    Two criteria:
    1. Bounding boxes are within `margin` mm of each other
    2. The actual overlap covers ≥ `min_overlap_ratio` of the STL extent
       in every dimension (prevents accepting marginal edge-touching overlaps)

    Args:
        s_min, s_max: STL bounding box min/max (3,)
        v_min, v_max: Volume bounding box min/max (3,)
        margin: Maximum gap (mm) between bounding boxes (default: 10.0)
        min_overlap_ratio: Minimum overlap as fraction of STL extent (default: 0.05)

    Returns:
        True if substantial overlap exists
    """
    if np.any(s_min > v_max + margin) or np.any(s_max < v_min - margin):
        return False
    overlap_lo = np.maximum(s_min, v_min)
    overlap_hi = np.minimum(s_max, v_max)
    overlap_ext = np.maximum(0, overlap_hi - overlap_lo)
    stl_ext = np.maximum(s_max - s_min, 1e-6)
    if np.any(overlap_ext / stl_ext < min_overlap_ratio):
        return False
    return True
