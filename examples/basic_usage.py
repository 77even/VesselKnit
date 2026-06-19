#!/usr/bin/env python3
# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Basic usage examples for vesselknit.

This script demonstrates the core API:
1. Single STL → NIfTI conversion
2. Multi-STL → multi-label NIfTI with priority ordering
3. Vessel connectivity repair (26-conn and 6-conn bridging)

Usage:
    python basic_usage.py
"""

from pathlib import Path

# --- Example 1: Single STL → NIfTI ---

def example_single_stl():
    """Convert a single STL file to a NIfTI binary mask."""
    from vesselknit import load_reference_image, stl_to_mask, save_nifti

    # Load reference image (DICOM directory or NIfTI file)
    # The reference defines the output volume geometry (size, spacing, origin, direction)
    ref = load_reference_image("path/to/ct_scan.nii.gz")
    # Or: ref = load_reference_image("path/to/dicom_directory/")

    # Convert STL to binary mask
    mask = stl_to_mask(
        "path/to/vessel.stl",
        ref,
        check_coordinate_space=True,   # auto-detect LPS/RAS mismatch
        keep_largest_component=False,   # set True for organs (liver, spleen, etc.)
    )

    if mask is not None:
        # Save as NIfTI with spatial metadata from the reference image
        save_nifti(mask, ref, "vessel_mask.nii.gz")
        print(f"Saved vessel_mask.nii.gz ({mask.sum()} voxels)")
    else:
        print("Voxelization failed — check STL and reference image alignment")


# --- Example 2: Multiple STLs → Multi-label NIfTI ---

def example_multi_stl():
    """Convert multiple STL files to a single multi-label NIfTI."""
    from vesselknit import load_reference_image, stl_to_mask, combine_masks, save_nifti

    ref = load_reference_image("path/to/ct_scan.nii.gz")

    # Voxelize each STL individually
    pv_mask = stl_to_mask("path/to/portal_vein.stl", ref)
    hv_mask = stl_to_mask("path/to/hepatic_vein.stl", ref)
    ha_mask = stl_to_mask("path/to/hepatic_artery.stl", ref)

    masks = {}
    if pv_mask is not None:
        masks["portal_vein"] = pv_mask
    if hv_mask is not None:
        masks["hepatic_vein"] = hv_mask
    if ha_mask is not None:
        masks["hepatic_artery"] = ha_mask

    # Combine into multi-label volume
    # Priority: later labels overwrite earlier at overlap voxels
    combined = combine_masks(
        masks=masks,
        label_map={
            "portal_vein": 1,
            "hepatic_vein": 2,
            "hepatic_artery": 3,
        },
        priority=["hepatic_artery", "hepatic_vein", "portal_vein"],
    )

    save_nifti(combined, ref, "vessels_multi.nii.gz")
    print(f"Saved vessels_multi.nii.gz (labels: {set(combined.flat)})")


# --- Example 3: Vessel connectivity repair ---

def example_connectivity_repair():
    """Fix vessel fragmentation caused by label-priority overlap."""
    from vesselknit import (
        load_reference_image, stl_to_mask, combine_masks,
        bridge_vessel_26conn, bridge_vessel_6conn, save_nifti,
    )

    ref = load_reference_image("path/to/ct_scan.nii.gz")

    pv_mask = stl_to_mask("path/to/portal_vein.stl", ref)
    hv_mask = stl_to_mask("path/to/hepatic_vein.stl", ref)

    masks = {}
    label_map = {"portal_vein": 1, "hepatic_vein": 2}
    if pv_mask is not None:
        masks["portal_vein"] = pv_mask
    if hv_mask is not None:
        masks["hepatic_vein"] = hv_mask

    combined = combine_masks(masks, label_map, priority=["portal_vein", "hepatic_vein"])

    # Step 1: Fix 26-connectivity (BFS pathfinding through non-vessel space)
    # This reconnects fragments of the same vessel that were split
    # when the higher-priority vessel overwrote junction voxels
    combined = bridge_vessel_26conn(
        combined,
        label_map,
        vessel_labels={"portal_vein", "hepatic_vein"},
    )

    # Step 2: Ensure 6-connectivity matches 26-connectivity
    # Adds face-adjacent bridge voxels for diagonal-only connections
    combined = bridge_vessel_6conn(
        combined,
        label_map,
        vessel_labels={"portal_vein", "hepatic_vein"},
    )

    save_nifti(combined, ref, "vessels_bridged.nii.gz")
    print("Saved vessels_bridged.nii.gz with connectivity repair")


# --- Example 4: Command-line usage ---

def example_cli():
    """Equivalent commands using the CLI."""
    print("=== CLI Examples ===\n")

    print("# Single STL → NIfTI")
    print("vesselknit --stl vessel.stl --reference ct.nii.gz --output vessel.nii.gz\n")

    print("# Multi-label with bridging")
    print("vesselknit \\")
    print("  --stl portal_vein.stl --label 1 \\")
    print("  --stl hepatic_vein.stl --label 2 \\")
    print("  --reference ct.nii.gz \\")
    print("  --output vessels.nii.gz \\")
    print("  --bridge-26conn --bridge-6conn\n")

    print("# With custom names and priority")
    print("vesselknit \\")
    print('  --stl pv.stl --label 1 --name portal_vein \\')
    print('  --stl hv.stl --label 2 --name hepatic_vein \\')
    print("  --reference ct/ \\")
    print("  --output vessels.nii.gz \\")
    print('  --priority "portal_vein,hepatic_vein" \\')
    print("  --bridge-26conn\n")


if __name__ == "__main__":
    print("vesselknit usage examples")
    print("=" * 40)
    print("\nNote: These examples use placeholder paths.")
    print("Replace with actual file paths to run them.\n")

    example_cli()
