# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for vesselknit.

Usage:
    # Single STL → NIfTI
    vesselknit --stl vessel.stl --reference ct.nii.gz --output vessel.nii.gz

    # Batch: multiple STLs → multi-label NIfTI
    vesselknit \\
        --stl portal_vein.stl --label 1 \\
        --stl hepatic_vein.stl --label 2 \\
        --reference ct.nii.gz \\
        --output vessels.nii.gz \\
        --bridge-26conn
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import SimpleITK as sitk
except ImportError:
    sys.exit("Error: SimpleITK is required. Install with: pip install SimpleITK")

from vesselknit._version import __version__
from vesselknit.io import load_reference_image, save_nifti
from vesselknit.core import stl_to_mask, combine_masks
from vesselknit.bridge import bridge_vessel_26conn, bridge_vessel_6conn

logger = logging.getLogger(__name__)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="vesselknit",
        description=(
            "Convert STL surface meshes to volumetric NIfTI masks "
            "with coordinate alignment and vessel connectivity repair."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single STL\n"
            "  vesselknit --stl vessel.stl --reference ct/ --output vessel.nii.gz\n"
            "\n"
            "  # Multi-label with bridging\n"
            "  vesselknit \\\n"
            "    --stl portal_vein.stl --label 1 \\\n"
            "    --stl hepatic_vein.stl --label 2 \\\n"
            "    --reference ct.nii.gz \\\n"
            "    --output vessels.nii.gz \\\n"
            "    --bridge-26conn\n"
        ),
    )

    parser.add_argument(
        "--stl",
        action="append",
        required=True,
        metavar="PATH",
        help="Path to STL file. Can be specified multiple times for multi-label output.",
    )
    parser.add_argument(
        "--label",
        action="append",
        type=int,
        metavar="INT",
        help=(
            "Label index for the corresponding --stl (1-based). "
            "If omitted with a single --stl, defaults to 1. "
            "For multiple --stl, must provide one --label per --stl."
        ),
    )
    parser.add_argument(
        "--reference",
        required=True,
        metavar="PATH",
        help=(
            "Reference image: DICOM directory or NIfTI file (.nii/.nii.gz). "
            "Defines the output volume geometry."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output NIfTI file path (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--name",
        action="append",
        metavar="STR",
        help=(
            "Name for the corresponding --stl (used for logging and bridging). "
            "Defaults to the filename stem."
        ),
    )
    parser.add_argument(
        "--priority",
        metavar="NAMES",
        help=(
            "Comma-separated label write priority (later overwrites earlier). "
            'E.g., --priority "liver,portal_vein,hepatic_vein"'
        ),
    )

    # Voxelization options
    vox_group = parser.add_argument_group("Voxelization options")
    vox_group.add_argument(
        "--no-coordinate-check",
        action="store_true",
        help="Skip coordinate space alignment check.",
    )
    vox_group.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="Keep only the largest connected mesh component per STL.",
    )
    vox_group.add_argument(
        "--bridge-radii",
        type=str,
        default="3,5,8,12,18",
        help=(
            "Comma-separated progressive dilation radii for gap-bridging. "
            "Set to empty string to disable. (default: 3,5,8,12,18)"
        ),
    )
    vox_group.add_argument(
        "--min-fragment-size",
        type=int,
        default=50,
        help="Minimum component size (voxels) to keep after closing (default: 50).",
    )

    # Connectivity repair options
    conn_group = parser.add_argument_group("Connectivity repair options")
    conn_group.add_argument(
        "--bridge-26conn",
        action="store_true",
        help="Enable 26-connectivity vessel bridging (BFS pathfinding).",
    )
    conn_group.add_argument(
        "--bridge-6conn",
        action="store_true",
        help="Enable 6-connectivity vessel bridging (diagonal fill).",
    )
    conn_group.add_argument(
        "--vessel-labels",
        metavar="NAMES",
        help=(
            "Comma-separated label names to apply bridging to. "
            "Defaults to all labels when bridging is enabled."
        ),
    )

    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"vesselknit {__version__}",
    )

    args = parser.parse_args(argv)

    # Validate --stl / --label counts
    if args.label is not None and len(args.label) != len(args.stl):
        parser.error(
            f"Number of --label ({len(args.label)}) must match "
            f"number of --stl ({len(args.stl)})"
        )

    if args.name is not None and len(args.name) != len(args.stl):
        parser.error(
            f"Number of --name ({len(args.name)}) must match "
            f"number of --stl ({len(args.stl)})"
        )

    return args


def main(argv: List[str] = None) -> None:
    """Main entry point for the CLI."""
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress VTK warnings
    try:
        import vtk
        vtk.vtkObject.GlobalWarningDisplayOff()
    except ImportError:
        pass

    # Parse bridge radii
    if args.bridge_radii.strip():
        bridge_radii = tuple(int(x.strip()) for x in args.bridge_radii.split(","))
    else:
        bridge_radii = ()

    # Build STL → label mapping
    stl_items: List[Tuple[Path, int, str]] = []
    for i, stl_path in enumerate(args.stl):
        path = Path(stl_path)
        label_idx = args.label[i] if args.label else (i + 1)
        name = args.name[i] if args.name else path.stem
        stl_items.append((path, label_idx, name))

    # Load reference image
    logger.info(f"Loading reference image: {args.reference}")
    try:
        ref_image = load_reference_image(args.reference)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    # Voxelize each STL
    masks = {}
    label_map = {}
    for stl_path, label_idx, name in stl_items:
        logger.info(f"Voxelizing: {stl_path} → label {label_idx} ({name})")
        mask = stl_to_mask(
            stl_path,
            ref_image,
            check_coordinate_space=not args.no_coordinate_check,
            keep_largest_component=args.keep_largest_component,
            bridge_radii=bridge_radii,
            min_fragment_size=args.min_fragment_size,
        )
        if mask is None:
            logger.warning(f"Skipping {stl_path}: voxelization failed")
            continue

        masks[name] = mask
        label_map[name] = label_idx
        logger.info(f"  {name}: {mask.sum()} voxels")

    if not masks:
        sys.exit("Error: no STL files were successfully voxelized")

    # Combine masks
    priority = None
    if args.priority:
        priority = [n.strip() for n in args.priority.split(",")]
    elif len(masks) > 1:
        # Default: last STL has highest priority
        priority = list(masks.keys())

    combined = combine_masks(masks, label_map, priority=priority)

    # Connectivity repair
    if args.bridge_26conn or args.bridge_6conn:
        vessel_labels = None
        if args.vessel_labels:
            vessel_labels = set(n.strip() for n in args.vessel_labels.split(","))

        if args.bridge_26conn:
            logger.info("Applying 26-connectivity bridging...")
            combined = bridge_vessel_26conn(combined, label_map, vessel_labels)

        if args.bridge_6conn:
            logger.info("Applying 6-connectivity bridging...")
            combined = bridge_vessel_6conn(combined, label_map, vessel_labels)

    # Save output
    save_nifti(combined, ref_image, args.output)
    logger.info(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
