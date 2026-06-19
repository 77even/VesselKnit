# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Vessel connectivity repair: reconnect fragmented vessel structures.

When multiple vessel labels are combined with priority ordering, higher-priority
vessels can overwrite voxels at thin junctions of lower-priority vessels,
splitting them into disconnected components. These functions restore connectivity.

Two repair strategies:
- `bridge_vessel_26conn`: BFS pathfinding through non-vessel space (26-connectivity)
- `bridge_vessel_6conn`: Add face-adjacent bridges for diagonal connections (6-connectivity)
"""

import logging
from collections import deque
from typing import Dict, Optional, Set

import numpy as np

try:
    from scipy import ndimage as _ndi
except ImportError:
    raise ImportError("scipy is required. Install with: pip install scipy")

logger = logging.getLogger(__name__)


def _resolve_vessel_indices(
    label_map: Dict[str, int],
    vessel_labels: Optional[Set[str]],
) -> Set[int]:
    """
    Resolve which label indices to bridge.

    Returns the indices of the named ``vessel_labels`` (skipping any not in
    ``label_map``), or every index in ``label_map`` when ``vessel_labels`` is
    None (treat all labels as vessels).
    """
    if vessel_labels is None:
        return set(label_map.values())
    return {label_map[name] for name in vessel_labels if name in label_map}


def bridge_vessel_26conn(
    volume: np.ndarray,
    label_map: Dict[str, int],
    vessel_labels: Optional[Set[str]] = None,
) -> np.ndarray:
    """
    Reconnect vessel fragments split by label-priority overlap.

    When `combine_masks()` writes vessel labels by priority, a higher-priority
    vessel can overwrite voxels at thin junctions of a lower-priority vessel,
    splitting it into multiple 26-connected components even though the original
    STL was a single connected mesh.

    This function finds the shortest 26-connected path between fragments
    through non-vessel space and adds minimal bridge voxels to restore
    connectivity.

    Args:
        volume: Multi-label volume (z, y, x), uint8
        label_map: Dict mapping label_name → label_index
        vessel_labels: Set of label names to bridge. If None, bridges all
                      labels in label_map (i.e., treats every label as a
                      vessel). For typical use, specify only thin tubular
                      structures (vessels, ducts, bronchi).

    Returns:
        Modified volume with 26-conn bridge voxels inserted.

    Example:
        >>> volume = bridge_vessel_26conn(
        ...     volume,
        ...     label_map={"portal_vein": 1, "hepatic_vein": 2},
        ...     vessel_labels={"portal_vein", "hepatic_vein"},
        ... )
    """
    vessel_indices = _resolve_vessel_indices(label_map, vessel_labels)
    if not vessel_indices:
        return volume

    struct26 = _ndi.generate_binary_structure(3, 3)

    result = volume.copy()
    # Refreshed after every label so bridges from earlier labels are
    # treated as immovable obstacles for later labels.
    all_vessel_mask = np.isin(result, list(vessel_indices))
    Z, Y, X = volume.shape

    # 26-conn neighbor offsets
    offsets_26 = [
        (dz, dy, dx)
        for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    # Reverse lookup: index → name (for logging)
    idx_to_name = {v: k for k, v in label_map.items()}

    for lbl_idx in sorted(vessel_indices):
        mask = (result == lbl_idx)
        other_vessels = all_vessel_mask & ~mask

        labeled, n = _ndi.label(mask, struct26)
        if n <= 1:
            continue

        # Sort components by size (largest first = main)
        comp_sizes = _ndi.sum(mask, labeled, range(1, n + 1))
        sorted_ids = (np.argsort(-np.array(comp_sizes)) + 1).tolist()

        main_region = (labeled == sorted_ids[0])

        for fid in sorted_ids[1:]:
            frag = (labeled == fid)

            # BFS from fragment boundary toward main through non-vessel space
            frag_dilated = _ndi.binary_dilation(frag, struct26)
            start_voxels = np.argwhere(frag_dilated & ~frag & ~other_vessels)

            parent: Dict[tuple, Optional[tuple]] = {}
            queue = deque()
            for coord in start_voxels:
                key = (int(coord[0]), int(coord[1]), int(coord[2]))
                parent[key] = None
                queue.append(key)

            found = False
            while queue:
                z, y, x = queue.popleft()

                if main_region[z, y, x]:
                    # Trace back and collect bridge voxels
                    path = []
                    pos: Optional[tuple] = (z, y, x)
                    while pos is not None:
                        pz, py, px = pos
                        if not frag[pz, py, px] and not main_region[pz, py, px]:
                            path.append(pos)
                        pos = parent.get(pos)

                    if path:
                        for pz, py, px in path:
                            result[pz, py, px] = lbl_idx
                            main_region[pz, py, px] = True
                        name = idx_to_name.get(lbl_idx, str(lbl_idx))
                        logger.info(
                            f"  26-conn bridge: {name} +{len(path)} voxels"
                        )
                    main_region |= frag
                    found = True
                    break

                for dz, dy, dx in offsets_26:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                        nkey = (nz, ny, nx)
                        if nkey not in parent:
                            if not other_vessels[nz, ny, nx] or main_region[nz, ny, nx]:
                                parent[nkey] = (z, y, x)
                                queue.append(nkey)

            if not found:
                main_region |= frag  # merge anyway to avoid re-processing

        # Refresh so later labels' BFS treats this label's bridges as obstacles
        all_vessel_mask = np.isin(result, list(vessel_indices))

    return result


def bridge_vessel_6conn(
    volume: np.ndarray,
    label_map: Dict[str, int],
    vessel_labels: Optional[Set[str]] = None,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Add minimal bridge voxels so that vessel 6-connectivity matches
    26-connectivity.

    For each vessel label, finds voxel pairs that are 26-connected (diagonal)
    but not 6-connected (face-adjacent), and inserts the minimum number of
    intermediate voxels to create face-adjacent paths.

    Bridge voxels only overwrite background or non-vessel labels — they
    NEVER overwrite other vessel labels, ensuring zero cross-vessel damage.

    Typically adds < 0.1% extra voxels per vessel.

    Args:
        volume: Multi-label volume (z, y, x), uint8
        label_map: Dict mapping label_name → label_index
        vessel_labels: Set of label names to bridge. If None, bridges all
                      labels in label_map.
        max_iterations: Maximum iterations for 2-step dilation overlap
                       phase (default: 10)

    Returns:
        Modified volume with bridge voxels inserted.

    Example:
        >>> volume = bridge_vessel_6conn(
        ...     volume,
        ...     label_map={"portal_vein": 1, "hepatic_vein": 2},
        ...     vessel_labels={"portal_vein", "hepatic_vein"},
        ... )
    """
    vessel_indices = _resolve_vessel_indices(label_map, vessel_labels)
    if not vessel_indices:
        return volume

    struct6 = _ndi.generate_binary_structure(3, 1)
    struct26 = _ndi.generate_binary_structure(3, 3)

    Z, Y, X = volume.shape

    result = volume.copy()
    # Refreshed after every label so previous labels' bridges become
    # obstacles for later labels.
    all_vessel_mask = np.isin(result, list(vessel_indices))

    # Edge-diagonal: 2 coords differ by ±1, need 1 bridge voxel, 2 candidates
    edge_pairs = [
        ((1, 1, 0),  [(1, 0, 0), (0, 1, 0)]),
        ((1, -1, 0), [(1, 0, 0), (0, -1, 0)]),
        ((1, 0, 1),  [(1, 0, 0), (0, 0, 1)]),
        ((1, 0, -1), [(1, 0, 0), (0, 0, -1)]),
        ((0, 1, 1),  [(0, 1, 0), (0, 0, 1)]),
        ((0, 1, -1), [(0, 1, 0), (0, 0, -1)]),
    ]

    # Corner-diagonal: 3 coords differ by ±1, need 2 bridge voxels, 3 path options
    corner_pairs = [
        ((1, 1, 1),   [[(1, 0, 0), (1, 1, 0)], [(0, 1, 0), (1, 1, 0)], [(0, 0, 1), (0, 1, 1)]]),
        ((1, 1, -1),  [[(1, 0, 0), (1, 1, 0)], [(0, 1, 0), (1, 1, 0)], [(0, 0, -1), (0, 1, -1)]]),
        ((1, -1, 1),  [[(1, 0, 0), (1, -1, 0)], [(0, -1, 0), (1, -1, 0)], [(0, 0, 1), (0, -1, 1)]]),
        ((1, -1, -1), [[(1, 0, 0), (1, -1, 0)], [(0, -1, 0), (1, -1, 0)], [(0, 0, -1), (0, -1, -1)]]),
    ]

    # Reverse lookup for logging
    idx_to_name = {v: k for k, v in label_map.items()}

    for lbl_idx in sorted(vessel_indices):
        mask = (result == lbl_idx)
        if mask.sum() == 0:
            continue

        other_vessels = all_vessel_mask & ~mask
        vessel_result = mask.copy()

        coords = np.argwhere(mask)

        # Phase 1: Direct diagonal pair bridging
        for z, y, x in coords:
            # Edge-diagonals
            for (dz, dy, dx), bridges in edge_pairs:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X and mask[nz, ny, nx]:
                    if any(vessel_result[z + bz, y + by, x + bx]
                           for bz, by, bx in bridges):
                        continue
                    for bz, by, bx in bridges:
                        pz, py, px = z + bz, y + by, x + bx
                        if 0 <= pz < Z and 0 <= py < Y and 0 <= px < X:
                            if (not other_vessels[pz, py, px]
                                    and not vessel_result[pz, py, px]):
                                vessel_result[pz, py, px] = True
                                break

            # Corner-diagonals
            for (dz, dy, dx), bridge_paths in corner_pairs:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X and mask[nz, ny, nx]:
                    already = False
                    for path in bridge_paths:
                        for bz, by, bx in path:
                            if vessel_result[z + bz, y + by, x + bx]:
                                already = True
                                break
                        if already:
                            break
                    if already:
                        continue
                    for path in bridge_paths:
                        ok = True
                        positions = []
                        for bz, by, bx in path:
                            pz, py, px = z + bz, y + by, x + bx
                            if not (0 <= pz < Z and 0 <= py < Y and 0 <= px < X):
                                ok = False
                                break
                            if other_vessels[pz, py, px]:
                                ok = False
                                break
                            positions.append((pz, py, px))
                        if ok and positions:
                            for pz, py, px in positions:
                                if not vessel_result[pz, py, px]:
                                    vessel_result[pz, py, px] = True
                            break

        # Phase 2: 2-step dilation overlap for remaining gaps
        for _ in range(max_iterations):
            labeled26, n26 = _ndi.label(vessel_result, struct26)
            labeled6, n6 = _ndi.label(vessel_result, struct6)
            if n6 == n26:
                break
            added = False
            for comp26_id in range(1, n26 + 1):
                comp_mask = (labeled26 == comp26_id)
                sub_labeled6, sub_n6 = _ndi.label(comp_mask, struct6)
                if sub_n6 <= 1:
                    continue
                dilated_sum = np.zeros(volume.shape, dtype=np.int16)
                for sub_id in range(1, sub_n6 + 1):
                    sub_mask = (sub_labeled6 == sub_id)
                    dilated = _ndi.binary_dilation(sub_mask, struct6, iterations=2)
                    dilated_sum += dilated.astype(np.int16)
                bridges = (dilated_sum >= 2) & ~comp_mask & ~other_vessels
                if bridges.any():
                    vessel_result |= bridges
                    added = True
            if not added:
                break

        # Write bridge voxels into result
        new_voxels = vessel_result & ~mask
        added_count = int(new_voxels.sum())
        if added_count > 0:
            result[new_voxels] = lbl_idx
            # Bridges just added become obstacles for subsequent labels
            all_vessel_mask |= new_voxels
            name = idx_to_name.get(lbl_idx, str(lbl_idx))
            logger.info(f"  6-conn bridging: {name} += {added_count} voxels")

    return result
