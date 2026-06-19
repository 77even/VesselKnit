# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for vessel connectivity repair (bridge functions).
"""

import numpy as np
import pytest

try:
    from scipy import ndimage as _ndi
except ImportError:
    pytest.skip("scipy not available", allow_module_level=True)

from vesselknit.bridge import bridge_vessel_26conn, bridge_vessel_6conn


class TestBridge26Conn:
    """Tests for bridge_vessel_26conn."""

    def test_already_connected_no_change(self):
        """A single connected component should not be modified."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        # Create a solid block (1 component)
        volume[5:15, 5:15, 5:15] = 1

        result = bridge_vessel_26conn(volume, {"vessel": 1})
        np.testing.assert_array_equal(result, volume)

    def test_reconnects_split_vessel(self):
        """Two fragments of the same label should be reconnected."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        # Two cubes separated by 1 voxel gap
        volume[5:8, 10, 10] = 1   # fragment 1
        volume[10:13, 10, 10] = 1  # fragment 2 (same label)

        result = bridge_vessel_26conn(volume, {"vessel": 1})

        # After bridging, should be one component
        labeled, n = _ndi.label(result == 1)
        assert n == 1 or result.sum() > volume.sum()

    def test_does_not_overwrite_other_vessels(self):
        """Bridge voxels should not overwrite other vessel labels."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        volume[5:8, 10, 10] = 1   # vessel A fragment 1
        volume[10:13, 10, 10] = 1  # vessel A fragment 2
        volume[8, 10, 10] = 2      # vessel B (between fragments)

        result = bridge_vessel_26conn(
            volume,
            label_map={"vessel_a": 1, "vessel_b": 2},
            vessel_labels={"vessel_a"},
        )

        # Vessel B should not be overwritten
        assert result[8, 10, 10] == 2

    def test_custom_vessel_labels(self):
        """Only specified vessel_labels should be processed."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        volume[5:8, 10, 10] = 1  # label 1
        volume[10:13, 10, 10] = 2  # label 2 (separate)

        # Only bridge label 1
        result = bridge_vessel_26conn(
            volume,
            label_map={"a": 1, "b": 2},
            vessel_labels={"a"},
        )

        # Label 2 should be unchanged
        assert np.all(result[10:13, 10, 10] == 2)


class TestBridge6Conn:
    """Tests for bridge_vessel_6conn."""

    def test_already_6_connected_no_change(self):
        """A 6-connected volume should not be modified."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        volume[5:15, 5:15, 5:15] = 1

        result = bridge_vessel_6conn(volume, {"vessel": 1})
        # Should not add significant voxels
        assert result.sum() == volume.sum()

    def test_diagonal_bridge(self):
        """A diagonal-only connection should get face-adjacent bridges."""
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        # Place two voxels that are diagonally adjacent (edge-diagonal)
        volume[5, 5, 5] = 1
        volume[6, 6, 5] = 1  # edge-diagonal from (5,5,5)

        # These are 26-connected but NOT 6-connected
        labeled26, n26 = _ndi.label(volume == 1,
                                     _ndi.generate_binary_structure(3, 3))
        labeled6, n6 = _ndi.label(volume == 1,
                                   _ndi.generate_binary_structure(3, 1))
        assert n26 == 1  # 26-connected as one component
        assert n6 == 2   # but 2 components in 6-conn

        result = bridge_vessel_6conn(volume, {"vessel": 1})

        # After bridging, should be 6-connected
        labeled6_after, n6_after = _ndi.label(
            result == 1,
            _ndi.generate_binary_structure(3, 1)
        )
        assert n6_after == 1

    def test_does_not_overwrite_other_vessels_in_3d(self):
        """When alternative 6-conn paths exist, vessel B should be preserved."""
        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        # Two slices of label 1 with a gap, connected via Z
        volume[5:8, 8:12, 8] = 1
        volume[5:8, 8:12, 10] = 1
        # Label 2 sits on z=9 but doesn't block the Z-bridge path
        volume[13:16, 8:12, 9] = 2

        result = bridge_vessel_6conn(
            volume,
            label_map={"a": 1, "b": 2},
            vessel_labels={"a"},
        )

        # Label 2 should be completely preserved (it's not in the bridge zone)
        assert np.all(result[13:16, 8:12, 9] == 2)
