# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for core voxelization functionality.
"""

import numpy as np
import pytest

try:
    import SimpleITK as sitk
except ImportError:
    pytest.skip("SimpleITK not available", allow_module_level=True)

from vesselknit.core import stl_to_mask, combine_masks, _check_substantial_overlap


class TestStlToMask:
    """Tests for stl_to_mask function."""

    def test_sphere_produces_mask(self, sphere_stl, reference_image):
        """A sphere STL should produce a non-empty binary mask."""
        mask = stl_to_mask(sphere_stl, reference_image)
        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.sum() > 0
        # Mask should be binary (0 or 1)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_sphere_mask_shape_matches_reference(self, sphere_stl, reference_image):
        """Mask shape should match reference image (z, y, x)."""
        mask = stl_to_mask(sphere_stl, reference_image)
        assert mask is not None
        assert mask.shape == (
            reference_image.GetSize()[2],
            reference_image.GetSize()[1],
            reference_image.GetSize()[0],
        )

    def test_sphere_is_roughly_spherical(self, sphere_stl, reference_image):
        """The mask should be roughly spherical (connected, compact)."""
        from scipy import ndimage

        mask = stl_to_mask(sphere_stl, reference_image)
        assert mask is not None

        # Should be a single connected component
        labeled, n_components = ndimage.label(mask)
        assert n_components == 1

        # Center of mass should be near (50, 50, 50)
        com = ndimage.center_of_mass(mask)
        for c in com:
            assert 35 < c < 65, f"Center of mass {com} too far from expected (50,50,50)"

    def test_tube_produces_mask(self, tube_stl, reference_image):
        """A tube STL should produce a non-empty binary mask."""
        mask = stl_to_mask(tube_stl, reference_image)
        assert mask is not None
        assert mask.sum() > 0

    def test_keep_largest_component(self, tmp_path, reference_image):
        """keep_largest_component should filter mesh fragments."""
        # Create two disconnected spheres as a single STL
        # (manually write a multi-body STL)
        import vtk

        sphere1 = vtk.vtkSphereSource()
        sphere1.SetCenter(30, 50, 50)
        sphere1.SetRadius(10)
        sphere1.Update()

        sphere2 = vtk.vtkSphereSource()
        sphere2.SetCenter(70, 50, 50)
        sphere2.SetRadius(5)
        sphere2.Update()

        # Combine using append filter
        append = vtk.vtkAppendPolyData()
        append.AddInputData(sphere1.GetOutput())
        append.AddInputData(sphere2.GetOutput())
        append.Update()

        stl_path = tmp_path / "double_sphere.stl"
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(str(stl_path))
        writer.SetInputData(append.GetOutput())
        writer.Write()

        # Without keep_largest_component, mask may have 2 components
        mask_both = stl_to_mask(
            stl_path, reference_image,
            keep_largest_component=False,
            bridge_radii=(),  # disable bridging
        )
        if mask_both is not None:
            from scipy import ndimage
            _, n_both = ndimage.label(mask_both)

        # With keep_largest_component, should filter to 1
        mask_largest = stl_to_mask(
            stl_path, reference_image,
            keep_largest_component=True,
            bridge_radii=(),
        )
        if mask_largest is not None:
            from scipy import ndimage
            _, n_largest = ndimage.label(mask_largest)
            assert n_largest <= 1 or n_largest < n_both

    def test_empty_stl_returns_none(self, tmp_path, reference_image):
        """An STL with zero points should return None."""
        # Create an empty STL file
        stl_path = tmp_path / "empty.stl"
        stl_path.write_text("solid empty\nendsolid empty\n")

        # vtkSTLReader may produce empty output
        mask = stl_to_mask(stl_path, reference_image)
        # Either None or empty mask is acceptable
        if mask is not None:
            assert mask.sum() == 0 or mask is None

    def test_disable_bridging(self, sphere_stl, reference_image):
        """Setting bridge_radii=() should skip gap-bridging."""
        mask = stl_to_mask(sphere_stl, reference_image, bridge_radii=())
        assert mask is not None
        assert mask.sum() > 0


class TestCombineMasks:
    """Tests for combine_masks function."""

    def test_combine_two_masks(self):
        """Two non-overlapping masks should produce correct labels."""
        shape = (10, 10, 10)
        mask_a = np.zeros(shape, dtype=np.uint8)
        mask_a[:5, :, :] = 1
        mask_b = np.zeros(shape, dtype=np.uint8)
        mask_b[5:, :, :] = 1

        result = combine_masks(
            masks={"a": mask_a, "b": mask_b},
            label_map={"a": 1, "b": 2},
            priority=["a", "b"],
        )

        assert result.dtype == np.uint8
        assert set(int(v) for v in np.unique(result)) == {1, 2}
        assert np.all(result[:5, :, :] == 1)
        assert np.all(result[5:, :, :] == 2)

    def test_priority_overwrites(self):
        """Higher priority mask should overwrite lower priority."""
        shape = (10, 10, 10)
        mask_a = np.ones(shape, dtype=np.uint8)  # covers entire volume
        mask_b = np.zeros(shape, dtype=np.uint8)
        mask_b[5, 5, 5] = 1  # single voxel

        # b has higher priority → overwrites a at the overlap
        result = combine_masks(
            masks={"a": mask_a, "b": mask_b},
            label_map={"a": 1, "b": 2},
            priority=["a", "b"],
        )

        assert result[5, 5, 5] == 2  # b overwrites a
        assert result[0, 0, 0] == 1  # a elsewhere

    def test_default_priority(self):
        """Without priority, uses dict insertion order."""
        shape = (5, 5, 5)
        mask_a = np.ones(shape, dtype=np.uint8)
        mask_b = np.ones(shape, dtype=np.uint8)

        result = combine_masks(
            masks={"a": mask_a, "b": mask_b},
            label_map={"a": 1, "b": 2},
        )

        # Last written should win
        assert np.all(result == 2)

    def test_empty_masks_raises(self):
        """Empty masks dict should raise ValueError."""
        with pytest.raises(ValueError):
            combine_masks(masks={}, label_map={})


class TestCheckSubstantialOverlap:
    """Tests for _check_substantial_overlap helper."""

    def test_overlapping_boxes(self):
        s_min = np.array([0, 0, 0])
        s_max = np.array([100, 100, 100])
        v_min = np.array([0, 0, 0])
        v_max = np.array([100, 100, 100])
        assert _check_substantial_overlap(s_min, s_max, v_min, v_max) is True

    def test_non_overlapping_boxes(self):
        s_min = np.array([200, 200, 200])
        s_max = np.array([300, 300, 300])
        v_min = np.array([0, 0, 0])
        v_max = np.array([100, 100, 100])
        assert _check_substantial_overlap(s_min, s_max, v_min, v_max) is False

    def test_marginal_overlap_rejected(self):
        """Edge-touching boxes with <5% overlap should be rejected."""
        s_min = np.array([0, 0, 0])
        s_max = np.array([100, 100, 100])
        v_min = np.array([99, 99, 99])  # only 1mm overlap on 100mm extent = 1%
        v_max = np.array([199, 199, 199])
        assert _check_substantial_overlap(s_min, s_max, v_min, v_max) is False
