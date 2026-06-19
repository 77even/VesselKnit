# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Tests for I/O module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import SimpleITK as sitk
except ImportError:
    pytest.skip("SimpleITK not available", allow_module_level=True)

from vesselknit.io import load_reference_image, save_nifti


class TestLoadReferenceImage:
    """Tests for load_reference_image."""

    def test_load_nifti_file(self, tmp_path):
        """Should load a NIfTI file correctly."""
        # Create a test NIfTI
        image = sitk.Image((64, 64, 32), sitk.sitkInt16)
        image.SetSpacing((1.0, 1.0, 2.0))
        image.SetOrigin((10.0, 20.0, 30.0))

        nifti_path = tmp_path / "test.nii.gz"
        sitk.WriteImage(image, str(nifti_path))

        loaded = load_reference_image(nifti_path)
        assert loaded.GetSize() == (64, 64, 32)
        assert list(loaded.GetSpacing()) == [1.0, 1.0, 2.0]
        assert list(loaded.GetOrigin()) == [10.0, 20.0, 30.0]

    def test_load_nonexistent_path_raises(self):
        """Should raise FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_reference_image("/nonexistent/path.nii.gz")

    def test_load_unrecognized_type_raises(self, tmp_path):
        """Should raise ValueError for unrecognized file types."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello")
        with pytest.raises(ValueError, match="Unrecognized"):
            load_reference_image(txt_path)


class TestSaveNifti:
    """Tests for save_nifti."""

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Saved NIfTI should reload with correct metadata."""
        # Create reference image
        ref = sitk.Image((32, 32, 16), sitk.sitkInt16)
        ref.SetSpacing((0.5, 0.5, 1.0))
        ref.SetOrigin((5.0, 10.0, 15.0))
        ref.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

        # Create mask array (z, y, x) = (16, 32, 32)
        mask = np.zeros((16, 32, 32), dtype=np.uint8)
        mask[4:12, 8:24, 8:24] = 1

        output_path = tmp_path / "mask.nii.gz"
        save_nifti(mask, ref, output_path)

        # Reload and verify
        loaded = sitk.ReadImage(str(output_path))
        assert loaded.GetSize() == (32, 32, 16)
        assert list(loaded.GetSpacing()) == [0.5, 0.5, 1.0]
        assert list(loaded.GetOrigin()) == [5.0, 10.0, 15.0]

        # Verify mask values
        loaded_arr = sitk.GetArrayFromImage(loaded)
        np.testing.assert_array_equal(loaded_arr, mask)

    def test_save_multilabel(self, tmp_path):
        """Should preserve multi-label values."""
        ref = sitk.Image((10, 10, 10), sitk.sitkInt16)
        ref.SetSpacing((1.0, 1.0, 1.0))

        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        mask[6:9, 6:9, 6:9] = 2

        output_path = tmp_path / "multi.nii.gz"
        save_nifti(mask, ref, output_path)

        loaded = sitk.ReadImage(str(output_path))
        loaded_arr = sitk.GetArrayFromImage(loaded)
        assert set(np.unique(loaded_arr)) == {0, 1, 2}

    def test_output_directory_created(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        ref = sitk.Image((10, 10, 10), sitk.sitkInt16)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)

        output_path = tmp_path / "subdir" / "nested" / "output.nii.gz"
        save_nifti(mask, ref, output_path)
        assert output_path.exists()
