# Copyright (c) 2026 Justin
# SPDX-License-Identifier: Apache-2.0

"""
Shared test fixtures for vesselknit tests.

Generates synthetic STL meshes and SimpleITK reference images
for testing without requiring real medical data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import SimpleITK as sitk
except ImportError:
    pytest.skip("SimpleITK not available", allow_module_level=True)

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk # pyright: ignore[reportMissingImports]
except ImportError:
    pytest.skip("VTK not available", allow_module_level=True)


def make_sphere_stl(
    output_path: Path,
    center: tuple = (50.0, 50.0, 50.0),
    radius: float = 20.0,
    resolution: int = 32,
) -> Path:
    """Create a sphere STL file using VTK."""
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*center)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)

    # Triangulate to ensure valid STL
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(sphere.GetOutputPort())

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(tri.GetOutputPort())
    writer.Write()

    return output_path


def make_tube_stl(
    output_path: Path,
    point1: tuple = (10.0, 50.0, 50.0),
    point2: tuple = (90.0, 50.0, 50.0),
    radius: float = 5.0,
    resolution: int = 32,
) -> Path:
    """Create a tubular STL file using VTK."""
    tube = vtk.vtkTubeFilter()

    # Create a line source
    line = vtk.vtkLineSource()
    line.SetPoint1(*point1)
    line.SetPoint2(*point2)
    line.SetResolution(50)

    tube.SetInputConnection(line.GetOutputPort())
    tube.SetRadius(radius)
    tube.SetNumberOfSides(resolution)
    tube.CappingOn()

    # Need to wrap in polydata for STL writer
    tube.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(tube.GetOutputPort())
    writer.Write()

    return output_path


def make_reference_image(
    size: tuple = (100, 100, 100),
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Create a blank SimpleITK reference image."""
    image = sitk.Image(size, sitk.sitkInt16)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    image.SetDirection(direction)
    return image


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sphere_stl(tmp_path):
    """Create a sphere STL file and return its path."""
    return make_sphere_stl(tmp_path / "sphere.stl")


@pytest.fixture
def tube_stl(tmp_path):
    """Create a tube STL file and return its path."""
    return make_tube_stl(tmp_path / "tube.stl")


@pytest.fixture
def reference_image():
    """Create a reference SimpleITK image (100x100x100, 1mm spacing)."""
    return make_reference_image()
