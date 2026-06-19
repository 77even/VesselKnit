<div align="center">

# VesselKnit

**Voxelize vessel STL meshes into NIfTI masks — without breakage.**

[![PyPI](https://img.shields.io/pypi/v/vesselknit?cacheSeconds=600)](https://pypi.org/project/vesselknit/)
[![Python](https://img.shields.io/pypi/pyversions/vesselknit?cacheSeconds=600)](https://pypi.org/project/vesselknit/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

Rasterizing a thin vessel STL onto a CT/MR grid usually **fractures** it into
disconnected blobs at narrow junctions. VesselKnit keeps the vessel a single
connected structure — faithful to the original mesh in **morphology and
calibre** — via progressive gap-bridging during voxelization plus 26-/6-connectivity
repair. Built for medical-imaging workflows; coordinate alignment (LPS/RAS,
Z-reflection, direction matrix) is automatic.

## Install

```bash
pip install vesselknit
```

Optional extras: `[dicom]` for DICOM input, `[all]` for everything.
Python ≥ 3.9.

## Usage

```bash
vesselknit --stl vessel.stl --reference ct_scan/ --output vessel.nii.gz
```

Multiple STLs into one multi-label volume, with connectivity repair:

```bash
vesselknit \
  --stl portal_vein.stl --label 1 \
  --stl hepatic_vein.stl --label 2 \
  --reference ct_scan/ --output vessels.nii.gz \
  --bridge-26conn
```

```python
from vesselknit import load_reference_image, stl_to_mask, save_nifti

ref = load_reference_image("ct_scan/")      # DICOM dir or .nii.gz
mask = stl_to_mask("vessel.stl", ref)       # (z, y, x) uint8
save_nifti(mask, ref, "vessel.nii.gz")
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for multi-label and
connectivity-repair examples.

## License

[Apache 2.0](LICENSE) © 2026 Justin
