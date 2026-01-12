"""
Spectrum file readers.

This module provides readers for various mass spectrometry file formats:

Tier 1 (Open formats):
- MzMLReader: mzML and mzXML files (pyteomics)

Tier 2 (Vendor-specific, direct reading):
- ThermoReader: Thermo .raw files (planned)
- BrukerReader: Bruker .d folders/timsTOF (planned)

Tier 3 (Via ProteoWizard):
- ProteoWizardReader: Any format via msconvert (planned)

Convenience functions:
- read_mzml(): Load mzML file into MSRun
"""

from .mzml import MzMLReader, read_mzml

__all__ = [
    "MzMLReader",
    "read_mzml",
]
