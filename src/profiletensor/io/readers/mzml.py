"""
mzML file reader using pyteomics.

This module provides the MzMLReader class for reading mzML and mzXML files,
the standard open formats for mass spectrometry data interchange.
"""

import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Optional

from ..base import SpectrumReader
from ...core import (
    Spectrum,
    ScanMetadata,
    PrecursorInfo,
    MSRun,
    RunMetadata,
    Polarity,
    SpectrumType,
    ActivationType,
)

import numpy as np


# Mapping of CV terms to ActivationType
_ACTIVATION_MAP: dict[str, ActivationType] = {
    'collision-induced dissociation': ActivationType.CID,
    'cid': ActivationType.CID,
    'beam-type collision-induced dissociation': ActivationType.HCD,
    'hcd': ActivationType.HCD,
    'higher energy beam-type collision-induced dissociation': ActivationType.HCD,
    'electron transfer dissociation': ActivationType.ETD,
    'etd': ActivationType.ETD,
    'electron capture dissociation': ActivationType.ECD,
    'ecd': ActivationType.ECD,
    'ultraviolet photodissociation': ActivationType.UVPD,
    'uvpd': ActivationType.UVPD,
    'infrared multiphoton dissociation': ActivationType.IRMPD,
    'irmpd': ActivationType.IRMPD,
    'pulsed q dissociation': ActivationType.PQD,
    'pqd': ActivationType.PQD,
}


def _parse_activation_type(activation_info: dict) -> ActivationType:
    """Parse activation type from mzML activation dictionary."""
    for key in activation_info:
        key_lower = key.lower()
        if key_lower in _ACTIVATION_MAP:
            return _ACTIVATION_MAP[key_lower]
    return ActivationType.UNKNOWN


def _parse_polarity(spectrum_data: dict) -> Polarity:
    """Parse polarity from mzML spectrum dictionary."""
    if spectrum_data.get('positive scan'):
        return Polarity.POSITIVE
    if spectrum_data.get('negative scan'):
        return Polarity.NEGATIVE
    return Polarity.UNKNOWN


def _parse_spectrum_type(spectrum_data: dict, mz_array: Optional[np.ndarray] = None) -> SpectrumType:
    """
    Parse spectrum type (profile/centroid) from mzML spectrum dictionary.
    
    If not explicitly stated, attempts to infer from data characteristics:
    - Profile data typically has many closely-spaced points
    - Centroid data has fewer, discrete peaks
    """
    if spectrum_data.get('profile spectrum'):
        return SpectrumType.PROFILE
    if spectrum_data.get('centroid spectrum'):
        return SpectrumType.CENTROID
    
    # Try to infer from m/z array characteristics
    if mz_array is not None and len(mz_array) > 100:
        # Profile data typically has consistent small spacing
        # Centroid data has irregular, larger spacing
        diffs = np.diff(mz_array[:100])  # Check first 100 points
        if len(diffs) > 0:
            median_diff = np.median(diffs)
            std_diff = np.std(diffs)
            # Profile mode: very consistent spacing (low std/median ratio)
            # and small spacing relative to m/z values
            if median_diff < 0.1 and std_diff / (median_diff + 1e-10) < 0.5:
                return SpectrumType.PROFILE
            # If large number of points but irregular spacing, likely still profile
            if len(mz_array) > 1000:
                return SpectrumType.PROFILE
    
    return SpectrumType.UNKNOWN


def _extract_scan_number(native_id: str, index: int) -> int:
    """
    Extract scan number from native ID string.
    
    Common formats:
    - "controllerType=0 controllerNumber=1 scan=123"
    - "scan=123"
    - "spectrum=123"
    - "index=123"
    - Just a number
    
    Falls back to index + 1 if parsing fails.
    """
    if not native_id:
        return index + 1
    
    # Try various patterns
    patterns = [
        r'scan=(\d+)',
        r'spectrum=(\d+)',
        r'index=(\d+)',
        r'^(\d+)$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, native_id)
        if match:
            return int(match.group(1))
    
    return index + 1


def _get_cv_value(spectrum_data: dict, *cv_names: str, default=None):
    """
    Get a value from CV term names in spectrum data.
    
    Tries multiple possible CV term names and returns the first match.
    """
    for name in cv_names:
        if name in spectrum_data:
            return spectrum_data[name]
    return default


class MzMLReader(SpectrumReader):
    """
    Reader for mzML and mzXML files using pyteomics.
    
    This reader supports both mzML and mzXML formats and provides
    efficient iteration and random access to spectra.
    
    Example:
        >>> with MzMLReader("sample.mzML") as reader:
        ...     for spectrum in reader:
        ...         print(spectrum.scan_number, spectrum.ms_level)
        ...
        ...     # Random access
        ...     spec = reader.get_spectrum(100)
    """
    
    vendor: ClassVar[str] = "Open Format"
    supported_extensions: ClassVar[list[str]] = ['.mzml', '.mzxml']
    
    def __init__(self, path: Path | str):
        """
        Initialize the mzML reader.
        
        Args:
            path: Path to mzML or mzXML file.
        """
        super().__init__(path)
        self._reader = None
        self._is_mzxml = self.path.suffix.lower() == '.mzxml'
        self._index: Optional[list[str]] = None  # Native IDs for random access
        self._n_spectra: Optional[int] = None
        self._run_metadata: Optional[dict] = None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if pyteomics is installed."""
        try:
            import pyteomics.mzml
            import pyteomics.mzxml
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_installation_instructions(cls) -> str:
        """Return installation instructions for pyteomics."""
        return (
            "Install pyteomics:\n"
            "  pip install pyteomics\n"
            "  # or with lxml for better performance:\n"
            "  pip install pyteomics lxml"
        )
    
    def __enter__(self) -> 'MzMLReader':
        """Open the file for reading."""
        if self._is_mzxml:
            from pyteomics import mzxml
            self._reader = mzxml.MzXML(str(self.path))
        else:
            from pyteomics import mzml
            self._reader = mzml.MzML(str(self.path))
        
        # Build index for random access
        self._build_index()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the file."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
    
    def _build_index(self) -> None:
        """Build index of spectrum native IDs."""
        if self._reader is None:
            return
        
        # pyteomics index can contain multiple types (spectrum, chromatogram)
        # We need to get only spectrum entries
        self._index = []
        
        if hasattr(self._reader, 'index') and self._reader.index:
            index = self._reader.index
            # mzML files have hierarchical index with 'spectrum' key
            if 'spectrum' in index:
                spectrum_index = index['spectrum']
                if hasattr(spectrum_index, 'keys'):
                    self._index = list(spectrum_index.keys())
            # mzXML or simple index structure
            elif hasattr(index, 'keys') and 'spectrum' not in index and 'chromatogram' not in index:
                self._index = list(index.keys())
        
        self._n_spectra = len(self._index)
    
    def __iter__(self) -> Iterator[Spectrum]:
        """Iterate over all spectra in the file."""
        if self._reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        
        self._reader.reset()
        for idx, spectrum_data in enumerate(self._reader):
            yield self._parse_spectrum(spectrum_data, idx)
    
    def __len__(self) -> int:
        """Total number of spectra in the file."""
        if self._n_spectra is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        return self._n_spectra
    
    def get_spectrum(self, scan_number: int) -> Spectrum:
        """
        Get spectrum by scan number.
        
        Note: This searches for a spectrum with matching scan number.
        For files where scan numbers match native ID indices, this is O(1).
        
        Args:
            scan_number: The scan number to retrieve.
            
        Returns:
            The spectrum with the given scan number.
            
        Raises:
            KeyError: If scan number not found.
        """
        if self._reader is None or self._index is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        
        # Try direct access patterns
        for pattern in [
            f"controllerType=0 controllerNumber=1 scan={scan_number}",
            f"scan={scan_number}",
            f"spectrum={scan_number}",
            str(scan_number),
        ]:
            try:
                spectrum_data = self._reader.get_by_id(pattern)
                idx = self._index.index(pattern) if pattern in self._index else 0
                return self._parse_spectrum(spectrum_data, idx)
            except (KeyError, ValueError):
                continue
        
        # Fall back to linear search
        for idx, spectrum_data in enumerate(self._reader):
            native_id = spectrum_data.get('id', '')
            if _extract_scan_number(native_id, idx) == scan_number:
                return self._parse_spectrum(spectrum_data, idx)
        
        raise KeyError(f"Scan number {scan_number} not found")
    
    def get_spectrum_by_index(self, index: int) -> Spectrum:
        """
        Get spectrum by index (0-based position in file).
        
        Args:
            index: The 0-based index.
            
        Returns:
            The spectrum at that index.
        """
        if self._reader is None or self._index is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        
        if index < 0 or index >= len(self._index):
            raise IndexError(f"Index {index} out of range (0-{len(self._index)-1})")
        
        native_id = self._index[index]
        spectrum_data = self._reader.get_by_id(native_id)
        return self._parse_spectrum(spectrum_data, index)
    
    def _parse_spectrum(self, spectrum_data: dict, index: int) -> Spectrum:
        """
        Parse a pyteomics spectrum dictionary into a Spectrum object.
        
        Args:
            spectrum_data: Dictionary from pyteomics.
            index: Position in file (0-based).
            
        Returns:
            Parsed Spectrum object.
        """
        # Extract native ID and scan number
        native_id = spectrum_data.get('id', '')
        scan_number = _extract_scan_number(native_id, index)
        
        # MS level
        ms_level = int(spectrum_data.get('ms level', 1))
        
        # Retention time (convert to seconds if needed)
        rt = self._parse_retention_time(spectrum_data)
        
        # Extract m/z and intensity arrays first (needed for spectrum type inference)
        mz = spectrum_data.get('m/z array', np.array([], dtype=np.float64))
        intensity = spectrum_data.get('intensity array', np.array([], dtype=np.float64))
        
        # Ensure proper types
        if not isinstance(mz, np.ndarray):
            mz = np.array(mz, dtype=np.float64)
        if not isinstance(intensity, np.ndarray):
            intensity = np.array(intensity, dtype=np.float64)
        
        # Polarity and spectrum type (with inference from data)
        polarity = _parse_polarity(spectrum_data)
        spectrum_type = _parse_spectrum_type(spectrum_data, mz)
        
        # Scan window
        scan_window_lower = None
        scan_window_upper = None
        scan_list = spectrum_data.get('scanList', {})
        scans = scan_list.get('scan', [])
        if scans:
            scan_info = scans[0] if isinstance(scans, list) else scans
            scan_window = scan_info.get('scanWindowList', {}).get('scanWindow', [])
            if scan_window:
                window = scan_window[0] if isinstance(scan_window, list) else scan_window
                scan_window_lower = window.get('scan window lower limit')
                scan_window_upper = window.get('scan window upper limit')
        
        # Acquisition metrics
        tic = _get_cv_value(spectrum_data, 'total ion current', 'TIC')
        base_peak_mz = _get_cv_value(spectrum_data, 'base peak m/z')
        base_peak_intensity = _get_cv_value(spectrum_data, 'base peak intensity')
        injection_time = _get_cv_value(spectrum_data, 'ion injection time')
        
        # Filter string (Thermo-specific but often present)
        filter_string = _get_cv_value(spectrum_data, 'filter string')
        
        # Precursor info for MS2+
        precursor = None
        if ms_level > 1:
            precursor = self._parse_precursor(spectrum_data)
        
        # Create metadata
        metadata = ScanMetadata(
            scan_number=scan_number,
            ms_level=ms_level,
            retention_time=rt,
            polarity=polarity,
            spectrum_type=spectrum_type,
            scan_window_lower=scan_window_lower,
            scan_window_upper=scan_window_upper,
            total_ion_current=tic,
            base_peak_mz=base_peak_mz,
            base_peak_intensity=base_peak_intensity,
            injection_time=injection_time,
            precursor=precursor,
            filter_string=filter_string,
            native_id=native_id,
        )
        
        return Spectrum(mz=mz, intensity=intensity, metadata=metadata)
    
    def _parse_retention_time(self, spectrum_data: dict) -> float:
        """Parse retention time, converting to seconds."""
        # Try different locations and units
        rt = None
        unit = 'second'
        
        # Check scanList first (standard location)
        scan_list = spectrum_data.get('scanList', {})
        scans = scan_list.get('scan', [])
        if scans:
            scan_info = scans[0] if isinstance(scans, list) else scans
            rt = scan_info.get('scan start time')
            # Check unit
            if 'scan start time' in scan_info:
                # pyteomics handles unit conversion, but double-check
                pass
        
        # Fall back to direct access
        if rt is None:
            rt = spectrum_data.get('scan start time', 0.0)
        
        # mzXML uses retentionTime
        if rt is None or rt == 0.0:
            rt_str = spectrum_data.get('retentionTime', '')
            if isinstance(rt_str, str) and rt_str.startswith('PT'):
                # ISO 8601 duration: PT60.5S
                match = re.match(r'PT([\d.]+)([SM])', rt_str)
                if match:
                    rt = float(match.group(1))
                    if match.group(2) == 'M':
                        rt *= 60.0
            elif isinstance(rt_str, (int, float)):
                rt = float(rt_str)
        
        return float(rt) if rt else 0.0
    
    def _parse_precursor(self, spectrum_data: dict) -> Optional[PrecursorInfo]:
        """Parse precursor information from spectrum data."""
        precursor_list = spectrum_data.get('precursorList', {})
        precursors = precursor_list.get('precursor', [])
        
        if not precursors:
            return None
        
        prec = precursors[0] if isinstance(precursors, list) else precursors
        
        # Selected ion
        ion_list = prec.get('selectedIonList', {})
        ions = ion_list.get('selectedIon', [])
        
        if not ions:
            return None
        
        ion = ions[0] if isinstance(ions, list) else ions
        
        # Precursor m/z
        mz = ion.get('selected ion m/z')
        if mz is None:
            return None
        
        # Charge state
        charge = ion.get('charge state')
        if charge is not None:
            charge = int(charge)
        
        # Intensity
        intensity = ion.get('peak intensity')
        
        # Isolation window
        isolation = prec.get('isolationWindow', {})
        iso_target = isolation.get('isolation window target m/z')
        iso_lower = isolation.get('isolation window lower offset')
        iso_upper = isolation.get('isolation window upper offset')
        
        # Activation
        activation = prec.get('activation', {})
        activation_type = _parse_activation_type(activation)
        
        # Collision energy
        collision_energy = activation.get('collision energy')
        collision_energy_unit = None
        if collision_energy is not None:
            # Check for NCE vs eV
            if activation.get('normalized collision energy'):
                collision_energy = activation['normalized collision energy']
                collision_energy_unit = 'NCE'
            else:
                collision_energy_unit = 'eV'
        
        # Parent scan reference
        parent_scan = None
        spec_ref = prec.get('spectrumRef', '')
        if spec_ref:
            parent_scan = _extract_scan_number(spec_ref, -1)
            if parent_scan == 0:  # extraction failed
                parent_scan = None
        
        return PrecursorInfo(
            mz=float(mz),
            charge=charge,
            intensity=float(intensity) if intensity else None,
            isolation_window_lower=float(iso_lower) if iso_lower else None,
            isolation_window_upper=float(iso_upper) if iso_upper else None,
            activation_type=activation_type,
            collision_energy=float(collision_energy) if collision_energy else None,
            collision_energy_unit=collision_energy_unit,
            parent_scan_number=parent_scan,
        )
    
    def iter_ms_level(self, ms_level: int) -> Iterator[Spectrum]:
        """Iterate over spectra of a specific MS level."""
        for spectrum in self:
            if spectrum.ms_level == ms_level:
                yield spectrum
    
    def get_ms_level_counts(self) -> dict[int, int]:
        """Return count of spectra per MS level."""
        counts: dict[int, int] = {}
        for spectrum in self:
            level = spectrum.ms_level
            counts[level] = counts.get(level, 0) + 1
        return counts
    
    @property
    def run_metadata(self) -> dict:
        """
        File-level metadata from mzML header.
        
        Returns:
            Dictionary with instrument, software, and file info.
        """
        if self._run_metadata is not None:
            return self._run_metadata
        
        if self._reader is None:
            return {}
        
        metadata: dict = {
            'source_file': str(self.path),
        }
        
        # For mzML, try to extract run-level metadata
        if not self._is_mzxml and hasattr(self._reader, 'iterfind'):
            try:
                # Get instrument configuration
                for inst in self._reader.iterfind('instrumentConfigurationList/instrumentConfiguration'):
                    if 'instrument model' in inst:
                        metadata['instrument_model'] = inst.get('instrument model')
                    if 'instrument serial number' in inst:
                        metadata['instrument_serial'] = inst.get('instrument serial number')
                    break
                
                # Get software info
                for soft in self._reader.iterfind('softwareList/software'):
                    metadata['software_version'] = soft.get('version', '')
                    break
                    
            except Exception:
                pass
        
        self._run_metadata = metadata
        return metadata
    
    def to_run(self) -> MSRun:
        """
        Load the entire file into an MSRun object.
        
        This loads all spectra into memory. For large files,
        consider iterating directly instead.
        
        Returns:
            MSRun containing all spectra and run metadata.
        """
        if self._reader is None:
            raise RuntimeError("Reader not opened. Use 'with' context manager.")
        
        # Collect all spectra
        spectra = list(self)
        
        # Build run metadata
        run_meta_dict = self.run_metadata
        run_metadata = RunMetadata(
            source_file=self.path,
            instrument_model=run_meta_dict.get('instrument_model'),
            instrument_serial=run_meta_dict.get('instrument_serial'),
            software_version=run_meta_dict.get('software_version'),
        )
        
        return MSRun(spectra=spectra, metadata=run_metadata)


def read_mzml(path: Path | str) -> MSRun:
    """
    Convenience function to read an mzML file into an MSRun.
    
    Args:
        path: Path to mzML or mzXML file.
        
    Returns:
        MSRun containing all spectra.
        
    Example:
        >>> run = read_mzml("sample.mzML")
        >>> print(f"Loaded {len(run)} spectra")
    """
    with MzMLReader(path) as reader:
        return reader.to_run()
