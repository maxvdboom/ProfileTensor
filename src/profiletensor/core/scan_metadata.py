"""
Scan metadata for LC-MS/MS spectra.

This module defines the ScanMetadata dataclass that captures all relevant
metadata for a single scan/spectrum, including acquisition parameters,
precursor information for MS2+ scans, and chromatographic context.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Polarity(Enum):
    """Ion polarity mode."""
    POSITIVE = auto()
    NEGATIVE = auto()
    UNKNOWN = auto()


class SpectrumType(Enum):
    """Spectrum data representation type."""
    PROFILE = auto()
    CENTROID = auto()
    UNKNOWN = auto()


class ActivationType(Enum):
    """Fragmentation/activation method for MS2+ scans."""
    CID = auto()      # Collision-Induced Dissociation
    HCD = auto()      # Higher-energy Collisional Dissociation
    ETD = auto()      # Electron Transfer Dissociation
    ECD = auto()      # Electron Capture Dissociation
    UVPD = auto()     # Ultraviolet Photodissociation
    IRMPD = auto()    # Infrared Multiphoton Dissociation
    PQD = auto()      # Pulsed Q Dissociation
    BIRD = auto()     # Blackbody Infrared Radiative Dissociation
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class PrecursorInfo:
    """
    Precursor ion information for MS2+ spectra.
    
    Attributes:
        mz: Precursor m/z value (selected for fragmentation).
        charge: Charge state (None if unknown).
        intensity: Precursor intensity in MS1 scan (None if unknown).
        isolation_window_lower: Lower offset of isolation window in Da.
        isolation_window_upper: Upper offset of isolation window in Da.
        activation_type: Fragmentation method used.
        collision_energy: Collision energy value (eV or normalized).
        collision_energy_unit: Unit of collision energy ("eV", "NCE", etc.).
        parent_scan_number: Scan number of the parent MS1 scan.
    """
    mz: float
    charge: Optional[int] = None
    intensity: Optional[float] = None
    isolation_window_lower: Optional[float] = None
    isolation_window_upper: Optional[float] = None
    activation_type: ActivationType = ActivationType.UNKNOWN
    collision_energy: Optional[float] = None
    collision_energy_unit: Optional[str] = None
    parent_scan_number: Optional[int] = None

    @property
    def isolation_window_width(self) -> Optional[float]:
        """Total isolation window width in Da."""
        if self.isolation_window_lower is not None and self.isolation_window_upper is not None:
            return self.isolation_window_lower + self.isolation_window_upper
        return None


@dataclass(frozen=True, slots=True)
class ScanMetadata:
    """
    Comprehensive metadata for a single MS scan.
    
    This dataclass captures all relevant information about a scan's
    acquisition parameters and context within the LC-MS run.
    
    Attributes:
        scan_number: Unique scan identifier (1-based, vendor-assigned).
        ms_level: MS level (1 for MS1, 2 for MS2, etc.).
        retention_time: Retention time in seconds.
        polarity: Ion polarity mode.
        spectrum_type: Profile or centroid mode.
        
        # Scan range
        scan_window_lower: Lower m/z limit of scan range.
        scan_window_upper: Upper m/z limit of scan range.
        
        # Acquisition details
        total_ion_current: Total ion current (TIC) for the scan.
        base_peak_mz: m/z of the base (most intense) peak.
        base_peak_intensity: Intensity of the base peak.
        injection_time: Ion injection time in milliseconds.
        
        # Precursor information (for MS2+ only)
        precursor: Precursor information for MSn scans.
        
        # Additional vendor-specific metadata
        filter_string: Vendor-specific scan filter string.
        native_id: Native spectrum ID from source file.
        extras: Additional metadata not covered by standard fields.
    """
    # Required fields
    scan_number: int
    ms_level: int
    retention_time: float  # in seconds
    
    # Polarity and type
    polarity: Polarity = Polarity.UNKNOWN
    spectrum_type: SpectrumType = SpectrumType.UNKNOWN
    
    # Scan range
    scan_window_lower: Optional[float] = None
    scan_window_upper: Optional[float] = None
    
    # Acquisition metrics
    total_ion_current: Optional[float] = None
    base_peak_mz: Optional[float] = None
    base_peak_intensity: Optional[float] = None
    injection_time: Optional[float] = None  # milliseconds
    
    # Precursor info (for MS2+)
    precursor: Optional[PrecursorInfo] = None
    
    # Vendor-specific
    filter_string: Optional[str] = None
    native_id: Optional[str] = None
    extras: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metadata consistency."""
        if self.scan_number < 1:
            raise ValueError(f"scan_number must be >= 1, got {self.scan_number}")
        if self.ms_level < 1:
            raise ValueError(f"ms_level must be >= 1, got {self.ms_level}")
        if self.retention_time < 0:
            raise ValueError(f"retention_time must be >= 0, got {self.retention_time}")
        if self.ms_level > 1 and self.precursor is None:
            # Warning: MS2+ without precursor info - this is allowed but unusual
            pass
    
    @property
    def is_ms1(self) -> bool:
        """Check if this is an MS1 scan."""
        return self.ms_level == 1
    
    @property
    def is_msn(self) -> bool:
        """Check if this is an MSn (n > 1) scan."""
        return self.ms_level > 1
    
    @property
    def retention_time_minutes(self) -> float:
        """Retention time in minutes."""
        return self.retention_time / 60.0
    
    @property
    def scan_window_width(self) -> Optional[float]:
        """Width of the scan window in Da."""
        if self.scan_window_lower is not None and self.scan_window_upper is not None:
            return self.scan_window_upper - self.scan_window_lower
        return None
