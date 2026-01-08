from pathlib import Path
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SpectrumReader

class Vendor(Enum):
    THERMO = auto()
    AGILENT = auto()
    BRUKER = auto()
    SCIEX = auto()
    SHIMADZU = auto()
    WATERS = auto()
    OPEN_FORMAT = auto()  # mzML, mzXML
    UNKNOWN = auto()

# Extension to vendor mapping
VENDOR_EXTENSIONS: dict[str, Vendor] = {
    # Thermo
    '.raw': Vendor.THERMO,
    # Agilent (folder-based)
    '.d': Vendor.AGILENT,  # Note: Bruker also uses .d - need content inspection
    '.yep': Vendor.AGILENT,
    # Sciex
    '.wiff': Vendor.SCIEX,
    '.wiff.scan': Vendor.SCIEX,
    # Shimadzu
    '.lcd': Vendor.SHIMADZU,
    # Waters
    '.raw': Vendor.WATERS,  # Collision with Thermo - need content inspection
    # Open formats
    '.mzml': Vendor.OPEN_FORMAT,
    '.mzxml': Vendor.OPEN_FORMAT,
    '.mgf': Vendor.OPEN_FORMAT,
}

def detect_vendor(path: Path) -> Vendor:
    """
    Detect vendor from file path and contents.
    
    Some formats require content inspection:
    - .raw: Thermo vs Waters (check magic bytes or internal structure)
    - .d folders: Agilent vs Bruker (check for analysis.tdf = Bruker)
    """
    path = Path(path)
    
    # Handle .wiff.scan compound extension
    if path.name.endswith('.wiff.scan'):
        return Vendor.SCIEX
    
    suffix = path.suffix.lower()
    
    # .d folders need special handling
    if suffix == '.d' and path.is_dir():
        return _detect_d_folder_vendor(path)
    
    # .raw files need disambiguation
    if suffix == '.raw' and path.is_file():
        return _detect_raw_vendor(path)
    
    return VENDOR_EXTENSIONS.get(suffix, Vendor.UNKNOWN)

def _detect_d_folder_vendor(path: Path) -> Vendor:
    """Distinguish Agilent .d from Bruker .d folders."""
    # Bruker timsTOF has analysis.tdf
    if (path / 'analysis.tdf').exists():
        return Vendor.BRUKER
    # Agilent has AcqData subfolder
    if (path / 'AcqData').exists():
        return Vendor.AGILENT
    # Check for other Bruker indicators
    if (path / 'analysis.baf').exists():
        return Vendor.BRUKER
    return Vendor.UNKNOWN

def _detect_raw_vendor(path: Path) -> Vendor:
    """Distinguish Thermo .raw from Waters .raw."""
    # Thermo RAW files have "Finnigan" magic bytes at specific offset
    # Waters .raw is actually a folder disguised as file on some systems
    try:
        with open(path, 'rb') as f:
            header = f.read(64)
            # Thermo signature
            if b'Finnigan' in header or b'Thermo' in header:
                return Vendor.THERMO
    except Exception:
        pass
    
    # Could be Waters (often folder-based on Windows)
    if path.is_dir():
        return Vendor.WATERS
    
    # Default to Thermo as most common
    return Vendor.THERMO


class ReaderRegistry:
    """Registry for spectrum readers with automatic vendor detection."""
    
    _readers: dict[Vendor, type['SpectrumReader']] = {}
    _fallback: type['SpectrumReader'] | None = None
    
    @classmethod
    def register(cls, vendor: Vendor):
        """Decorator to register a reader class for a vendor."""
        def decorator(reader_class: type['SpectrumReader']):
            cls._readers[vendor] = reader_class
            return reader_class
        return decorator
    
    @classmethod
    def register_fallback(cls, reader_class: type['SpectrumReader']):
        """Register the fallback reader (ProteoWizard bridge)."""
        cls._fallback = reader_class
        return reader_class
    
    @classmethod
    def get_reader(cls, path: Path | str) -> 'SpectrumReader':
        """Get appropriate reader for a file, with automatic vendor detection."""
        path = Path(path)
        vendor = detect_vendor(path)
        
        # Try vendor-specific reader first
        if vendor in cls._readers:
            reader_class = cls._readers[vendor]
            if reader_class.is_available():
                return reader_class(path)
        
        # Fall back to ProteoWizard bridge
        if cls._fallback and cls._fallback.is_available():
            return cls._fallback(path)
        
        raise RuntimeError(
            f"No reader available for {path} (detected vendor: {vendor.name}). "
            f"Install vendor-specific dependencies or ProteoWizard."
        )
    
    @classmethod
    def list_available(cls) -> dict[str, bool]:
        """List all readers and their availability status."""
        status = {}
        for vendor, reader in cls._readers.items():
            status[vendor.name] = reader.is_available()
        if cls._fallback:
            status['PROTEOwizard_FALLBACK'] = cls._fallback.is_available()
        return status