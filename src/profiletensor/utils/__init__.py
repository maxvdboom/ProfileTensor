"""
Utility modules for ProfileTensor.

This module provides:
- System resource detection
- External tool management (Apptainer, ProteoWizard)
- Parallelization helpers
"""

from .external import (
    ParallelMode,
    SystemResources,
    get_system_resources,
    find_apptainer,
    find_sif_file,
    validate_sif_file,
    check_apptainer_available,
)

__all__ = [
    "ParallelMode",
    "SystemResources",
    "get_system_resources",
    "find_apptainer",
    "find_sif_file",
    "validate_sif_file",
    "check_apptainer_available",
]
