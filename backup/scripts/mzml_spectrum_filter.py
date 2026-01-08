#!/usr/bin/env python3
"""
mzML Spectrum Analysis and Filtering Script (Memory-Optimized)
"""

import os
import glob
import base64
import struct
import zlib
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gc


def parse_binary_array(binary_array, namespace, use_float32=True):
    """Parse a binary data array from mzML format."""
    cv_params = binary_array.findall('.//mzml:cvParam', namespace)
    
    is_mz = any(param.get('name') == 'm/z array' for param in cv_params)
    is_intensity = any(param.get('name') == 'intensity array' for param in cv_params)
    is_compressed = any(param.get('name') == 'zlib compression' for param in cv_params)
    is_64bit = any(param.get('name') == '64-bit float' for param in cv_params)
    
    binary_elem = binary_array.find('.//mzml:binary', namespace)
    if binary_elem is not None and binary_elem.text:
        encoded_data = binary_elem.text
        decoded_data = base64.b64decode(encoded_data)
        
        if is_compressed:
            decoded_data = zlib.decompress(decoded_data)
        
        fmt = 'd' if is_64bit else 'f'
        num_values = len(decoded_data) // (8 if is_64bit else 4)
        values = struct.unpack(f'<{num_values}{fmt}', decoded_data)
        
        # Use float32 to save memory
        dtype = np.float32 if use_float32 else np.float64
        return np.array(values, dtype=dtype), is_mz, is_intensity
    
    return None, is_mz, is_intensity


def round_mz(mz_values, decimals=4):
    """Round m/z values to reduce unique count and enable integer-based tracking."""
    return np.round(mz_values, decimals)


def process_file_pass1(mzml_file, namespace):
    """First pass: collect m/z values with non-zero intensity."""
    mz_nonzero = set()
    datapoint_counts = []
    
    tree = ET.parse(mzml_file)
    root = tree.getroot()
    spectra = root.findall('.//mzml:spectrum', namespace)
    
    for spectrum in spectra:
        binary_arrays = spectrum.findall('.//mzml:binaryDataArray', namespace)
        
        mz_array = None
        intensity_array = None
        
        for binary_array in binary_arrays:
            values, is_mz, is_intensity = parse_binary_array(binary_array, namespace)
            
            if values is not None:
                if is_mz:
                    mz_array = values
                elif is_intensity:
                    intensity_array = values
        
        if mz_array is not None and intensity_array is not None:
            datapoint_counts.append(len(mz_array))
            
            # Round m/z values and convert to integers (multiplied) for memory efficiency
            mz_rounded = round_mz(mz_array)
            nonzero_mask = intensity_array > 0
            # Store as integers (scaled) to save memory
            mz_nonzero.update((mz_rounded[nonzero_mask] * 10000).astype(np.int64).tolist())
    
    del tree, root, spectra
    gc.collect()
    
    return mz_nonzero, datapoint_counts


def process_file_pass2(mzml_file, namespace, mz_to_keep_set):
    """Second pass: count filtered datapoints per spectrum."""
    filtered_counts = []
    
    tree = ET.parse(mzml_file)
    root = tree.getroot()
    spectra = root.findall('.//mzml:spectrum', namespace)
    
    for spectrum in spectra:
        binary_arrays = spectrum.findall('.//mzml:binaryDataArray', namespace)
        
        mz_array = None
        intensity_array = None
        
        for binary_array in binary_arrays:
            values, is_mz, is_intensity = parse_binary_array(binary_array, namespace)
            
            if values is not None:
                if is_mz:
                    mz_array = values
                elif is_intensity:
                    intensity_array = values
        
        if mz_array is not None and intensity_array is not None:
            # Round and convert to scaled integers for matching
            mz_scaled = (round_mz(mz_array) * 10000).astype(np.int64)
            
            # Count matches using set lookup (much faster and memory-efficient)
            keep_count = sum(1 for mz in mz_scaled if mz in mz_to_keep_set)
            filtered_counts.append(keep_count)
    
    del tree, root, spectra
    gc.collect()
    
    return filtered_counts


def main():
    # Configuration
    mzml_dir = '/home/maxvandenboom/Documents/ProfileTensor/data/msnlib/mzml/20240408_pluskal_mcedrug_MSn_positive'
    namespace = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # Find all mzML files
    mzml_files = glob.glob(os.path.join(mzml_dir, '*.mzML'))
    print(f"Found {len(mzml_files)} mzML files in the directory\n")
    
    # First pass: collect all unique m/z values with non-zero intensity
    print("First pass: Identifying m/z values with zero intensity across all spectra...")
    mz_has_nonzero = set()
    original_datapoint_counts = []
    
    for mzml_file in sorted(mzml_files):
        filename = os.path.basename(mzml_file)
        print(f"Processing (pass 1): {filename}")
        
        file_mz, file_counts = process_file_pass1(mzml_file, namespace)
        mz_has_nonzero.update(file_mz)
        original_datapoint_counts.extend(file_counts)
        
        del file_mz, file_counts
        gc.collect()
    
    print(f"\nProcessed {len(original_datapoint_counts)} spectra")
    print(f"Found {len(mz_has_nonzero)} unique m/z values with non-zero intensity")
    
    # Second pass: filter spectra and compute statistics
    print("\nSecond pass: Filtering spectra...")
    filtered_datapoint_counts = []
    
    for mzml_file in sorted(mzml_files):
        filename = os.path.basename(mzml_file)
        print(f"Processing (pass 2): {filename}")
        
        file_filtered_counts = process_file_pass2(mzml_file, namespace, mz_has_nonzero)
        filtered_datapoint_counts.extend(file_filtered_counts)
        
        gc.collect()
    
    # Free the large set
    del mz_has_nonzero
    gc.collect()
    
    # Convert to numpy arrays for statistics
    filtered_datapoint_counts = np.array(filtered_datapoint_counts, dtype=np.int32)
    original_datapoint_counts = np.array(original_datapoint_counts, dtype=np.int32)
    
    # Print summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)
    orig_total = int(np.sum(original_datapoint_counts))
    filt_total = int(np.sum(filtered_datapoint_counts))
    print(f"Original total datapoints: {orig_total}")
    print(f"Filtered total datapoints: {filt_total}")
    print(f"Removed datapoints: {orig_total - filt_total}")
    removed_pct = 100 * (orig_total - filt_total) / orig_total
    print(f"Percentage removed: {removed_pct:.2f}%")
    
    print("\n" + "="*80)
    print("FILTERED DATAPOINTS PER SPECTRUM STATISTICS")
    print("="*80)
    print(f"Total spectra: {len(filtered_datapoint_counts)}")
    print(f"\nDatapoints per spectrum statistics:")
    print(f"  Mean: {np.mean(filtered_datapoint_counts):.2f}")
    print(f"  Median: {np.median(filtered_datapoint_counts):.2f}")
    print(f"  Std Dev: {np.std(filtered_datapoint_counts):.2f}")
    print(f"  Min: {int(np.min(filtered_datapoint_counts))}")
    print(f"  Max: {int(np.max(filtered_datapoint_counts))}")
    print(f"  Total datapoints: {filt_total}")
    
    print(f"\nQuartiles:")
    print(f"  25th percentile: {np.percentile(filtered_datapoint_counts, 25):.2f}")
    print(f"  75th percentile: {np.percentile(filtered_datapoint_counts, 75):.2f}")
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(original_datapoint_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Datapoints')
    plt.ylabel('Frequency')
    plt.title('Original Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.hist(filtered_datapoint_counts, bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Number of Datapoints')
    plt.ylabel('Frequency')
    plt.title('Filtered Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.hist(original_datapoint_counts, bins=50, edgecolor='black', alpha=0.5, label='Original')
    plt.hist(filtered_datapoint_counts, bins=50, edgecolor='black', alpha=0.5, color='green', label='Filtered')
    plt.xlabel('Number of Datapoints')
    plt.ylabel('Frequency')
    plt.title('Overlay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.boxplot(original_datapoint_counts, vert=True)
    plt.ylabel('Number of Datapoints')
    plt.title('Original Boxplot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.boxplot(filtered_datapoint_counts, vert=True)
    plt.ylabel('Number of Datapoints')
    plt.title('Filtered Boxplot')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.boxplot([original_datapoint_counts, filtered_datapoint_counts])
    plt.xticks([1, 2], ['Original', 'Filtered'])
    plt.ylabel('Number of Datapoints')
    plt.title('Comparison Boxplot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mzml_filtering_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'mzml_filtering_analysis.png'")


if __name__ == "__main__":
    main()