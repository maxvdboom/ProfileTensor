import os
import gzip
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from ftplib import FTP
import time
import subprocess
import shutil
import json
import urllib.parse
from urllib.parse import quote

# Parser backend (try lxml, fallback to xml.etree.ElementTree)
try:
    from lxml import etree as ET
    PARSER_BACKEND = "lxml"
    def get_iterparse_kwargs():
        return dict(resolve_entities=False, no_network=True, huge_tree=False, recover=False, remove_pis=True, remove_comments=True)
except Exception:
    import xml.etree.ElementTree as ET
    PARSER_BACKEND = "xml.etree.ElementTree"
    def get_iterparse_kwargs():
        return {}

def load_study_raw_files_json(json_file_path: str) -> dict:
    """
    Load the study raw files JSON data.
    
    Args:
        json_file_path: Path to the JSON file containing study raw file mappings
        
    Returns:
        dict: Dictionary mapping study IDs to lists of raw file URLs
    """
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file {json_file_path}: {e}")
        return {}

def get_file_size_from_ftp(ftp: FTP, ftp_url: str) -> int:
    """
    Get file size from FTP URL.
    
    Args:
        ftp: Active FTP connection
        ftp_url: Full FTP URL to the file
        
    Returns:
        int: File size in bytes, 0 if failed
    """
    try:
        parsed_url = urllib.parse.urlparse(ftp_url)
        # Properly encode the path, especially spaces
        file_path = quote(parsed_url.path, safe='/')
        
        ftp.voidcmd("TYPE I")
        size = ftp.size(file_path)
        return size if size else 0
    except Exception as e:
        print(f"  ⚠️  Could not get size for {ftp_url}: {e}")
        return 0

def find_largest_raw_files_from_json(ftp: FTP, study_id: str, study_raw_files: dict, max_files: int = 5) -> list:
    """
    Find the largest raw vendor files for a study from the JSON data.
    
    Args:
        ftp: Active FTP connection
        study_id: Study ID to look up
        study_raw_files: Dictionary mapping study IDs to raw file URLs
        max_files: Maximum number of files to return
    
    Returns:
        List of dicts with file info: [{'found': bool, 'url': str, 'filename': str, 'size': int, 'vendor': str, 'companion_file': str}]
    """
    result = []
    
    if study_id not in study_raw_files:
        print(f"  ❌ Study {study_id} not found in JSON data")
        return result
    
    raw_file_urls = study_raw_files[study_id]
    
    if not raw_file_urls:
        print(f"  ❌ No raw files found for study {study_id}")
        return result
    
    print(f"  🔍 Found {len(raw_file_urls)} raw files in JSON for {study_id}")
    
    # Raw file extensions and their associated vendors
    raw_extensions = {
        '.raw': 'Thermo',
        '.d': 'Agilent/Bruker', 
        '.wiff': 'Sciex',
        '.wiff.scan': 'Sciex',
        '.lcd': 'Shimadzu',
        '.yep': 'Agilent'
    }
    
    # Improved Sciex file pairing - use filename without directory path for matching
    wiff_files = {}  # base_name -> {'url': url, 'size': size}
    wiff_scan_files = {}  # base_name -> {'url': url, 'size': size}
    other_files = []
    
    for url in raw_file_urls:
        try:
            filename = os.path.basename(url)
            filename_lower = filename.lower()
            
            if filename_lower.endswith('.wiff.scan'):
                # This is a .wiff.scan file
                base_name = filename[:-10]  # Remove '.wiff.scan'
                wiff_scan_files[base_name] = {'url': url}
            elif filename_lower.endswith('.wiff'):
                # This is a .wiff file
                base_name = filename[:-5]  # Remove '.wiff'
                wiff_files[base_name] = {'url': url}
            else:
                # Other file types
                other_files.append(url)
            
        except Exception as e:
            print(f"  ⚠️  Error processing {url}: {e}")
            continue
    
    # Process Sciex file pairs and standalone files
    file_info_list = []
    processed_wiff_bases = set()
    
    # First, try to pair .wiff and .wiff.scan files
    for base_name in wiff_files.keys():
        if base_name in wiff_scan_files:
            # Perfect pair found
            wiff_url = wiff_files[base_name]['url']
            wiff_scan_url = wiff_scan_files[base_name]['url']
            
            try:
                # Get size of the .wiff file (main file)
                size = get_file_size_from_ftp(ftp, wiff_url)
                
                if size > 0:
                    file_info = {
                        'found': True,
                        'url': wiff_url,
                        'filename': os.path.basename(wiff_url),
                        'size': size,
                        'vendor': 'Sciex',
                        'companion_file': wiff_scan_url
                    }
                    file_info_list.append(file_info)
                    print(f"    • {base_name}.wiff + .wiff.scan: {size/1024/1024:.1f} MB (Sciex pair)")
                    processed_wiff_bases.add(base_name)
                
            except Exception as e:
                print(f"  ⚠️  Error processing Sciex pair {base_name}: {e}")
                continue
    
    # Process standalone .wiff files (without matching .wiff.scan)
    for base_name, wiff_info in wiff_files.items():
        if base_name not in processed_wiff_bases:
            wiff_url = wiff_info['url']
            try:
                size = get_file_size_from_ftp(ftp, wiff_url)
                
                if size > 0:
                    file_info = {
                        'found': True,
                        'url': wiff_url,
                        'filename': os.path.basename(wiff_url),
                        'size': size,
                        'vendor': 'Sciex',
                        'companion_file': None
                    }
                    file_info_list.append(file_info)
                    print(f"    • {base_name}.wiff: {size/1024/1024:.1f} MB (Sciex standalone)")
                    if base_name in wiff_scan_files:
                        print(f"      ⚠️  Found corresponding .wiff.scan but processing .wiff standalone")
                
            except Exception as e:
                print(f"  ⚠️  Error processing standalone .wiff {base_name}: {e}")
                continue
    
    # Process standalone .wiff.scan files (without matching .wiff)
    for base_name, scan_info in wiff_scan_files.items():
        if base_name not in processed_wiff_bases and base_name not in wiff_files:
            scan_url = scan_info['url']
            try:
                size = get_file_size_from_ftp(ftp, scan_url)
                
                if size > 0:
                    file_info = {
                        'found': True,
                        'url': scan_url,
                        'filename': os.path.basename(scan_url),
                        'size': size,
                        'vendor': 'Sciex',
                        'companion_file': None
                    }
                    file_info_list.append(file_info)
                    print(f"    • {base_name}.wiff.scan: {size/1024/1024:.1f} MB (Sciex scan file only)")
                
            except Exception as e:
                print(f"  ⚠️  Error processing standalone .wiff.scan {base_name}: {e}")
                continue
    
    # Process other file types
    for url in other_files:
        try:
            filename = os.path.basename(url)
            filename_lower = filename.lower()
            
            # Determine vendor from extension
            vendor = 'Unknown'
            for ext, vendor_name in raw_extensions.items():
                if filename_lower.endswith(ext):
                    vendor = vendor_name
                    break
            
            # Get file size
            size = get_file_size_from_ftp(ftp, url)
            
            if size > 0:
                file_info = {
                    'found': True,
                    'url': url,
                    'filename': filename,
                    'size': size,
                    'vendor': vendor,
                    'companion_file': None
                }
                file_info_list.append(file_info)
                print(f"    • {filename}: {size/1024/1024:.1f} MB ({vendor})")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {url}: {e}")
            continue
    
    # Sort by size (largest first) and take the top files
    file_info_list.sort(key=lambda x: x['size'], reverse=True)
    result = file_info_list[:max_files]
    
    print(f"  ✅ Selected {len(result)} largest files (max {max_files})")
    
    return result

def download_raw_file_from_ftp_url(ftp: FTP, file_info: dict, temp_dir: str) -> str:
    """
    Download a raw vendor file from FTP URL. For Sciex files, also downloads the companion .wiff.scan file.
    
    Args:
        ftp: Active FTP connection
        file_info: File information dict with 'url', 'filename', 'size', and optionally 'companion_file'
        temp_dir: Temporary directory to download to
        
    Returns:
        str: Path to downloaded file, or empty string if failed
    """
    if not file_info.get('found', False):
        return ""
    
    local_path = os.path.join(temp_dir, file_info['filename'])
    
    try:
        print(f"    📥 Downloading {file_info['filename']} ({file_info['size']/1024/1024:.1f} MB)...")
        
        # Parse the FTP URL to get the path
        parsed_url = urllib.parse.urlparse(file_info['url'])
        remote_path = parsed_url.path
        
        if file_info['filename'].lower().endswith('.d'):
            # Handle .d directories - download recursively
            os.makedirs(local_path, exist_ok=True)
            
            # Navigate to the .d directory on FTP
            original_path = ftp.pwd()
            try:
                ftp.cwd(os.path.dirname(remote_path))
                d_name = os.path.basename(remote_path)
                ftp.cwd(d_name)
                
                def download_directory_recursive(local_subpath=""):
                    items = ftp.nlst()
                    for item in items:
                        local_item_path = os.path.join(local_path, local_subpath, item)
                        
                        try:
                            # Try to get file size (if it's a file)
                            size = ftp.size(item)
                            if size is not None:
                                # It's a file, download it
                                os.makedirs(os.path.dirname(local_item_path), exist_ok=True)
                                with open(local_item_path, 'wb') as f:
                                    ftp.retrbinary(f"RETR {item}", f.write, blocksize=8192)
                        except:
                            # Might be a directory
                            try:
                                ftp.cwd(item)
                                os.makedirs(local_item_path, exist_ok=True)
                                download_directory_recursive(os.path.join(local_subpath, item))
                                ftp.cwd('..')
                            except:
                                continue
                
                download_directory_recursive()
                ftp.cwd(original_path)
                
            except Exception as e:
                ftp.cwd(original_path)
                raise e
            
        else:
            # Handle regular files
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {remote_path}", f.write, blocksize=8192)
            
            # For Sciex files, also download the companion .wiff.scan file
            if file_info.get('companion_file'):
                companion_url = file_info['companion_file']
                companion_filename = os.path.basename(companion_url)
                companion_local_path = os.path.join(temp_dir, companion_filename)
                
                print(f"    📥 Downloading companion file {companion_filename}...")
                
                # Parse the companion FTP URL
                companion_parsed_url = urllib.parse.urlparse(companion_url)
                companion_remote_path = companion_parsed_url.path
                
                with open(companion_local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {companion_remote_path}", f.write, blocksize=8192)
                
                print(f"    ✅ Downloaded companion file successfully")
        
        print(f"    ✅ Downloaded successfully")
        return local_path
        
    except Exception as e:
        print(f"    ❌ Error downloading file: {e}")
        return ""

def convert_raw_to_mzml_with_msconvert(raw_file_path: str, output_dir: str, container_path: str = "pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif") -> str:
    """
    Convert raw vendor file to mzML using msconvert in apptainer container.
    For Sciex .wiff files, ensures the corresponding .wiff.scan file is in the same directory.
    
    Args:
        raw_file_path: Path to the raw vendor file
        output_dir: Directory to save the converted mzML file
        container_path: Path to the apptainer container
        
    Returns:
        str: Path to converted mzML file, or empty string if failed
    """
    try:
        print(f"  🔄 Converting {os.path.basename(raw_file_path)} to mzML...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare the command
        # Mount the directory containing the raw file and output directory
        raw_dir = os.path.dirname(raw_file_path)
        raw_filename = os.path.basename(raw_file_path)
        
        # For Sciex .wiff files, verify that the .wiff.scan file is present
        if raw_filename.lower().endswith('.wiff'):
            base_name = raw_filename[:-5]  # Remove '.wiff'
            scan_filename = f"{base_name}.wiff.scan"
            scan_path = os.path.join(raw_dir, scan_filename)
            
            if not os.path.exists(scan_path):
                print(f"  ❌ Missing required companion file: {scan_filename}")
                return ""
            else:
                print(f"  ✅ Found companion file: {scan_filename}")
        
        cmd = [
            "apptainer", "exec",
            "--bind", f"{raw_dir}:/input",
            "--bind", f"{output_dir}:/output",
            container_path,
            "wine", "msconvert",
            f"/input/{raw_filename}",
            "-o", "/output"
        ]
        
        print(f"  🔧 Running: {' '.join(cmd)}")
        
        # Run the conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            # Find the generated mzML file
            base_name = os.path.splitext(raw_filename)[0]
            if raw_filename.lower().endswith('.d'):
                base_name = raw_filename[:-2]  # Remove .d extension
            
            mzml_path = os.path.join(output_dir, f"{base_name}.mzML")
            
            if os.path.exists(mzml_path):
                print(f"  ✅ Conversion successful: {mzml_path}")
                return mzml_path
            else:
                print(f"  ❌ Conversion completed but mzML file not found")
                return ""
        else:
            print(f"  ❌ Conversion failed: {result.stderr}")
            return ""
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Conversion timed out")
        return ""
    except Exception as e:
        print(f"  ❌ Error during conversion: {e}")
        return ""

def check_msms_in_file(file_path: str) -> bool:
    """
    Check if a mzML/mzXML file contains MS/MS data by looking for 'ms level" value="2"'.
    Handles both regular and gzipped files.
    """
    try:
        # Determine if file is gzipped
        is_gzipped = file_path.lower().endswith('.gz')
        
        # Read file in chunks to avoid loading entire file into memory
        chunk_size = 8192  # 8KB chunks
        search_string = b'"ms level" value="2"'  # Search as bytes for efficiency
        
        open_func = gzip.open if is_gzipped else open
        
        with open_func(file_path, 'rb') as f:
            buffer = b''
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Combine with previous buffer to handle split matches
                buffer += chunk
                
                # Check for the search string
                if search_string in buffer:
                    return True
                
                # Keep last part of buffer to handle matches across chunks
                # Keep enough to handle the search string split across chunks
                if len(buffer) > len(search_string):
                    buffer = buffer[-(len(search_string)-1):]
                
        return False
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    
def check_acquisition_mode(mzml_file_path: str) -> dict:
    """
    Analyzes an mzML file to determine the acquisition mode (DDA vs DIA).
    Searches for isolation window offset values to determine mode:
    - If isolation window offsets > 1.0: DIA
    - If isolation window offsets <= 1.0: DDA
    
    Args:
        mzml_file_path (str): Path to the mzML file.
    
    Returns:
        dict: {
            'mode': str,  # "DDA" or "DIA"
            'confidence': str,  # "High" or "Low"
            'ms2_count': int,  # Not used in simplified version
            'unique_precursors': int,  # Not used in simplified version
            'isolation_window_analysis': dict,  # Contains offset statistics
            'details': str  # Human-readable explanation
        }
    """
    try:
        result = {
            'mode': 'Unknown',  # Default to Unknown
            'confidence': 'Low',
            'ms2_count': 0,
            'unique_precursors': 0,
            'isolation_window_analysis': {},
            'details': 'Isolation window offset-based detection using XML search'
        }
        
        print(f"  🔍 Analyzing acquisition mode using isolation window offsets...")
        
        # Determine if file is gzipped
        is_gzipped = mzml_file_path.lower().endswith('.gz')
        
        # Search patterns for isolation window offsets
        chunk_size = 8192  # 8KB chunks
        lower_offset_pattern = b'<cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="'
        upper_offset_pattern = b'<cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="'
        
        open_func = gzip.open if is_gzipped else open
        
        lower_offsets = []
        upper_offsets = []
        
        with open_func(mzml_file_path, 'rb') as f:
            buffer = b''
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Combine with previous buffer to handle split matches
                buffer += chunk
                
                # Search for lower offset values
                start_pos = 0
                while True:
                    pos = buffer.find(lower_offset_pattern, start_pos)
                    if pos == -1:
                        break
                    
                    # Find the end quote of the value
                    value_start = pos + len(lower_offset_pattern)
                    value_end = buffer.find(b'"', value_start)
                    
                    if value_end != -1:
                        try:
                            value_str = buffer[value_start:value_end].decode('utf-8')
                            value = float(value_str)
                            lower_offsets.append(value)
                        except (ValueError, UnicodeDecodeError):
                            pass
                    
                    start_pos = pos + 1
                
                # Search for upper offset values
                start_pos = 0
                while True:
                    pos = buffer.find(upper_offset_pattern, start_pos)
                    if pos == -1:
                        break
                    
                    # Find the end quote of the value
                    value_start = pos + len(upper_offset_pattern)
                    value_end = buffer.find(b'"', value_start)
                    
                    if value_end != -1:
                        try:
                            value_str = buffer[value_start:value_end].decode('utf-8')
                            value = float(value_str)
                            upper_offsets.append(value)
                        except (ValueError, UnicodeDecodeError):
                            pass
                    
                    start_pos = pos + 1
                
                # Keep last part of buffer to handle matches across chunks
                max_pattern_len = max(len(lower_offset_pattern), len(upper_offset_pattern)) + 50  # Extra space for value
                if len(buffer) > max_pattern_len:
                    buffer = buffer[-(max_pattern_len-1):]
        
        # Analyze the collected offset values
        all_offsets = lower_offsets + upper_offsets
        
        if all_offsets:
            max_offset = max(all_offsets)
            min_offset = min(all_offsets)
            avg_offset = sum(all_offsets) / len(all_offsets)
            
            # Store isolation window analysis
            result['isolation_window_analysis'] = {
                'lower_offsets': lower_offsets,
                'upper_offsets': upper_offsets,
                'all_offsets': all_offsets,
                'max_offset': max_offset,
                'min_offset': min_offset,
                'avg_offset': avg_offset,
                'num_offsets_found': len(all_offsets)
            }
            
            # Determine acquisition mode based on offset values
            if max_offset > 1.0:
                result['mode'] = 'DIA'
                result['confidence'] = 'High'
                result['details'] = f'Found isolation window offsets > 1.0 (max: {max_offset:.2f}, avg: {avg_offset:.2f}) - indicates DIA acquisition'
            else:
                result['mode'] = 'DDA'
                result['confidence'] = 'High'
                result['details'] = f'Found isolation window offsets <= 1.0 (max: {max_offset:.2f}, avg: {avg_offset:.2f}) - indicates DDA acquisition'
            
            print(f"  📊 Found {len(all_offsets)} isolation window offset values")
            print(f"  📊 Offset range: {min_offset:.2f} - {max_offset:.2f} (avg: {avg_offset:.2f})")
        else:
            result['mode'] = 'Unknown'
            result['confidence'] = 'Low'
            result['details'] = 'No isolation window offset values found in mzML file'
            print(f"  ⚠️  No isolation window offset values found")
        
        print(f"  📊 Acquisition mode: {result['mode']} (confidence: {result['confidence']})")
        print(f"     Detection method: Isolation window offset analysis")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error analyzing acquisition mode: {e}")
        result = {
            'mode': 'Error',
            'confidence': 'Low',
            'ms2_count': 0,
            'unique_precursors': 0,
            'isolation_window_analysis': {},
            'details': f"Error during analysis: {str(e)}"
        }
        return result
    

def check_scan_mode(mzml_file_path: str) -> str:
    """
    Analyzes an mzML file to determine the scan mode (positive or negative).
    Searches for specific cvParam patterns to determine mode:
    - MS:1000130 "positive scan": returns 'positive'
    - MS:1000129 "negative scan": returns 'negative'
    - If both or neither found: returns 'unknown'
    
    Args:
        mzml_file_path (str): Path to the mzML file.
    
    Returns:
        str: "positive", "negative", or "unknown"
    """
    try:
        print(f"  🔍 Analyzing scan mode...")
        
        # Determine if file is gzipped
        is_gzipped = mzml_file_path.lower().endswith('.gz')
        
        # Search patterns for positive and negative scan modes
        chunk_size = 8192  # 8KB chunks
        positive_pattern = b'<cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>'
        negative_pattern = b'<cvParam cvRef="MS" accession="MS:1000129" name="negative scan" value=""/>'
        
        open_func = gzip.open if is_gzipped else open
        
        found_positive = False
        found_negative = False
        
        with open_func(mzml_file_path, 'rb') as f:
            buffer = b''
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Combine with previous buffer to handle split matches
                buffer += chunk
                
                # Check for both patterns
                if positive_pattern in buffer:
                    found_positive = True
                if negative_pattern in buffer:
                    found_negative = True
                
                # If we found both, we can stop early
                if found_positive and found_negative:
                    break
                
                # Keep last part of buffer to handle matches across chunks
                # Keep enough to handle the longest search pattern split across chunks
                max_pattern_len = max(len(positive_pattern), len(negative_pattern))
                if len(buffer) > max_pattern_len:
                    buffer = buffer[-(max_pattern_len-1):]
        
        # Determine result based on what was found
        if found_positive and not found_negative:
            result = 'positive'
        elif found_negative and not found_positive:
            result = 'negative'
        else:
            result = 'unknown'  # Both found, neither found, or error
        
        print(f"  📊 Scan mode: {result}")
        return result
        
    except Exception as e:
        print(f"  ❌ Error analyzing scan mode: {e}")
        return 'unknown'
    
def extract_mz_range_from_mzml(mzml_file_path: str) -> dict:
    """
    Extract min and max m/z values from base peak m/z values in an mzML file.
    
    Args:
        mzml_file_path: Path to the mzML file
        
    Returns:
        dict: {'min_mz': float, 'max_mz': float, 'mz_values_count': int}
    """
    try:
        print(f"  🔍 Extracting m/z range from base peak values...")
        
        # Determine if file is gzipped
        is_gzipped = mzml_file_path.lower().endswith('.gz')
        
        # Search pattern for base peak m/z values
        chunk_size = 8192  # 8KB chunks
        base_peak_pattern = b'<cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="'
        
        open_func = gzip.open if is_gzipped else open
        
        mz_values = []
        
        with open_func(mzml_file_path, 'rb') as f:
            buffer = b''
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Combine with previous buffer to handle split matches
                buffer += chunk
                
                # Search for base peak m/z values
                start_pos = 0
                while True:
                    pos = buffer.find(base_peak_pattern, start_pos)
                    if pos == -1:
                        break
                    
                    # Find the end quote of the value
                    value_start = pos + len(base_peak_pattern)
                    value_end = buffer.find(b'"', value_start)
                    
                    if value_end != -1:
                        try:
                            value_str = buffer[value_start:value_end].decode('utf-8')
                            value = float(value_str)
                            mz_values.append(value)
                        except (ValueError, UnicodeDecodeError):
                            pass
                    
                    start_pos = pos + 1
                
                # Keep last part of buffer to handle matches across chunks
                max_pattern_len = len(base_peak_pattern) + 50  # Extra space for value
                if len(buffer) > max_pattern_len:
                    buffer = buffer[-(max_pattern_len-1):]
        
        # Calculate min and max
        if mz_values:
            min_mz = min(mz_values)
            max_mz = max(mz_values)
            print(f"  📊 Found {len(mz_values)} base peak m/z values")
            print(f"  📊 m/z range: {min_mz:.2f} - {max_mz:.2f}")
            
            return {
                'min_mz': min_mz,
                'max_mz': max_mz,
                'mz_values_count': len(mz_values)
            }
        else:
            print(f"  ⚠️  No base peak m/z values found")
            return {
                'min_mz': None,
                'max_mz': None,
                'mz_values_count': 0
            }
        
    except Exception as e:
        print(f"  ❌ Error extracting m/z range: {e}")
        return {
            'min_mz': None,
            'max_mz': None,
            'mz_values_count': 0
        }

def download_convert_and_check_msms_metabolights_multiple(ftp: FTP, file_info_list: list, temp_base_dir: str, container_path: str) -> dict:
    """
    Download multiple raw vendor files, convert them to mzML, check for MS/MS data, and analyze acquisition mode.
    
    Args:
        ftp: Active FTP connection
        file_info_list: List of file information dicts
        temp_base_dir: Base directory for temporary files
        container_path: Path to the msconvert container
    
    Returns:
        dict: {
            'has_msms': bool,
            'acquisition_mode': dict,  # Result from check_acquisition_mode for the first file with MS/MS
            'scan_mode': str,
            'files_processed': int,
            'files_with_msms': int,
            'msms_files': list,  # List of filenames that contained MS/MS
            'mz_range': dict  # Min and max m/z values across all processed files
        }
    """
    result = {
        'has_msms': False,
        'acquisition_mode': {
            'mode': 'Unknown',
            'confidence': 'Low',
            'ms2_count': 0,
            'unique_precursors': 0,
            'isolation_window_analysis': {},
            'details': 'No analysis performed'
        },
        'scan_mode': 'unknown',
        'files_processed': 0,
        'files_with_msms': 0,
        'msms_files': [],
        'mz_range': {
            'min_mz': None,
            'max_mz': None,
            'mz_values_count': 0
        }
    }
    
    if not file_info_list:
        return result
    
    # Create temporary directories
    raw_temp_dir = os.path.join(temp_base_dir, "raw")
    mzml_temp_dir = os.path.join(temp_base_dir, "mzml")
    os.makedirs(raw_temp_dir, exist_ok=True)
    os.makedirs(mzml_temp_dir, exist_ok=True)
    
    acquisition_analysis = None
    scan_mode = 'unknown'
    
    # Track m/z values across all files
    all_min_mz = []
    all_max_mz = []
    total_mz_values = 0
    
    try:
        for i, file_info in enumerate(file_info_list):
            print(f"  📁 Processing file {i+1}/{len(file_info_list)}: {file_info['filename']}")
            
            # Step 1: Download the raw file
            raw_file_path = download_raw_file_from_ftp_url(ftp, file_info, raw_temp_dir)
            if not raw_file_path:
                print(f"    ❌ Failed to download {file_info['filename']}")
                continue
            
            # Step 2: Convert to mzML
            mzml_file_path = convert_raw_to_mzml_with_msconvert(raw_file_path, mzml_temp_dir, container_path)
            if not mzml_file_path:
                print(f"    ❌ Failed to convert {file_info['filename']}")
                continue
            
            result['files_processed'] += 1
            
            # Step 3: Check for MS/MS data
            print(f"    🔍 Checking for MS/MS data...")
            has_msms = check_msms_in_file(mzml_file_path)
            
            if has_msms:
                result['has_msms'] = True
                result['files_with_msms'] += 1
                result['msms_files'].append(file_info['filename'])
                print(f"    ✅ MS/MS data found in {file_info['filename']}!")
                
                # Step 4: If this is the first file with MS/MS, analyze acquisition mode
                if acquisition_analysis is None:
                    acquisition_analysis = check_acquisition_mode(mzml_file_path)
                    result['acquisition_mode'] = acquisition_analysis
                
                # Step 5: Check scan mode (use the first file with MS/MS)
                if scan_mode == 'unknown':
                    scan_mode = check_scan_mode(mzml_file_path)
                    result['scan_mode'] = scan_mode
            else:
                print(f"    ❌ No MS/MS data found in {file_info['filename']}")
            
            # Step 6: Extract m/z range from this file (regardless of MS/MS presence)
            mz_range = extract_mz_range_from_mzml(mzml_file_path)
            if mz_range['min_mz'] is not None and mz_range['max_mz'] is not None:
                all_min_mz.append(mz_range['min_mz'])
                all_max_mz.append(mz_range['max_mz'])
                total_mz_values += mz_range['mz_values_count']
            
            # Clean up this file's temporary data to save space
            try:
                if os.path.exists(raw_file_path):
                    if os.path.isdir(raw_file_path):
                        shutil.rmtree(raw_file_path)
                    else:
                        os.remove(raw_file_path)
                if os.path.exists(mzml_file_path):
                    os.remove(mzml_file_path)
            except:
                pass
        
        # Calculate overall m/z range across all processed files
        if all_min_mz and all_max_mz:
            # Use the overall minimum and maximum across all files
            result['mz_range'] = {
                'min_mz': min(all_min_mz),
                'max_mz': max(all_max_mz),
                'mz_values_count': total_mz_values
            }
            print(f"  📊 Overall m/z range across {len(all_min_mz)} files: {result['mz_range']['min_mz']:.2f} - {result['mz_range']['max_mz']:.2f}")
        
        # If no MS/MS was found in any file, still check scan mode from the first converted file
        if scan_mode == 'unknown' and result['files_processed'] > 0:
            # We'll have to re-convert one file just for scan mode analysis
            # This is a fallback - ideally we'd save this info during the main loop
            pass
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error in download/convert/check pipeline: {e}")
        return result
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(raw_temp_dir):
                shutil.rmtree(raw_temp_dir)
            if os.path.exists(mzml_temp_dir):
                shutil.rmtree(mzml_temp_dir)
        except:
            pass

def test_metabolights_ftp_connection():
    """
    Test MetaboLights FTP connection and return working configuration.
    """
    print("🔍 Testing MetaboLights FTP connection...")
    
    config = {
        'host': 'ftp.ebi.ac.uk',
        'user': '',  # Anonymous
        'passwd': ''  # Anonymous
    }
    
    try:
        with FTP(config['host'], timeout=30) as ftp:
            ftp.login(user=config['user'], passwd=config['passwd'])
            print(f"✅ FTP connection successful to {config['host']}")
            return config
    except Exception as e:
        print(f"❌ FTP connection failed: {e}")
        return None

def verify_msms_metabolights_datasets_with_conversion(
    csv_file: str, 
    json_file: str,
    output_file: str = 'metabolights_datasets_msms_verified_converted.csv', 
    no_msfile_output: str = 'metabolights_datasets_no_rawfiles.csv',
    container_path: str = "pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif",
    temp_base_dir: str = "/tmp/metabolights_conversion",
    max_files_per_study: int = 5
):
    """
    Verify MS/MS data for MetaboLights datasets by downloading raw files and converting them.
    
    Args:
        csv_file: Input CSV file with dataset metadata
        json_file: JSON file with study raw file mappings
        output_file: Output file for datasets with MS/MS data
        no_msfile_output: Output file for datasets without raw files
        container_path: Path to the msconvert apptainer container
        temp_base_dir: Base directory for temporary files
        max_files_per_study: Maximum number of files to process per study
    """
    # Load the dataset information
    df = pd.read_csv(csv_file)
    print(f"📋 Loaded {len(df)} datasets from {csv_file}")
    
    # Load the study raw files JSON
    study_raw_files = load_study_raw_files_json(json_file)
    if not study_raw_files:
        print("❌ Could not load study raw files JSON")
        return
    
    print(f"📋 Loaded raw file mappings for {len(study_raw_files)} studies from {json_file}")
    
    # Test FTP connection
    print("\n🔍 Testing FTP connection to MetaboLights...")
    working_config = test_metabolights_ftp_connection()
    
    if not working_config:
        print("❌ Could not establish FTP connection")
        return
    
    # Check if container exists
    if not os.path.exists(container_path):
        print(f"❌ Container not found at: {container_path}")
        print("Please ensure the apptainer container is available")
        return
    
    msms_datasets = []
    no_rawfile_datasets = []
    
    print(f"\n🚀 Using FTP config: {working_config['host']}")
    print(f"🔧 Using container: {container_path}")
    print(f"📁 Processing max {max_files_per_study} files per study")
    
    # Create base temp directory
    os.makedirs(temp_base_dir, exist_ok=True)
    
    try:
        with FTP(working_config['host'], timeout=60) as ftp:
            ftp.login(user=working_config['user'], passwd=working_config['passwd'])
            
            for i, row in df.iterrows():
                study_id = row['study_id']
                print(f"\n[{i+1}/{len(df)}] Checking {study_id}...")
                
                # Find largest raw vendor files from JSON
                print(f"  🔍 Looking up raw files in JSON...")
                file_info_list = find_largest_raw_files_from_json(ftp, study_id, study_raw_files, max_files_per_study)
                
                if not file_info_list:
                    print(f"  ❌ No raw vendor files found")
                    # Add to no raw files list
                    row_dict = row.to_dict()
                    row_dict['largest_raw_file'] = 'None found'
                    row_dict['raw_file_size_mb'] = 0
                    row_dict['vendor_format'] = 'None'
                    row_dict['has_msms'] = False
                    row_dict['scan_mode'] = 'unknown'
                    row_dict['files_processed'] = 0
                    row_dict['files_with_msms'] = 0
                    row_dict['msms_files'] = ''
                    row_dict['min_mz'] = None
                    row_dict['max_mz'] = None
                    no_rawfile_datasets.append(row_dict)
                    continue
                
                print(f"  ✅ Found {len(file_info_list)} files to process")
                
                # Download, convert, and check for MS/MS
                dataset_temp_dir = os.path.join(temp_base_dir, study_id)
                analysis_result = download_convert_and_check_msms_metabolights_multiple(ftp, file_info_list, dataset_temp_dir, container_path)
                
                if analysis_result['has_msms']:
                    print(f"  ✅ MS/MS data confirmed in {analysis_result['files_with_msms']}/{analysis_result['files_processed']} files!")
                    print(f"  📊 Acquisition mode: {analysis_result['acquisition_mode']['mode']} ({analysis_result['acquisition_mode']['confidence']} confidence)")
                    
                    # Add the row to our MS/MS results
                    row_dict = row.to_dict()
                    row_dict['largest_raw_file'] = file_info_list[0]['filename']  # First (largest) file
                    row_dict['raw_file_size_mb'] = file_info_list[0]['size'] / 1024 / 1024
                    row_dict['vendor_format'] = file_info_list[0]['vendor']
                    row_dict['has_msms'] = True
                    
                    # Add acquisition mode information
                    row_dict['acquisition_mode'] = analysis_result['acquisition_mode']['mode']
                    row_dict['acquisition_confidence'] = analysis_result['acquisition_mode']['confidence']
                    row_dict['ms2_spectra_count'] = analysis_result['acquisition_mode']['ms2_count']
                    row_dict['unique_precursors'] = analysis_result['acquisition_mode']['unique_precursors']
                    row_dict['acquisition_details'] = analysis_result['acquisition_mode']['details']
                    
                    # Add scan mode information
                    row_dict['scan_mode'] = analysis_result['scan_mode']
                    
                    # Add file processing statistics
                    row_dict['files_processed'] = analysis_result['files_processed']
                    row_dict['files_with_msms'] = analysis_result['files_with_msms']
                    row_dict['msms_files'] = '; '.join(analysis_result['msms_files'])
                    
                    # Add isolation window analysis if available
                    iso_analysis = analysis_result['acquisition_mode']['isolation_window_analysis']
                    if iso_analysis:
                        # Use the correct keys that check_acquisition_mode actually returns
                        row_dict['isolation_window_max'] = iso_analysis.get('max_offset', None)
                        row_dict['isolation_window_min'] = iso_analysis.get('min_offset', None)
                        row_dict['isolation_window_avg'] = iso_analysis.get('avg_offset', None)
                        row_dict['isolation_window_count'] = iso_analysis.get('num_offsets_found', None)
                    else:
                        row_dict['isolation_window_max'] = None
                        row_dict['isolation_window_min'] = None
                        row_dict['isolation_window_avg'] = None
                        row_dict['isolation_window_count'] = None
                    
                    # Add m/z range information - NEW ADDITION
                    mz_range = analysis_result['mz_range']
                    row_dict['min_mz'] = mz_range.get('min_mz', None)
                    row_dict['max_mz'] = mz_range.get('max_mz', None)
                    
                    msms_datasets.append(row_dict)
                else:
                    print(f"  ❌ No MS/MS data found in any of the {analysis_result['files_processed']} processed files")
                    # Add to no MS files list (has raw but no MS/MS)
                    row_dict = row.to_dict()
                    row_dict['largest_raw_file'] = file_info_list[0]['filename']
                    row_dict['raw_file_size_mb'] = file_info_list[0]['size'] / 1024 / 1024
                    row_dict['vendor_format'] = file_info_list[0]['vendor']
                    row_dict['has_msms'] = False
                    row_dict['acquisition_mode'] = 'N/A'
                    row_dict['acquisition_confidence'] = 'N/A'
                    row_dict['ms2_spectra_count'] = 0
                    row_dict['unique_precursors'] = 0
                    row_dict['acquisition_details'] = 'No MS/MS data found'
                    row_dict['isolation_window_max'] = None
                    row_dict['isolation_window_min'] = None
                    row_dict['isolation_window_avg'] = None
                    row_dict['isolation_window_count'] = None
                    
                    # Add scan mode and file processing info
                    row_dict['scan_mode'] = analysis_result['scan_mode']
                    row_dict['files_processed'] = analysis_result['files_processed']
                    row_dict['files_with_msms'] = analysis_result['files_with_msms']
                    row_dict['msms_files'] = ''
                    
                    # Add m/z range information even for non-MS/MS files - NEW ADDITION
                    mz_range = analysis_result['mz_range']
                    row_dict['min_mz'] = mz_range.get('min_mz', None)
                    row_dict['max_mz'] = mz_range.get('max_mz', None)
                    
                    no_rawfile_datasets.append(row_dict)
                
                # Clean up dataset temp directory
                try:
                    if os.path.exists(dataset_temp_dir):
                        shutil.rmtree(dataset_temp_dir)
                except:
                    pass
                
                # Small delay to be nice to the server
                time.sleep(0.5)
    
    except Exception as e:
        print(f"❌ FTP error: {e}")
        return
    finally:
        # Clean up base temp directory
        try:
            if os.path.exists(temp_base_dir):
                shutil.rmtree(temp_base_dir)
        except:
            pass
    
    # Create output DataFrames and save
    if msms_datasets:
        msms_df = pd.DataFrame(msms_datasets)
        msms_df.to_csv(output_file, index=False)
        
        print(f"\n🎉 MS/MS Results Summary:")
        print(f"   📊 Total datasets checked: {len(df)}")
        print(f"   ✅ Datasets with MS/MS data: {len(msms_df)}")
        print(f"   💾 MS/MS results saved to: {output_file}")
        
        # Show sample of MS/MS results
        if len(msms_df) > 0:
            print(f"\n📋 Sample of MS/MS datasets:")
            display_cols = ['study_id', 'title', 'largest_raw_file', 'vendor_format', 'raw_file_size_mb', 'acquisition_mode', 'scan_mode', 'min_mz', 'max_mz', 'files_processed', 'files_with_msms']
            available_cols = [col for col in display_cols if col in msms_df.columns]
            print(msms_df[available_cols].head())
            
            # File processing statistics
            if 'files_processed' in msms_df.columns:
                total_files_processed = msms_df['files_processed'].sum()
                total_files_with_msms = msms_df['files_with_msms'].sum()
                print(f"\n📁 File Processing Statistics:")
                print(f"   • Total files processed: {total_files_processed}")
                print(f"   • Total files with MS/MS: {total_files_with_msms}")
                print(f"   • Average files per study: {msms_df['files_processed'].mean():.1f}")
            
            # m/z range statistics - NEW ADDITION
            if 'min_mz' in msms_df.columns and 'max_mz' in msms_df.columns:
                valid_mz_data = msms_df.dropna(subset=['min_mz', 'max_mz'])
                if len(valid_mz_data) > 0:
                    print(f"\n📊 m/z Range Statistics:")
                    print(f"   • Datasets with m/z data: {len(valid_mz_data)}")
                    print(f"   • Overall min m/z: {valid_mz_data['min_mz'].min():.2f}")
                    print(f"   • Overall max m/z: {valid_mz_data['max_mz'].max():.2f}")
                    print(f"   • Average min m/z: {valid_mz_data['min_mz'].mean():.2f}")
                    print(f"   • Average max m/z: {valid_mz_data['max_mz'].mean():.2f}")
            
            # Acquisition mode analysis
            if 'acquisition_mode' in msms_df.columns:
                print(f"\n📊 Acquisition Mode Distribution:")
                acq_mode_counts = msms_df['acquisition_mode'].value_counts()
                for mode, count in acq_mode_counts.items():
                    print(f"   • {mode}: {count} datasets")
                
                # Scan mode analysis
                if 'scan_mode' in msms_df.columns:
                    print(f"\n🔬 Scan Mode Distribution:")
                    scan_mode_counts = msms_df['scan_mode'].value_counts()
                    for mode, count in scan_mode_counts.items():
                        print(f"   • {mode}: {count} datasets")
                
                if 'acquisition_confidence' in msms_df.columns:
                    print(f"\n🎯 Confidence Distribution:")
                    conf_counts = msms_df['acquisition_confidence'].value_counts()
                    for conf, count in conf_counts.items():
                        print(f"   • {conf}: {count} datasets")
    else:
        print(f"\n❌ No datasets with MS/MS data found")
    
    # Save datasets without raw files or without MS/MS
    if no_rawfile_datasets:
        no_rawfile_df = pd.DataFrame(no_rawfile_datasets)
        no_rawfile_df.to_csv(no_msfile_output, index=False)
        
        print(f"\n📋 No Raw Files Summary:")
        print(f"   ❌ Datasets without raw files or MS/MS: {len(no_rawfile_df)}")
        print(f"   💾 No raw files results saved to: {no_msfile_output}")
        
        # Show breakdown
        no_files = len(no_rawfile_df[no_rawfile_df['largest_raw_file'] == 'None found'])
        has_files_no_msms = len(no_rawfile_df[no_rawfile_df['largest_raw_file'] != 'None found'])
        
        print(f"   📊 Breakdown:")
        print(f"      • No raw files found: {no_files}")
        print(f"      • Has raw files but no MS/MS: {has_files_no_msms}")

def analyze_metabolights_msms_conversion_results():
    """
    Analyze the results of MetaboLights MS/MS verification with conversion.
    """
    print("\n📊 === METABOLIGHTS MS/MS CONVERSION VERIFICATION ANALYSIS ===")
    print("=" * 60)
    
    # Try to load both result files
    msms_file = 'metabolights_datasets_msms_verified_converted.csv'
    no_raw_file = 'metabolights_datasets_no_rawfiles.csv'
    
    msms_df = None
    no_raw_df = None
    
    try:
        if Path(msms_file).exists():
            msms_df = pd.read_csv(msms_file)
            print(f"✅ Loaded {len(msms_df)} datasets with MS/MS data (from conversion)")
        else:
            print(f"❌ {msms_file} not found")
            
        if Path(no_raw_file).exists():
            no_raw_df = pd.read_csv(no_raw_file)
            print(f"✅ Loaded {len(no_raw_df)} datasets without raw files/MS/MS")
        else:
            print(f"❌ {no_raw_file} not found")
            
    except Exception as e:
        print(f"❌ Error loading result files: {e}")
        return
    
    if msms_df is not None:
        print(f"\n📈 MS/MS Datasets Analysis (Converted):")
        print(f"   • Total datasets with MS/MS: {len(msms_df)}")
        
        # Vendor format distribution
        if 'vendor_format' in msms_df.columns:
            print(f"   • Vendor format distribution:")
            vendor_counts = msms_df['vendor_format'].value_counts()
            for vendor, count in vendor_counts.items():
                print(f"     - {vendor}: {count} datasets")
        
        # Species distribution for MS/MS datasets
        if 'species' in msms_df.columns:
            print(f"   • Species distribution (Top 5):")
            species_counts = msms_df['species'].value_counts().head(5)
            for species, count in species_counts.items():
                print(f"     - {species}: {count} datasets")
        
        # File size analysis
        if 'raw_file_size_mb' in msms_df.columns:
            total_size_gb = msms_df['raw_file_size_mb'].sum() / 1024
            avg_size_mb = msms_df['raw_file_size_mb'].mean()
            print(f"   • Total raw file size: {total_size_gb:.1f} GB")
            print(f"   • Average raw file size: {avg_size_mb:.1f} MB")
    
    if no_raw_df is not None:
        print(f"\n📉 Non-MS/MS Datasets Analysis:")
        print(f"   • Total datasets without raw files/MS/MS: {len(no_raw_df)}")
        
        # Breakdown by reason
        if 'largest_raw_file' in no_raw_df.columns:
            no_files = len(no_raw_df[no_raw_df['largest_raw_file'] == 'None found'])
            has_files = len(no_raw_df[no_raw_df['largest_raw_file'] != 'None found'])
            print(f"   • No raw files: {no_files}")
            print(f"   • Has raw files but no MS/MS: {has_files}")

# Execute the verification with conversion
print("🚀 === METABOLIGHTS MS/MS VERIFICATION WITH RAW FILE CONVERSION ===")
print("=" * 60)

# Update these paths according to your setup
CONTAINER_PATH = "/home/maxvandenboom/Documents/MS-data-mining/pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif"
TEMP_BASE_DIR = "/tmp/metabolights_conversion"
JSON_FILE_PATH = "/home/maxvandenboom/Documents/MS-data-mining/dataset_metadata/metabolights/study_raw_files.json"

# Verify MS/MS data in MetaboLights datasets with conversion
verify_msms_metabolights_datasets_with_conversion(
    '/home/maxvandenboom/Documents/MS-data-mining/dataset_metadata/metabolights/metabolights_metadata_verified.csv',  # Input file with datasets
    JSON_FILE_PATH,  # JSON file with raw file mappings
    container_path=CONTAINER_PATH,
    temp_base_dir=TEMP_BASE_DIR,
    max_files_per_study=5  # Process up to 5 files per study
)

print(f"\n🎯 === SUMMARY ===")
print("✅ MetaboLights MS/MS verification with conversion complete!")
print("📄 Check the output CSV files for detailed results")
print("🔧 Raw vendor files were converted to mzML using msconvert")
print(f"📁 Processed up to 5 files per study for comprehensive MS/MS detection")