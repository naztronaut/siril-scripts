import os
import numpy as np
from astropy.io import fits

def is_black_frame(data, threshold=30, crop_fraction=0.2):
    if data.ndim > 2:
        data = data[0]  
    ny, nx = data.shape
    crop_x = int(nx * crop_fraction)
    crop_y = int(ny * crop_fraction)
    start_x = (nx - crop_x) // 2
    start_y = (ny - crop_y) // 2

    crop = data[start_y:start_y + crop_y, start_x:start_x + crop_x]
    median_val = np.median(crop)

    return median_val < threshold, median_val

def scan_r_bkg_lights(folder=".", threshold=30, crop_fraction=0.2):
    black_frames = []
    black_indices = []
    all_frames_info = []

    for idx, filename in enumerate(sorted(os.listdir(folder))): 
        if filename.startswith("r_bkg_lights") and filename.lower().endswith((".fit", ".fits")):
            filepath = os.path.join(folder, filename)
            try:
                with fits.open(filepath) as hdul:
                        data = hdul[0].data
                        if data is not None and data.ndim >= 2:
                            dynamic_threshold = threshold
                            data_max = np.max(data)
                            if np.issubdtype(data.dtype, np.floating) or data_max <= 10.0:
                                dynamic_threshold = 0.004  

                            is_black, median_val = is_black_frame(data, dynamic_threshold, crop_fraction)
                            all_frames_info.append((filename, median_val))

                            # Log for debugging
                            print(f"{filename} | shape: {data.shape} | dtype: {data.dtype} | min: {np.min(data)} | max: {data_max} | median: {median_val} | threshold used: {dynamic_threshold}")

                            if is_black:
                                black_frames.append(filename)
                                black_indices.append(len(all_frames_info))
                        else:
                            print(f"{filename}: Unexpected data shape {data.shape if data is not None else 'None'}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(black_frames)
    print(all_frames_info)
    print(black_indices)
    
scan_r_bkg_lights()
    
