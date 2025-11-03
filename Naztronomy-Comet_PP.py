"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Smart Telescope Preprocessing script
Version: 2.0.0
=====================================

The author of this script is Nazmus Nasir (Naztronomy) and can be reached at:
https://www.Naztronomy.com or https://www.YouTube.com/Naztronomy
Join discord for support and discussion: https://discord.gg/yXKqrawpjr
Support me on Patreon: https://www.patreon.com/c/naztronomy
Support me on Buy me a Coffee: https://www.buymeacoffee.com/naztronomy

The following directory is required inside the working directory:
    lights/

The following subdirectories are optional:
    darks/
    flats/
    biases/

"""

"""
CHANGELOG:

1.0.0 - initial release
      - Basic conversion and preprocessing for comet imaging
      - Supports darks, flats, biases
      - Works with all telescopes
"""

from fileinput import filename
import os
from pathlib import Path
import sys
import math
import shutil
import sirilpy as s
from datetime import datetime
import json
import tifffile


s.ensure_installed("PyQt6", "numpy", "astropy")
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QSpinBox,
    QLineEdit,
)
from PyQt6.QtCore import pyqtSlot as Slot, Qt
from PyQt6.QtGui import QFont, QShortcut, QKeySequence
from sirilpy import LogColor, NoImageError
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import subprocess

# from tkinter import filedialog

APP_NAME = "Naztronomy - Comet Preprocessor"
VERSION = "1.0.0"
BUILD = "20251102"
AUTHOR = "Nazmus Nasir"
WEBSITE = "Naztronomy.com"
YOUTUBE = "YouTube.com/Naztronomy"


UI_DEFAULTS = {
    "max_files_per_batch": 2000,
}


class PreprocessingInterface(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")

        self.siril = s.SirilInterface()

        # Flags for mosaic mode and drizzle status
        # if drizzle is off, images will be debayered on convert
        self.initialization_successful = False


        try:
            self.siril.connect()
            self.siril.log("Connected to Siril", LogColor.GREEN)
        except s.SirilConnectionError:
            self.siril.log("Failed to connect to Siril", LogColor.RED)
            self.close_dialog()
            return
        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

        self.fits_extension = self.siril.get_siril_config("core", "extension")
        self.starnet_location = self.siril.get_siril_config("core", "starnet_exe")
        self.starnet_available = True if self.starnet_location != "(not set)" else False

        self.gaia_catalogue_available = False
        try:
            catalog_status = self.siril.get_siril_config("core", "catalogue_gaia_astro")
            if (
                catalog_status
                and catalog_status != "(not set)"
                and os.path.isfile(catalog_status)
            ):
                self.gaia_catalogue_available = True

        except s.CommandError:
            pass
        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label_text = ""

        changed_cwd = False  # a way not to run the prompting loop
        initial_cwd = os.path.join(self.current_working_directory, "lights")
        if os.path.isdir(initial_cwd):
            self.siril.log(
                f"Current working directory is valid: {self.current_working_directory}",
                LogColor.GREEN,
            )
            self.siril.cmd("cd", f'"{self.current_working_directory}"')
            self.cwd_label_text = (
                f"Current working directory: {self.current_working_directory}"
            )
            changed_cwd = True
        elif os.path.basename(self.current_working_directory.lower()) == "lights":
            msg = "You're currently in the 'lights' directory, do you want to select the parent directory?"
            answer = QMessageBox.question(self, "Already in Lights Dir", msg)
            if answer == QMessageBox.Yes:
                self.siril.cmd("cd", "../")
                os.chdir(os.path.dirname(self.current_working_directory))
                self.current_working_directory = os.path.dirname(
                    self.current_working_directory
                )
                self.cwd_label_text = (
                    f"Current working directory: {self.current_working_directory}"
                )
                self.siril.log(
                    f"Updated current working directory to: {self.current_working_directory}",
                    LogColor.GREEN,
                )
                changed_cwd = True
            else:
                self.siril.log(
                    f"Current working directory is invalid: {self.current_working_directory}, reprompting...",
                    LogColor.SALMON,
                )
                changed_cwd = False

        if not changed_cwd:
            while True:
                prompt_title = (
                    "Select the parent directory containing the 'lights' directory"
                )

                selected_dir = QFileDialog.getExistingDirectory(
                    self,
                    prompt_title,
                    self.current_working_directory,
                    QFileDialog.Option.ShowDirsOnly,
                )

                if not selected_dir:
                    self.siril.log(
                        "Canceled selecting directory. Restart the script to try again.",
                        LogColor.SALMON,
                    )
                    self.siril.disconnect()
                    self.close()
                    return  # Stop initialization completely

                lights_directory = os.path.join(selected_dir, "lights")
                if os.path.isdir(lights_directory):
                    self.siril.cmd("cd", f'"{selected_dir}"')
                    os.chdir(selected_dir)
                    self.current_working_directory = selected_dir
                    self.cwd_label_text = f"Current working directory: {selected_dir}"
                    self.siril.log(
                        f"Updated current working directory to: {selected_dir}",
                        LogColor.GREEN,
                    )
                    break

                elif os.path.basename(selected_dir.lower()) == "lights":
                    msg = "The selected directory is the 'lights' directory, do you want to select the parent directory?"
                    answer = QMessageBox.question(
                        self,
                        "Already in Lights Dir",
                        msg,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    )
                    if answer == QMessageBox.StandardButton.Yes:
                        parent_dir = os.path.dirname(selected_dir)
                        self.siril.cmd("cd", f'"{parent_dir}"')
                        os.chdir(parent_dir)
                        self.current_working_directory = parent_dir
                        self.cwd_label_text = f"Current working directory: {parent_dir}"
                        self.siril.log(
                            f"Updated current working directory to: {parent_dir}",
                            LogColor.GREEN,
                        )
                    break
                else:
                    msg = f"The selected directory must contain a subdirectory named 'lights'.\nYou selected: {selected_dir}. Please try again."
                    self.siril.log(msg, LogColor.SALMON)
                    QMessageBox.critical(
                        self, "Invalid Directory", msg, QMessageBox.StandardButton.Ok
                    )
                    continue
        self.create_widgets()

        # self.setup_shortcuts()
        self.initialization_successful = True

    # Dirname: lights, darks, biases, flats
    def convert_files(self, dir_name):
        directory = os.path.join(self.current_working_directory, dir_name)
        if os.path.isdir(directory):
            self.siril.cmd("cd", dir_name)
            file_count = len(
                [
                    name
                    for name in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, name))
                ]
            )
            if file_count == 1:
                self.siril.log(
                    f"Only one file found in {dir_name} directory. Treating it like a master {dir_name} frame.",
                    LogColor.BLUE,
                )
                src = os.path.join(directory, os.listdir(directory)[0])

                dst = os.path.join(
                    self.current_working_directory,
                    "process",
                    f"{dir_name}_stacked{self.fits_extension}",
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                self.siril.log(
                    f"Copied master {dir_name} to process as {dir_name}_stacked.",
                    LogColor.BLUE,
                )
                self.siril.cmd("cd", "..")
                # return false because there's no conversion
                return False
            try:
                args = ["convert", dir_name, "-out=../process"]
                self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                self.siril.cmd(*args)
            except (s.DataError, s.CommandError, s.SirilError) as e:
                self.siril.log(f"File conversion failed: {e}", LogColor.RED)
                self.close_dialog()

            self.siril.cmd("cd", "../process")
            self.siril.log(
                f"Converted {file_count} {dir_name} files for processing!",
                LogColor.GREEN,
            )
            return True
        else:
            self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            raise NoImageError(
                (
                    f'No directory named "{dir_name}" at this location. Make sure the working directory is correct.'
                )
            )

    def seq_bg_extract(self, seq_name):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        try:
            self.siril.cmd("seqsubsky", seq_name, "1", "-samples=10")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Seq BG Extraction failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Background extracted from Sequence", LogColor.GREEN)

    def seq_apply_reg(
        self, seq_name, filter_roundness, filter_fwhm
    ):
        """Apply Existing Registration to the sequence."""
        cmd_args = [
            "seqapplyreg",
            seq_name,
            f"-filter-round={filter_roundness}k",
            f"-filter-wfwhm={filter_fwhm}k",
            "-kernel=square",
            "-framing=current",
        ]

        self.siril.log("Command arguments: " + " ".join(cmd_args), LogColor.BLUE)

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Data error occurred: {e}", LogColor.RED)

        self.siril.log("Registered Sequence", LogColor.GREEN)

    def is_black_frame(self, data, threshold=10, crop_fraction=0.4):
        if data.ndim > 2:
            data = data[0]

        ny, nx = data.shape
        crop_x = int(nx * crop_fraction)
        crop_y = int(ny * crop_fraction)
        start_x = (nx - crop_x) // 2
        start_y = (ny - crop_y) // 2

        crop = data[start_y : start_y + crop_y, start_x : start_x + crop_x]
        nonzero = crop[crop != 0]

        if nonzero.size == 0:
            median_val = 0.0
        else:
            median_val = np.median(nonzero)

        return median_val < threshold, median_val

    def scan_black_frames(
        self, folder="process", threshold=30, crop_fraction=0.4, seq_name=None
    ):
        black_frames = []
        black_indices = []
        all_frames_info = []
        self.siril.log("Starting scan for black frames...", LogColor.BLUE)
        self.siril.log(
            "Note: This process is running in the background and may take a while depending on your system and drizzle factor.",
            LogColor.BLUE,
        )

        for idx, filename in enumerate(sorted(os.listdir(folder))):
            if filename.startswith(seq_name) and filename.lower().endswith(
                self.fits_extension
            ):
                filepath = os.path.join(folder, filename)
                try:
                    with fits.open(filepath) as hdul:
                        data = hdul[0].data
                        if data is not None and data.ndim >= 2:
                            dynamic_threshold = threshold
                            data_max = np.max(data)
                            if (
                                np.issubdtype(data.dtype, np.floating)
                                or data_max <= 10.0
                            ):
                                dynamic_threshold = 0.0001

                            is_black, median_val = self.is_black_frame(
                                data, dynamic_threshold, crop_fraction
                            )
                            all_frames_info.append((filename, median_val))

                            # Log for debugging
                            # print(
                            #     f"{filename} | shape: {data.shape} | dtype: {data.dtype} | min: {np.min(data)} | max: {data_max} | median: {median_val} | threshold used: {dynamic_threshold}"
                            # )

                            if is_black:
                                black_frames.append(filename)
                                black_indices.append(len(all_frames_info))
                        else:
                            self.siril.log(
                                f"{filename}: Unexpected data shape {data.shape if data is not None else 'None'}",
                                LogColor.SALMON,
                            )
                except Exception as e:
                    self.siril.log(f"Error reading {filename}: {e}", LogColor.RED)

        self.siril.log(f"Following files are black: {black_frames}", LogColor.SALMON)
        self.siril.log(
            f"Black indices skipped in stacking: {black_indices}", LogColor.SALMON
        )
        for index in black_indices:
            self.siril.cmd("unselect", seq_name, index, index)

    def calibration_stack(self, seq_name):
        # not in /process dir here
        file_name_end = "_stacked"
        if seq_name == "flats":
            if os.path.exists(
                os.path.join(
                    self.current_working_directory,
                    f"process/biases{file_name_end}{self.fits_extension}",
                )
            ):
                # Saves as pp_flats
                self.siril.cmd("calibrate", "flats", f"-bias=biases{file_name_end}")
                self.siril.cmd(
                    "stack", "pp_flats rej 3 3", "-norm=mul", f"-out={seq_name}_stacked"
                )
                # self.siril.cmd("cd", "..")

            else:
                self.siril.cmd(
                    "stack",
                    f"{seq_name} rej 3 3",
                    "-norm=mul",
                    f"-out={seq_name}_stacked",
                )

        else:
            # Don't run code below for flats
            # biases and darks
            cmd_args = [
                "stack",
                f"{seq_name} rej 3 3 -nonorm",
                f"-out={seq_name}{file_name_end}",
            ]
            self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

            try:
                self.siril.cmd(*cmd_args)
            except (s.DataError, s.CommandError, s.SirilError) as e:
                self.siril.log(f"Command execution failed: {e}", LogColor.RED)
                self.close_dialog()

        self.siril.log(f"Completed stacking {seq_name}!", LogColor.GREEN)

        # Copy the stacked calibration files to ../masters directory
        masters_dir = os.path.join(self.current_working_directory, "masters")
        os.makedirs(masters_dir, exist_ok=True)
        src = os.path.join(
            self.current_working_directory,
            f"process/{seq_name}{file_name_end}{self.fits_extension}",
        )
        # Read FITS headers if file exists
        filename_parts = [seq_name, "stacked"]

        if os.path.exists(src):
            try:
                with fits.open(src) as hdul:
                    headers = hdul[0].header
                    # Add temperature if exists
                    if "CCD-TEMP" in headers:
                        temp = f"{headers['CCD-TEMP']:.1f}C"
                        filename_parts.insert(1, temp)

                    # Add date if exists
                    if "DATE-OBS" in headers:
                        try:
                            dt = datetime.fromisoformat(headers["DATE-OBS"])
                            date = dt.date().isoformat()  # "2025-09-29"
                        except ValueError:
                            # fallback if DATE-OBS is not strict ISO format
                            date = headers["DATE-OBS"].split("T")[0]

                        filename_parts.insert(1, date)

                    # Add exposure time if exists
                    if "EXPTIME" in headers:
                        exp = f"{headers['EXPTIME']:.0f}s"
                        filename_parts.insert(1, exp)
            except Exception as e:
                self.siril.log(f"Error reading FITS headers: {e}", LogColor.SALMON)

        dst = os.path.join(
            masters_dir, f"{'_'.join(filename_parts)}{self.fits_extension}"
        )

        if os.path.exists(src):
            # Remove destination file if it exists to ensure override
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)
            self.siril.log(
                f"Copied {seq_name} to masters directory as {'_'.join(filename_parts)}{self.fits_extension}",
                LogColor.BLUE,
            )
        self.siril.cmd("cd", "..")

    def calibrate_lights(self, seq_name, use_darks=False, use_flats=False):
        cmd_args = [
            "calibrate",
            f"{seq_name}",
            "-dark=darks_stacked" if use_darks else "",
            "-flat=flats_stacked" if use_flats else "",
            "-cfa -equalize_cfa",
        ]

        # Calibrate with -debayer if drizle is not set

        cmd_args.append("-debayer")

        self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Command execution failed: {e}", LogColor.RED)
            self.close_dialog()

    def register_sequence(self, seq_name):
        """Registers the sequence using the 'register' command."""
        cmd_args = ["register", seq_name, "-2pass"]
        self.siril.log(
            "Global Star Registration Done: " + " ".join(cmd_args), LogColor.BLUE
        )

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Data error occurred: {e}", LogColor.RED)

        self.siril.log("Registered Sequence", LogColor.GREEN)
    def seq_stack(
        self, seq_name, rejection=False, output_name=None
    ):
        """Stack it all, and feather if it's provided"""
        out = "result" if output_name is None else output_name
        rejection_type = {
            "Sigma Clipping": "s",
            "Winsorized Sigma Clipping": "w", 
            "Linear Fit Clipping": "l",
            "MAD Clipping": "a"
        }.get(self.stacking_algorithm_combo.currentText(), "w")

        sigma_low = self.sigma_low_spinbox.value()
        sigma_high = self.sigma_high_spinbox.value()
        
        cmd_args = [
            "stack",
            f"{seq_name}",
            f" rej {rejection_type} {sigma_low} {sigma_high}" if rejection else " rej none",
            "-norm=addscale",
            "-output_norm",
            "-rgb_equal", 
            # "-maximize",
            "-filter-included",
            f"-out={out}",
        ]


        self.siril.log(
            f"Running seq_stack with arguments:\n"
            f"seq_name={seq_name}\n",
            LogColor.BLUE,
        )

        self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Stacking failed: {e}", LogColor.RED)
            self.close_dialog()

        self.siril.log(f"Completed stacking {seq_name}!", LogColor.GREEN)

    def save_image(self, suffix):
        """Saves the image as a FITS file."""

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M")

        # Default filename
        file_name = f"result_{current_datetime}{suffix}"

        # Get header info from loaded image for filename
        current_fits_headers = self.siril.get_image_fits_header(return_as="dict")

        object_name = (
            current_fits_headers.get("OBJECT", "Unknown").strip().replace(" ", "_")
        )
        exptime = int(current_fits_headers.get("EXPTIME", 0))
        stack_count = int(current_fits_headers.get("STACKCNT", 0))
        date_obs = current_fits_headers.get("DATE-OBS", current_datetime)

        try:
            dt = datetime.fromisoformat(date_obs)
            date_obs_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_obs_str = datetime.now().strftime("%Y%m%d")

        file_name = f"{object_name}_{stack_count:03d}x{exptime}sec_{date_obs_str}"

        file_name += f"__{current_datetime}{suffix}"

        try:
            self.siril.cmd(
                "save",
                f"{file_name}",
            )
            return file_name
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Save command execution failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log(f"Saved file: {file_name}", LogColor.GREEN)

    def load_registered_image(self):
        """Loads the registered image. Currently unused"""
        try:
            self.siril.cmd("load", "result")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Load command execution failed: {e}", LogColor.RED)
        self.save_image("_og")

    # Custom sequence autostretch function to load each frame in a particular sequence and apply an autostretch and save it
    # def seq_autostretch(self, seq_name):
    #     """Apply autostretch to all images in a sequence."""
    #     try:
    #         # Get current sequence 
    #         self.siril.cmd("load_seq", seq_name)
    #         seq = self.siril.get_seq()
    #         if not seq:
    #             raise s.NoSequenceError("No sequence loaded")

    #         # Loop through each frame
    #         print(f"Stats: {seq.stats}")
    #         for i in range(seq.number):
    #             # Get full filename from sequence name and index
    #             frame_filename = f"{seq_name}{i+1:05d}{self.fits_extension}"
    #             print(f"Processing frame {i+1}/{seq.number}: {frame_filename}")
                
    #             # Load frame
    #             self.siril.cmd("load", frame_filename)
    #             self.siril.cmd("autostretch")
    #             self.siril.cmd("save", f"tmlps_{i+1:05d}")

    #         self.siril.log(f"Applied autostretch to sequence {seq_name} with output of tmlps_", LogColor.GREEN)
    #         self.siril.cmd("close")
    #         self.siril.create_new_seq("tmlps_")
    #         args = ["load_seq", "tmlps_"]
    #         print( " ".join(str(arg) for arg in args))  
    #         self.siril.cmd(*args)

    def seq_autostretch(self, seq_name):
        """Apply autostretch to all images in a sequence."""
        try:
            # Get current sequence 
            self.siril.cmd("load_seq", seq_name)
            seq = self.siril.get_seq()
            if not seq:
                raise s.NoSequenceError("No sequence loaded")

            print(f"Processing {seq.number} frames...")
            
            # Pre-construct filenames to reduce string operations in loop
            filenames = [f"{seq_name}{i+1:05d}{self.fits_extension}" for i in range(seq.number)]
            
            # Process each frame
            for i, filename in enumerate(filenames):
                print(f"Processing frame {i+1}/{seq.number}: {Path(filename)}")
                print(os.getcwd())
                if not os.path.isfile(os.path.join(os.getcwd(), "process", filename)):
                    self.siril.log(f"File not found: {filename}", LogColor.RED)
                    continue  # Skip to next frame
                try: 
                    self.siril.cmd("load", filename)
                    self.siril.cmd("autostretch")
                    self.siril.cmd("savetif", f"tmlps_{i+1:05d}")
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(f"Failed to process frame {filename}: {e}", LogColor.RED)
                    continue  # Skip to next frame

            self.siril.log(f"Applied autostretch to sequence {seq_name}", LogColor.GREEN)
            self.siril.cmd("close")
                
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Sequence autostretch failed: {e}", LogColor.RED)
            self.close_dialog()            


    def seq_starnet(self, seq_name):
        """Apply starnet to all images in a sequence. Custom method because current seqstarnet does not work."""
        try:
            # Get current sequence 
            self.siril.cmd("load_seq", seq_name)
            seq = self.siril.get_seq()
            if not seq:
                raise s.NoSequenceError("No sequence loaded")

            print(f"Processing {seq.number} frames...")
            
            # Pre-construct filenames to reduce string operations in loop
            filenames = [f"{seq_name}{i+1:05d}{self.fits_extension}" for i in range(seq.number)]
            
            # Process each frame
            for i, filename in enumerate(filenames):
                print(f"Starnet on frame {i+1}/{seq.number}: {Path(filename)}")
                print(os.getcwd())
                if not os.path.isfile(os.path.join(os.getcwd(), "process", filename)):
                    self.siril.log(f"File not found: {filename}", LogColor.RED)
                    continue  # Skip to next frame
                try: 
                    self.siril.cmd("load", filename)
                    self.siril.cmd("starnet", "-stretch", "-nostarmask")
                    # self.siril.cmd("save", f"starless_comet_{i+1:05d}"){
                    self.siril.log(f"Processed frame {filename} with starnet.", LogColor.GREEN)
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(f"Failed to process frame {filename}: {e}", LogColor.RED)
                    continue  # Skip to next frame

            self.siril.log(f"Applied custom starnet to sequence {seq_name}", LogColor.GREEN)
            self.siril.cmd("close")
            
            # Create new sequence from saved files
            # self.siril.create_new_seq("starless_comet_")
            # self.siril.cmd("load_seq", "tmlps_")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Sequence autostretch failed: {e}", LogColor.RED)
            self.close_dialog()

    def remove_stars(self, sequence=True, seq_name=None):
        """Removes stars from the loaded image or sequence."""
        self.siril.log(f"Starnet availability: {self.starnet_available}", LogColor.BLUE)

        if(not self.starnet_available):
            self.siril.log("Starnet is not available in Siril configuration. Skipping remove stars!", LogColor.RED)
            return False

        try:
            if sequence:
                self.siril.cmd("seqstarnet", seq_name, "-stretch")
            else:
                self.siril.cmd("starnet", "-stretch")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Remove Stars command execution failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Removed stars from image/sequence", LogColor.GREEN)
        return True

    def image_plate_solve(self):
        """Plate solve the loaded image with the '-force' argument."""
        try:
            self.siril.cmd("platesolve", "-force")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Plate Solve command execution failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Platesolved image", LogColor.GREEN)

    def load_image(self, image_name):
        """Loads the result."""
        try:
            self.siril.cmd("load", image_name)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Load image failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log(f"Loaded image: {image_name}", LogColor.GREEN)


    def clean_up(self, prefix=None):
        """Cleans up all files in the process directory."""
        if not self.current_working_directory.endswith("process"):
            process_dir = os.path.join(self.current_working_directory, "process")
        else:
            process_dir = self.current_working_directory
        for f in os.listdir(process_dir):
            # Skip the stacked file
            name, ext = os.path.splitext(f.lower())
            if name in (f"{prefix}_stacked", "result") and ext in (self.fits_extension):
                continue

            # Check if file starts with prefix_ or pp_flats_
            if (
                f.startswith(prefix)
                or f.startswith(f"{prefix}_")
                or f.startswith("pp_flats_")
            ):
                file_path = os.path.join(process_dir, f)
                if os.path.isfile(file_path):
                    # print(f"Removing: {file_path}")
                    os.remove(file_path)
        self.siril.log(f"Cleaned up {prefix}", LogColor.BLUE)


    def show_help(self):
        help_text = (
            f"Author: {AUTHOR} ({WEBSITE})\n"
            f"Youtube: {YOUTUBE}\n"
            "Discord: https://discord.gg/yXKqrawpjr\n"
            "Patreon: https://www.patreon.com/c/naztronomy\n"
            "Buy me a Coffee: https://www.buymeacoffee.com/naztronomy\n\n"
            "Info:\n"
            '1. Must have a "lights" subdirectory inside of the working directory.\n'
            "2. For Calibration frames, you can have one or more of the following types: darks, flats, biases.\n"
            "3. If only one calibration frame is present, it will be treated as a master frame.\n"
            "4. Recommended to process only a single comet session.\n"
            "5. Does NOT do plate solving and will not do mosaics.\n"
            "6. If using Remove Stars, Starnet++ must be installed.\n"
            "7. If doing timelapse, ffmpeg must be installed.\n"
            "8. When asking for help, please have the logs handy."
        )

        # Show help in Qt message box
        QMessageBox.information(self, "Help", help_text)
        self.siril.log(help_text, LogColor.BLUE)

    def create_widgets(self):
        """Creates the UI widgets."""
        # Create main widget and layout
        main_widget = QWidget()
        self.setMinimumSize(700, 600)
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 10, 15, 15)
        main_layout.setSpacing(8)

        # Title and version
        title_label = QLabel(f"{APP_NAME}")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        # Current working directory label
        self.cwd_label = QLabel(self.cwd_label_text)
        main_layout.addWidget(self.cwd_label)
        self.starnet_location_label = QLabel("Starnet++ Location: " + (self.starnet_location if self.starnet_available else "Not Set, please install."))
        main_layout.addWidget(self.starnet_location_label)

        # Telescope section
        telescope_section = QGroupBox("Conversion Options")
        telescope_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        main_layout.addWidget(telescope_section)
        telescope_layout = QGridLayout(telescope_section)
        telescope_layout.setSpacing(3)
        telescope_layout.setContentsMargins(10, 15, 10, 10)

        # Optional Calibration Frames
        calib_frames_label = QLabel("Calibration Frames:")
        calib_frames_label.setFont(title_font)
        calib_frames_tooltip = "Select which calibration frames to use in preprocessing. Calibration frames help reduce noise and correct optical imperfections."
        calib_frames_label.setToolTip(calib_frames_tooltip)
        telescope_layout.addWidget(calib_frames_label, 1, 0)

        self.darks_checkbox = QCheckBox("Darks")
        self.darks_checkbox.setToolTip(
            "Dark frames help remove thermal noise and hot pixels. Use if you have matching exposure dark frames."
        )
        telescope_layout.addWidget(self.darks_checkbox, 1, 1)

        self.flats_checkbox = QCheckBox("Flats")
        self.flats_checkbox.setToolTip(
            "Flat frames correct for vignetting and dust spots."
        )
        telescope_layout.addWidget(self.flats_checkbox, 1, 2)

        self.biases_checkbox = QCheckBox("Biases")
        self.biases_checkbox.setToolTip(
            "Bias frames correct for read noise. Only used with flats."
        )
        telescope_layout.addWidget(self.biases_checkbox, 1, 3)

        # Add some vertical spacing between calibration and cleanup
        telescope_layout.setRowMinimumHeight(1, 35)

        cleanup_files_label = QLabel("Clean Up Files:")
        cleanup_files_label.setFont(title_font)
        cleanup_tooltip = "Enable this option to delete all intermediary files after they are done processing. This saves space on your hard drive.\nNote: If your session is batched, this option is automatically enabled even if it's unchecked!"
        cleanup_files_label.setToolTip(cleanup_tooltip)
        telescope_layout.addWidget(cleanup_files_label, 2, 0)

        self.cleanup_files_checkbox = QCheckBox("")
        self.cleanup_files_checkbox.setToolTip(cleanup_tooltip)
        telescope_layout.addWidget(self.cleanup_files_checkbox, 2, 1)

        # Optional Preprocessing Steps
        preprocessing_section = QGroupBox("Optional Preprocessing Steps")
        preprocessing_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        main_layout.addWidget(preprocessing_section)
        preprocessing_layout = QGridLayout(preprocessing_section)
        preprocessing_layout.setSpacing(5)
        # preprocessing_layout.setContentsMargins(10, 15, 10, 10)
        preprocessing_layout.setHorizontalSpacing(15)  # space between label ↔ widget
        preprocessing_layout.setVerticalSpacing(10)  # space between rows
        preprocessing_layout.setContentsMargins(12, 18, 12, 12)  # outer padding

        bg_extract_label = QLabel("Background Extraction:")
        bg_extract_label.setFont(title_font)
        bg_extract_tooltip = "Removes background gradients from your images before stacking. Uses Polynomial value 1 and 10 samples."
        bg_extract_label.setToolTip(bg_extract_tooltip)
        preprocessing_layout.addWidget(bg_extract_label, 1, 0)

        self.bg_extract_checkbox = QCheckBox("")
        self.bg_extract_checkbox.setToolTip(bg_extract_tooltip)
        preprocessing_layout.addWidget(self.bg_extract_checkbox, 1, 1)

        registration_label = QLabel("Registration:")
        registration_label.setFont(title_font)
        registration_label.setToolTip("Options for aligning images before stacking.")
        preprocessing_layout.addWidget(registration_label, 2, 0)

        # Add spinboxes for roundness and FWHM filters

        filters_checkbox_tooltip = (
            "Options for filtering images based on various criteria."
        )
        self.filters_checkbox = QCheckBox("Filters")
        self.filters_checkbox.setToolTip(filters_checkbox_tooltip)
        preprocessing_layout.addWidget(self.filters_checkbox, 4, 1)

        roundness_label_tooltip = "Filters images by star roundness, calculated using the second moments of detected stars. \nA lower roundness value applies a stricter filter, keeping only frames with well-defined, circular stars. Higher roundness values allow more variation in star shapes."
        roundness_label = QLabel("Roundness:")
        roundness_label.setToolTip(roundness_label_tooltip)
        preprocessing_layout.addWidget(roundness_label, 4, 2)

        self.roundness_spinbox = QDoubleSpinBox()
        self.roundness_spinbox.setRange(0.1, 5.0)
        self.roundness_spinbox.setSingleStep(0.1)
        self.roundness_spinbox.setDecimals(1)
        self.roundness_spinbox.setValue(3.0)
        self.roundness_spinbox.setMinimumWidth(80)
        self.roundness_spinbox.setSuffix(" σ")
        self.roundness_spinbox.setEnabled(False)
        self.roundness_spinbox.setToolTip(roundness_label_tooltip)
        preprocessing_layout.addWidget(self.roundness_spinbox, 4, 3)

        self.filters_checkbox.toggled.connect(self.roundness_spinbox.setEnabled)

        fwhm_label_tooltip = "Filters images by weighted Full Width at Half Maximum (FWHM), calculated using star sharpness. \nA lower sigma value applies a stricter filter, keeping only frames close to the median FWHM. Higher sigma allows more variation."
        fwhm_label = QLabel("Weighted FWHM:")
        fwhm_label.setToolTip(fwhm_label_tooltip)
        preprocessing_layout.addWidget(fwhm_label, 5, 2)

        self.fwhm_spinbox = QDoubleSpinBox()
        self.fwhm_spinbox.setRange(0.1, 5.0)
        self.fwhm_spinbox.setSingleStep(0.1)
        self.fwhm_spinbox.setDecimals(1)
        self.fwhm_spinbox.setValue(3.0)
        self.fwhm_spinbox.setMinimumWidth(80)
        self.fwhm_spinbox.setSuffix(" σ")
        self.fwhm_spinbox.setEnabled(False)
        self.fwhm_spinbox.setToolTip(fwhm_label_tooltip)
        preprocessing_layout.addWidget(self.fwhm_spinbox, 5, 3)

        self.filters_checkbox.toggled.connect(self.fwhm_spinbox.setEnabled)

        # Add to create_widgets method
        stacking_section = QGroupBox("Stacking")
        stacking_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        main_layout.addWidget(stacking_section)
        stacking_layout = QGridLayout(stacking_section)
        stacking_layout.setSpacing(5)
        stacking_layout.setContentsMargins(12, 18, 12, 12)

        # Remove stars checkbox
        self.remove_stars_checkbox = QCheckBox("Remove Stars")
        self.remove_stars_checkbox.setToolTip("""
            Remove stars from individual images and save them as 'starless_' sequence. 
            \nNOTE: This will create a final 'stars_stacked' image but will NOT create a comet stacked image. You must do that yourself.
        """)
        stacking_layout.addWidget(self.remove_stars_checkbox, 0, 0)
        # If starnet is not available, disable the remove stars checkbox
        self.remove_stars_checkbox.setEnabled(self.starnet_available)
        if not self.starnet_available:
            self.remove_stars_checkbox.setToolTip("Starnet++ is not available. Please install it first.")

        # Stacking algorithm dropdown
        stacking_algorithm_label = QLabel("Pixel Rejection Algorithm:")
        stacking_layout.addWidget(stacking_algorithm_label, 1, 0)

        self.stacking_algorithm_combo = QComboBox()
        self.stacking_algorithm_combo.addItems([
            "Sigma Clipping",
            "MAD Clipping",
            "Winsorized Sigma Clipping",
            "Linear Fit Clipping"
        ])
        stacking_layout.addWidget(self.stacking_algorithm_combo, 1, 1, 1, 2)

        # Sigma high/low spinboxes
        sigma_high_label = QLabel("Sigma High:")
        stacking_layout.addWidget(sigma_high_label, 2, 0)

        self.sigma_high_spinbox = QDoubleSpinBox()
        self.sigma_high_spinbox.setRange(0.1, 10.0)
        self.sigma_high_spinbox.setSingleStep(0.1)
        self.sigma_high_spinbox.setDecimals(1)
        self.sigma_high_spinbox.setValue(3.0)
        self.sigma_high_spinbox.setSuffix(" σ")
        stacking_layout.addWidget(self.sigma_high_spinbox, 2, 1)

        sigma_low_label = QLabel("Sigma Low:")
        stacking_layout.addWidget(sigma_low_label, 2, 2)

        self.sigma_low_spinbox = QDoubleSpinBox()
        self.sigma_low_spinbox.setRange(0.1, 10.0)
        self.sigma_low_spinbox.setSingleStep(0.1)
        self.sigma_low_spinbox.setDecimals(1)
        self.sigma_low_spinbox.setValue(3.0)
        self.sigma_low_spinbox.setSuffix(" σ")
        stacking_layout.addWidget(self.sigma_low_spinbox, 2, 3)

        # self.remove_stars_checkbox.toggled.connect(self.stacking_algorithm_combo.setDisabled)
        # self.remove_stars_checkbox.toggled.connect(self.sigma_high_spinbox.setDisabled)
        # self.remove_stars_checkbox.toggled.connect(self.sigma_low_spinbox.setDisabled)

        # Add Timelapse Settings section
        timelapse_settings = QGroupBox("Timelapse Settings")
        timelapse_settings.setStyleSheet("QGroupBox { font-weight: bold; }")
        main_layout.addWidget(timelapse_settings)
        timelapse_layout = QGridLayout(timelapse_settings)
        timelapse_layout.setSpacing(5)
        timelapse_layout.setContentsMargins(12, 18, 12, 12)

        # Main timelapse enable checkbox
        timelapse_label = QLabel("Enable Timelapse:")
        timelapse_label.setFont(title_font)
        timelapse_tooltip = "Apply autostretch to all images in the sequence and create a timelapse"
        timelapse_label.setToolTip(timelapse_tooltip)
        timelapse_layout.addWidget(timelapse_label, 0, 0)

        self.timelapse_checkbox = QCheckBox("")
        self.timelapse_checkbox.setToolTip(timelapse_tooltip)
        timelapse_layout.addWidget(self.timelapse_checkbox, 0, 1)

        # FFmpeg path input
        ffmpeg_label = QLabel("FFmpeg Path:")
        timelapse_layout.addWidget(ffmpeg_label, 1, 0)
        
        self.ffmpeg_path_input = QLineEdit()
        self.ffmpeg_path_input.setPlaceholderText("Path to ffmpeg executable")
        self.ffmpeg_path_input.setEnabled(False)
        timelapse_layout.addWidget(self.ffmpeg_path_input, 1, 1, 1, 3)
        
        # Framerate spinbox
        framerate_label = QLabel("Framerate:")
        timelapse_layout.addWidget(framerate_label, 2, 0)
        
        self.framerate_spinbox = QSpinBox()
        self.framerate_spinbox.setRange(1, 60)
        self.framerate_spinbox.setValue(24)  # Default value
        self.framerate_spinbox.setSuffix(" fps")
        self.framerate_spinbox.setEnabled(False)
        self.framerate_spinbox.setToolTip("Frames per second for the output video (1-60 fps)")
        timelapse_layout.addWidget(self.framerate_spinbox, 2, 1)
        
        # Quality/CRF spinbox
        quality_label = QLabel("Quality (CRF):")
        timelapse_layout.addWidget(quality_label, 2, 2)
        
        self.crf_spinbox = QSpinBox()
        self.crf_spinbox.setRange(0, 51)  # 0 is lossless, 51 is worst
        self.crf_spinbox.setValue(23)  # Default value
        self.crf_spinbox.setEnabled(False)
        self.crf_spinbox.setToolTip("Video quality (0=lossless, 23=default, 51=worst)")
        timelapse_layout.addWidget(self.crf_spinbox, 2, 3)
        
        # Connect timelapse checkbox to enable/disable all timelapse settings
        self.timelapse_checkbox.toggled.connect(self.ffmpeg_path_input.setEnabled)
        self.timelapse_checkbox.toggled.connect(self.framerate_spinbox.setEnabled)
        self.timelapse_checkbox.toggled.connect(self.crf_spinbox.setEnabled)


        # Buttons section
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(
            0, 15, 0, 0
        )  # Add top margin to separate from content
        main_layout.addLayout(button_layout)

        help_button = QPushButton("Help")
        help_button.setMinimumWidth(50)
        help_button.setMinimumHeight(35)
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        save_presets_button = QPushButton("Save Presets")
        save_presets_button.setMinimumWidth(80)
        save_presets_button.setMinimumHeight(35)
        save_presets_button.clicked.connect(self.save_presets)
        button_layout.addWidget(save_presets_button)

        load_presets_button = QPushButton("Load Presets")
        load_presets_button.setMinimumWidth(80)
        load_presets_button.setMinimumHeight(35)
        load_presets_button.clicked.connect(self.load_presets)
        button_layout.addWidget(load_presets_button)

        button_layout.addStretch()  # Add space between buttons

        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setMinimumHeight(35)
        close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(close_button)

        # Add small spacing between close and run buttons
        button_layout.addSpacing(10)

        run_button = QPushButton("Run")
        run_button.setMinimumWidth(100)
        run_button.setMinimumHeight(35)
        run_button.setStyleSheet("QPushButton { font-weight: bold; }")
        run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(run_button)

        # Add stretch to push everything to the top
        main_layout.addStretch()

    def on_run_clicked(self):
        """Handle the Run button click"""
        self.run_script(
            use_darks=self.darks_checkbox.isChecked(),
            use_flats=self.flats_checkbox.isChecked(),
            use_biases=self.biases_checkbox.isChecked(),
            bg_extract=self.bg_extract_checkbox.isChecked(),
            filter_roundness=self.roundness_spinbox.value(),
            filter_fwhm=self.fwhm_spinbox.value(),
            clean_up_files=self.cleanup_files_checkbox.isChecked(),
        )

    def close_dialog(self):
        self.siril.disconnect()
        self.close()

    def extract_coords_from_fits(self, prefix: str):
        # Only process for specific D2 and Origin
        process_dir = "process"
        matching_files = sorted(
            [
                f
                for f in os.listdir(process_dir)
                if f.startswith(prefix) and f.lower().endswith(self.fits_extension)
            ]
        )

        if not matching_files:
            self.siril.log(
                f"No FITS files found in '{process_dir}' with prefix '{prefix}'",
                LogColor.RED,
            )
            return

        first_file = matching_files[0]
        print(first_file)
        file_path = os.path.join(process_dir, first_file)

        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                ra = header.get("RA")
                dec = header.get("DEC")

                if ra is not None and dec is not None:
                    self.target_coords = f"{ra},{dec}"
                    self.siril.log(
                        f"Target coordinates extracted: {self.target_coords}",
                        LogColor.GREEN,
                    )
                else:
                    self.siril.log(
                        "RA or DEC not found in FITS header.", LogColor.SALMON
                    )
        except Exception as e:
            self.siril.log(f"Error reading FITS header: {e}", LogColor.RED)

    # TODO: Remove function in RC1
    def swap_red_blue_channels(self, image_path):
        """Swaps the red and blue channels of a FITS image to mitigate Siril bug for seestars"""
        try:
            self.siril.log(
                "Swapping red and blue channels using Python...", LogColor.BLUE
            )

            # Read the FITS file
            with fits.open(image_path) as hdul:
                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                if data.ndim == 3 and data.shape[0] == 3:
                    # Swap channels: [R, G, B] -> [B, G, R]
                    data[[0, 2]] = data[[2, 0]]

                    base_name = os.path.splitext(image_path)[0]
                    output_path = f"{base_name}_RBswapped{self.fits_extension}"

                    hdul_out = fits.PrimaryHDU(data=data, header=header)
                    hdul_out.writeto(output_path, overwrite=True)

                    self.siril.log(
                        f"Successfully swapped channels and saved: {output_path}",
                        LogColor.GREEN,
                    )
                    return output_path

                else:
                    self.siril.log(
                        f"Image is not a 3-channel color image (shape: {data.shape})",
                        LogColor.SALMON,
                    )
                    return None

        except Exception as e:
            self.siril.log(
                f"Color channel swap failed, may not mean anything: {e}",
                LogColor.SALMON,
            )
            return None

    def batch(
        self,
        output_name: str,
        use_darks: bool = False,
        use_flats: bool = False,
        use_biases: bool = False,
        bg_extract: bool = False,
        filter_roundness: float = 3.0,
        filter_fwhm: float = 3.0,
        clean_up_files: bool = False,
    ):
        # If we're batching, force cleanup files so we don't collide with existing files
        self.siril.cmd("close")
        if output_name.startswith("batch_lights"):
            clean_up_files = True

        # Output name is actually the name of the batched working directory
        self.convert_files(dir_name=output_name)
        # self.unselect_bad_fits(seq_name=output_name)

        seq_name = f"{output_name}_"

        # self.siril.cmd("cd", batch_working_dir)

        # Using calibration frames puts pp_ prefix in process directory
        if True:
            self.calibrate_lights(
                seq_name=seq_name, use_darks=use_darks, use_flats=use_flats
            )
            try:
                if clean_up_files:
                    self.clean_up(
                        prefix=seq_name
                    )  # Remove "batch_lights_" or just "lights_" if not flat calibrated
            except Exception as e:
                self.siril.log(
                    f"Error during cleanup after calibration: {e}", LogColor.SALMON
                )
            seq_name = "pp_" + seq_name

        if bg_extract:
            self.seq_bg_extract(seq_name=seq_name)
            if clean_up_files:
                self.clean_up(
                    prefix=seq_name
                )  # Remove "pp_lights_" or just "lights_" if not flat calibrated
            seq_name = "bkg_" + seq_name

        # seq_name stays the same after plate solve
        self.register_sequence(seq_name=seq_name)
        self.seq_apply_reg(
            seq_name=seq_name,
            filter_roundness=filter_roundness,
            filter_fwhm=filter_fwhm,
        )
        if clean_up_files:
            self.clean_up(
                prefix=seq_name
            )  # Clean up bkg_ files or pp_ if flat calibrated, otherwise lights_
        seq_name = f"r_{seq_name}"
        
        if self.timelapse_checkbox.isChecked():
            self.seq_autostretch(seq_name=seq_name)

            input_folder = "process"
            output_folder = "timelapse_images"
            
            os.makedirs(output_folder, exist_ok=True)

            # interval = ZScaleInterval()

            for filename in os.listdir(input_folder):
                if filename.startswith("tmlps_") and filename.endswith(".tif"):
                    filepath = os.path.join(input_folder, filename)
                    # Move TIF files to output folder
                    shutil.move(filepath, os.path.join(output_folder, filename))
            
            # Import here to avoid requiring ffmpeg for non-timelapse users
            
            # Create timelapse output directory
            # timelapse_dir = os.path.join(self.current_working_directory, "timelapse")
            # os.makedirs(timelapse_dir, exist_ok=True)
            
            # Use ffmpeg to create video from sequence
            print(os.path.abspath(output_folder))
            try:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_mp4 = f"timelapse_{current_time}.mp4"
                ffmpeg_cmd = [
                    self.ffmpeg_path_input.text() if self.ffmpeg_path_input.text() else 'ffmpeg',
                    '-y',  # Overwrite output file if exists
                    '-framerate', str(self.framerate_spinbox.value()) if self.framerate_spinbox.value() else '24',  # Frames per second
                    '-pattern_type', 'sequence',
                    '-i', 'tmlps_%05d.tif',  # Input pattern
                    '-c:v', 'libx264',  # Video codec
                    '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                    '-crf', str(self.crf_spinbox.value()) if self.crf_spinbox.value() else '23',  # Quality (lower is better, 23 is default)
                    output_mp4
                ]
                result = subprocess.run(ffmpeg_cmd, check=True, cwd=output_folder, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                self.siril.log(f"Created timelapse video: {output_mp4}", LogColor.GREEN)
                if result.returncode != 0:
                    self.siril.log(result.stderr.decode(), LogColor.RED)
                else:
                    self.siril.log(f"Created timelapse video: {output_mp4}", LogColor.GREEN)
            except subprocess.CalledProcessError as e:
                self.siril.log(f"Failed to create timelapse video: {e}", LogColor.RED)
            except FileNotFoundError:
                self.siril.log("ffmpeg not found. Please install ffmpeg to create timelapses.", LogColor.RED)


        if self.remove_stars_checkbox.isChecked():            
            # self.remove_stars(sequence=True, seq_name=seq_name)
            self.seq_starnet(seq_name=seq_name)
            # seq_name = f"starless_{seq_name}"

        # Still stack starless file
        self.seq_stack(
            seq_name=seq_name,
            rejection=True,
            output_name=output_name,
        )
    
        # Load the result (e.g. batch_lights_001.fits)
        self.load_image(image_name=output_name)
        # removes stars from currently loaded image
        if self.remove_stars_checkbox.isChecked():
            self.remove_stars(sequence=False)

            try: 
                shutil.move(
                    os.path.join(self.siril.get_siril_wd(), "starmask_lights.fits"),
                    os.path.join(os.path.dirname(self.siril.get_siril_wd()), "stars.fits")
                )
                self.siril.log("Moved stars.fits to working directory", LogColor.GREEN)
            except Exception as e:
                self.siril.log(f"Failed to move stars.fits: {e}", LogColor.SALMON)

        if clean_up_files:
            # Don't clean up registered files because we need them for comet stacking
            # self.clean_up(prefix=seq_name)  # clean up r_ files
            try:
                shutil.rmtree(os.path.join(self.siril.get_siril_wd(), "cache"))
            except Exception as e:
                self.siril.log(
                    f"Error cleaning up temporary files, continuing with the script: {e}",
                    LogColor.SALMON,
                )

        
        # self.siril.get_seq_frame(seq_name=seq_name)
        # Go back to working dir
        self.siril.cmd("cd", "../")

        # Save og image in WD - might have drizzle factor in name
        if output_name.startswith("batch_lights"):
            out = output_name
        else:
            out = "og"
        file_name = self.save_image(f"_{out}")

        return file_name

    def unselect_bad_fits(self, seq_name, folder="process"):
        """
        Checks all FITS files in the given folder with the given prefix
        for integrity and unselects any bad ones in the current sequence
        in Siril.

        Parameters
        ----------
        seq_name : str
            The prefix of the sequence to check.
        folder : str, optional
            The folder to check for FITS files. Defaults to "process".

        Returns
        -------
        None
        """
        self.siril.log("Checking for bad FITS files...", LogColor.BLUE)
        bad_fits = []
        all_files = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.startswith(seq_name) and f.lower().endswith(self.fits_extension)
            ]
        )
        for idx, filename in enumerate(all_files):
            file_path = os.path.join(folder, filename)
            try:
                with fits.open(file_path) as hdul:
                    _ = hdul[0].data  # Try to access data

            except Exception as e:
                self.siril.log(f"Bad FITS file: {filename} — {e}", LogColor.SALMON)
                bad_fits.append(idx + 1)  # Siril indices start at 1

        if bad_fits:
            self.siril.log(f"Unselecting bad frames: {bad_fits}", LogColor.SALMON)
            for index in bad_fits:
                try:
                    self.siril.cmd("unselect", seq_name, index, index)
                except Exception as e:
                    self.siril.log(
                        f"Failed to unselect index {index}: {e}", LogColor.RED
                    )
        else:
            self.siril.log("No bad FITS files found.", LogColor.GREEN)

    # Save and Load Presets code
    def save_presets(self):
        """Save current UI settings to a JSON file in the working directory."""
        presets = {
            "darks": self.darks_checkbox.isChecked(),
            "flats": self.flats_checkbox.isChecked(),
            "biases": self.biases_checkbox.isChecked(),
            "cleanup": self.cleanup_files_checkbox.isChecked(),
            "bg_extract": self.bg_extract_checkbox.isChecked(),
            "filters": self.filters_checkbox.isChecked(),
            "roundness": self.roundness_spinbox.value(),
            "fwhm": self.fwhm_spinbox.value(),

            # New stacking options
            "remove_stars": self.remove_stars_checkbox.isChecked(),
            "stacking_algorithm": self.stacking_algorithm_combo.currentText(),
            "sigma_high": self.sigma_high_spinbox.value(),
            "sigma_low": self.sigma_low_spinbox.value(),

            # Timelapse options
            "timelapse": self.timelapse_checkbox.isChecked(),
            "ffmpeg_path": self.ffmpeg_path_input.text(),
            "framerate": self.framerate_spinbox.value(),
            "crf": self.crf_spinbox.value()
        }

        presets_dir = os.path.join(self.current_working_directory, "presets")
        os.makedirs(presets_dir, exist_ok=True)
        presets_file = os.path.join(presets_dir, "naztronomy_comet_processing_presets.json")

        try:
            with open(presets_file, "w") as f:
                json.dump(presets, f, indent=4)
            self.siril.log(f"Saved presets to {presets_file}", LogColor.GREEN)
        except Exception as e:
            self.siril.log(f"Failed to save presets: {e}", LogColor.RED)

    def load_presets(self):
        """Load UI settings from JSON file using file dialog."""
        try:
            # Open file dialog to select presets file
            # First check for default presets file
            default_presets_file = os.path.join(
                self.current_working_directory,
                "presets",
                "naztronomy_comet_processing_presets.json",
            )

            if os.path.exists(default_presets_file):
                presets_file = default_presets_file
            else:
                # If default presets don't exist, show file dialog
                presets_file, _ = QFileDialog.getOpenFileName(
                    self,
                    "Load Presets",
                    os.path.join(self.current_working_directory, "presets"),
                    "JSON Files (*.json);;All Files (*.*)",
                )

                if not presets_file:  # User canceled
                    self.siril.log("Preset loading canceled", LogColor.BLUE)
                    return

            with open(presets_file) as f:
                presets = json.load(f)

            # Set UI elements based on loaded presets
            self.darks_checkbox.setChecked(presets.get("darks", False))
            self.flats_checkbox.setChecked(presets.get("flats", False))
            self.biases_checkbox.setChecked(presets.get("biases", False))
            self.cleanup_files_checkbox.setChecked(presets.get("cleanup", False))
            self.bg_extract_checkbox.setChecked(presets.get("bg_extract", False))
            self.filters_checkbox.setChecked(presets.get("filters", False))
            self.roundness_spinbox.setValue(presets.get("roundness", 3.0))
            self.fwhm_spinbox.setValue(presets.get("fwhm", 3.0))

            # Load new stacking options
            self.remove_stars_checkbox.setChecked(presets.get("remove_stars", False))
            algorithm = presets.get("stacking_algorithm", "Sigma Clipping")
            index = self.stacking_algorithm_combo.findText(algorithm)
            if index >= 0:
                self.stacking_algorithm_combo.setCurrentIndex(index)
            self.sigma_high_spinbox.setValue(presets.get("sigma_high", 3.0))
            self.sigma_low_spinbox.setValue(presets.get("sigma_low", 3.0))

            # Load timelapse options
            self.timelapse_checkbox.setChecked(presets.get("timelapse", False))
            self.ffmpeg_path_input.setText(presets.get("ffmpeg_path", ""))
            self.framerate_spinbox.setValue(presets.get("framerate", 24))
            self.crf_spinbox.setValue(presets.get("crf", 23))

            self.siril.log(f"Loaded presets from {presets_file}", LogColor.GREEN)
        except Exception as e:
            self.siril.log(f"Failed to load presets: {e}", LogColor.RED)

    def run_script(
        self,
        use_darks: bool = False,
        use_flats: bool = False,
        use_biases: bool = False,
        max_files_per_batch: float = UI_DEFAULTS["max_files_per_batch"],
        bg_extract: bool = False,
        filter_roundness: float = 3.0,
        filter_fwhm: float = 3.0,
        clean_up_files: bool = False,
    ):
        self.siril.log(
            f"Running script version {VERSION} with arguments:\n"
            f"use_darks={use_darks}\n"
            f"use_flats={use_flats}\n"
            f"use_biases={use_biases}\n"
            f"batch_size={max_files_per_batch}\n"
            f"bg_extract={bg_extract}\n"
            f"filter_roundness={filter_roundness}\n"
            f"filter_fwhm={filter_fwhm}\n"
            f"clean_up_files={clean_up_files}\n"
            f"build={VERSION}-{BUILD}",
            LogColor.BLUE,
        )
        self.siril.cmd("close")

        # Check if old processing directories exist
        if (
            os.path.exists("sessions")
            or os.path.exists("process")
            or os.path.exists("final_stack")
        ):
            msg = "Old processing directories found. Do you want to delete them and start fresh?"
            answer = QMessageBox.question(
                self,
                "Old Processing Files Found",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    if os.path.exists("sessions"):
                        shutil.rmtree("sessions")
                        self.siril.log("Cleaned up old sessions directories", LogColor.BLUE)
                    if os.path.exists("process"):
                        shutil.rmtree("process") 
                        self.siril.log("Cleaned up old process directory", LogColor.BLUE)
                    if os.path.exists("final_stack"):
                        shutil.rmtree("final_stack")
                        self.siril.log(
                            "Cleaned up old final_stack directory", LogColor.BLUE
                        )
                except (PermissionError, OSError) as e:
                    error_msg = ("Unable to delete one or more processing directories.\n\n"
                               "Please manually delete the following directories if they exist:\n"
                               "- sessions\n"
                               "- process\n" 
                               "- final_stack\n\n"
                               "Then click Run again to continue.")
                    QMessageBox.critical(self, "Manual Cleanup Required", error_msg)
                    self.siril.log("Error cleaning up directories: " + str(e), LogColor.RED)
                    return
            else:
                self.siril.log(
                    "User chose to preserve old processing files. Stopping script.",
                    LogColor.BLUE,
                )
                return

        # TODO: Stack calibration frames and copy to the various batch dirs
        if use_biases:
            converted = self.convert_files("biases")
            if converted:
                self.calibration_stack("biases")
            if clean_up_files:
                self.clean_up("biases")
        if use_flats:
            converted = self.convert_files("flats")
            if converted:
                self.calibration_stack("flats")
            if clean_up_files:
                self.clean_up("flats")
        if use_darks:
            converted = self.convert_files("darks")
            if converted:
                self.calibration_stack("darks")
            if clean_up_files:
                self.clean_up("darks")

        # Check files in working directory/lights.
        # create sub folders with more than 2048 divided by equal amounts

        lights_directory = "lights"

        # Get list of all files in the lights directory
        all_files = [
            name
            for name in os.listdir(lights_directory)
            if os.path.isfile(os.path.join(lights_directory, name))
        ]
        num_files = len(all_files)
        is_windows = sys.platform.startswith("win")

        # only one batch will be run if less than max_files_per_batch OR not windows.
        if num_files <= max_files_per_batch: # or not is_windows:
            self.siril.log(
                f"{num_files} files found in the lights directory which is less than or equal to {max_files_per_batch} files allowed per batch - no batching needed.",
                LogColor.BLUE,
            )
            file_name = self.batch(
                output_name=lights_directory,
                use_darks=use_darks,
                use_flats=use_flats,
                use_biases=use_biases,
                bg_extract=bg_extract,
                filter_roundness=filter_roundness,
                filter_fwhm=filter_fwhm,
                clean_up_files=clean_up_files,
            )

            self.load_image(image_name=file_name)
        else:
            num_batches = math.ceil(num_files / max_files_per_batch)

            self.siril.log(
                f"{num_files} files found in the lights directory, splitting into {num_batches} batches...",
                LogColor.BLUE,
            )

            # Ensure temp folders exist and are empty
            for i in range(num_batches):
                batch_dir = f"batch_lights{i+1}"
                os.makedirs(batch_dir, exist_ok=True)
                # Optionally clean out existing files:
                for f in os.listdir(batch_dir):
                    os.remove(os.path.join(batch_dir, f))

            # Split and create symlinks/copies of files into batches
            def copy_file(filename, src_dir, batch_info):
                max_files = batch_info['max_files']
                i = batch_info['file_index'][filename]
                batch_index = i // max_files
                batch_dir = f"batch_lights{batch_index + 1}"
                src_path = os.path.join(src_dir, filename)
                dest_path = os.path.join(batch_dir, filename)
                
                try:
                    shutil.copy2(src_path, dest_path)
                    self.siril.log(f"Copied: {filename} to {batch_dir}", LogColor.BLUE)
                except Exception as e:
                    self.siril.log(f"Error copying {filename}: {e}", LogColor.SALMON)

            # Create a dict with file indices for parallel processing
            batch_info = {
                'max_files': max_files_per_batch,
                'file_index': {filename: i for i, filename in enumerate(all_files)}
            }

            # Use a partial function to avoid passing same args repeatedly
            copy_func = partial(copy_file, src_dir=lights_directory, batch_info=batch_info)

            # Use ThreadPoolExecutor for parallel file copying
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
                list(executor.map(copy_func, all_files))

            self.siril.log(f"Finished copying {len(all_files)} files to batch directories", LogColor.GREEN)

            # Send each of the new lights dir into batch directory
            for i in range(num_batches):
                batch_dir = f"batch_lights{i+1}"
                self.siril.log(f"Processing batch: {batch_dir}", LogColor.BLUE)
                self.batch(
                    output_name=batch_dir,
                    use_darks=use_darks,
                    use_flats=use_flats,
                    use_biases=use_biases,
                    bg_extract=bg_extract,
                    filter_roundness=filter_roundness,
                    filter_fwhm=filter_fwhm,
                    clean_up_files=clean_up_files,
                )
            self.siril.log("Batching complete.", LogColor.GREEN)

            # Create batched_lights directory
            final_stack_seq_name = "final_stack"
            batch_lights = "batch_lights"
            os.makedirs(final_stack_seq_name, exist_ok=True)
            source_dir = os.path.join(os.getcwd(), "process")
            # Move batch result files into batched_lights
            target_subdir = os.path.join(os.getcwd(), final_stack_seq_name)

            # Create the target subdirectory if it doesn't exist
            os.makedirs(target_subdir, exist_ok=True)

            # Loop through all files in the source directory
            for filename in os.listdir(source_dir):
                if f"{batch_lights}" in filename:
                    full_src_path = os.path.join(source_dir, filename)
                    full_dst_path = os.path.join(target_subdir, filename)

                # Only move files, skip directories
                # Should only moved the final batched files
                if os.path.isfile(full_src_path):
                    shutil.move(full_src_path, full_dst_path)
                    self.siril.log(f"Moved: {filename}", LogColor.BLUE)

            # Clean up temp_lightsX directories
            for i in range(num_batches):
                batch_dir = f"{batch_lights}{i+1}"
                shutil.rmtree(batch_dir, ignore_errors=True)

            self.convert_files(final_stack_seq_name)
            self.register_sequence(seq_name=final_stack_seq_name)

            # Force filters to 3 sigma
            self.seq_apply_reg(
                seq_name=final_stack_seq_name,
                filter_roundness=3.0,
                filter_fwhm=3.0,
            )
            self.clean_up(prefix=final_stack_seq_name)
            registered_final_stack_seq_name = f"r_{final_stack_seq_name}"
            # final stack needs feathering and amount
            self.seq_stack(
                seq_name=registered_final_stack_seq_name,
                rejection=False,
                output_name="final_result",
            )
            self.load_image(image_name="final_result")

            # cleanup final_stack directory
            # shutil.rmtree(final_stack_seq_name, ignore_errors=True)
            # Don't clean up registered frames because we need it for comet stacking
            # self.clean_up(prefix=registered_final_stack_seq_name)

            # Go back to working dir
            self.siril.cmd("cd", "../")

            # Save og image in WD - might have drizzle factor in name
            file_name = self.save_image("_batched")
            self.load_image(image_name=file_name)

        # self.clean_up()

        # TODO: Remove in RC1
        # if bg extraction AND drizzle are checked, we swap the channels to mitigate a siril bug that's only exists for Seestars
        # self.siril.log("Checking if color channel swap is needed...", LogColor.BLUE)
        # self.siril.log(
        #     f"Telescope: {telescope}, Drizzle: {self.drizzle_status}, BG Extract: {self.bg_extract_checkbox.isChecked()}",
        #     LogColor.BLUE,
        # )
        # if (
        #     self.bg_extract_checkbox.isChecked()
        #     and self.drizzle_status
        #     and telescope in ["ZWO Seestar S50", "ZWO Seestar S30"]
        # ):
        #     img_path = file_name + self.fits_extension
        #     self.swap_red_blue_channels(image_path=img_path)
        #     self.siril.log(
        #         "If the colors look off, please load the RBswapped image.",
        #         LogColor.SALMON,
        #     )

        self.siril.log(
            f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            LogColor.GREEN,
        )
        self.siril.log(
            """
        Thank you for using the Naztronomy Smart Telescope Preprocessor! 
        The author of this script is Nazmus Nasir (Naztronomy).
        Website: https://www.Naztronomy.com 
        YouTube: https://www.YouTube.com/Naztronomy 
        Discord: https://discord.gg/yXKqrawpjr
        Patreon: https://www.patreon.com/c/naztronomy
        Buy me a Coffee: https://www.buymeacoffee.com/naztronomy
        """,
            LogColor.BLUE,
        )

        self.close_dialog()


def main():
    try:
        app = QApplication(sys.argv)
        window = PreprocessingInterface()

        # Only show window if initialization was successful
        if window.initialization_successful:
            window.show()
            sys.exit(app.exec())
        else:
            # User canceled during initialization - exit gracefully
            sys.exit(0)
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


##############################################################################

# Website: https://www.Naztronomy.com
# YouTube: https://www.YouTube.com/Naztronomy
# Discord: https://discord.gg/yXKqrawpjr
# Patreon: https://www.patreon.com/c/naztronomy
# Buy me a Coffee: https://www.buymeacoffee.com/naztronomy

##############################################################################
