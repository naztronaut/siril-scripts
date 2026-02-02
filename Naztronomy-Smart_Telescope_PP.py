"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Naztronomy - Smart Telescope Preprocessing script
Version: 2.0.6
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

2.0.6 - Ignore dot files from macs
      - Fix black frames check bug
      - PR#75 - support compressed fits in lights dir
      - Refactored UI code for better maintainability
      - Allow safe cancellation of processing
      - Added safe deletes 
      - Added stack weighting option (Noise, Number of Stars, Weighted FWHM)
2.0.5 - Bugfix: Black Frames Scan now sees both compressed and uncompressed fits
      - Bugfix: Compression turned on at batch instead of run code
2.0.4 - Compression is now an optional checkbox
      - Compression is turned off when failed in try/except blocks
      - Compression can still be left on if the script crashes another way or the user ends the script manually
      - Fix bug: Disable SPCC for Celestron Origin
2.0.3 - Support for new Unistellar/Evscope telescopes (nicastel)
      - Compression during processing (enabled by default and controlled by a flag)
      - Forcing stacking to use -32b to fix Dwarf3's "milky" stacked image
      - Better error handling with clean_ups
      - Force local photometry catalog for SPCC
      - Better gaia status layout in GUI
      - Updated and more Tooltips
      - Output stacking details
      - Fixed max frames bug for linux and mac
      - Updated tolerance for BGE 
      - Add Dwarf Mini Support
2.0.2 - Small Bug fixes
      - Reenable feathering
      - Fixed pixel fraction decimal precision
      - Added 'DWARF 3' to auto find telescope from FITS header
      - Disallow SPCC for Celestron Origin
      - Bypasss seqplatesolve false error for now
      - Issue #56 - don't crash if there are no lights
2.0.1 - Allowing all os to batch
      - Batch min size set to 50. Batch Max Size set based on OS: Windows 2000, Linux/Mac 25000
      - Optional Black Frames Check
      - Automatic Telescope Detection from FITS Header when available
      - Removed feathering. Automatic feathering of panels still work.
      - Fallback to regular registration if plate solving fails (which should accommodate any telescope now) and will not mosaic
      - Added additional filters: background and star count
      - Filters used only if checkbox is checked without default fallback
      - Removed rbswapped file for Siril 1.4 RC1
      - Full Celestron Origin Support - latest version of Celestron firmware only
2.0.0 - Major version update:
      - Refactored code to use Qt6 instead of Tkinter for the GUI
      - Exposed extra filter options
      - Allow changing batch size
      - Accepts master calibration frames (also creates master calibration frames)
      - Temporary workaround to cfa debayering bug in Siril when using drizzle and background extraction for seestars
1.1.1 - Bug fixes: 
      - Fixed Celestron Origin focal length to 335mm
      - Fixed clean up for pre-pp files
1.1.0 - Minor version update:
      - Added Batching support for 2000+ files on Windows
      - Removed Autocrop due to reported errors
      - Added support for Dwarf 2 and Celestron Origin
1.0.1 - minor refactoring to work with both .fit and .fits outputs (e.g. result.fit vs result.fits)
  - added support autocrop script created by Gottfried Rotter
1.0.0 - initial release
"""

import os
import sys
import math
import shutil
import time
import sirilpy as s
from datetime import datetime
import json


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
    QScrollArea,
    QProgressBar,
)
from PyQt6.QtCore import pyqtSlot as Slot, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QShortcut, QKeySequence
from sirilpy import LogColor, NoImageError
from astropy.io import fits
import numpy as np


# from tkinter import filedialog

APP_NAME = "Naztronomy - Smart Telescope Preprocessing"
VERSION = "2.0.6"
BUILD = "20260201"
AUTHOR = "Nazmus Nasir"
WEBSITE = "Naztronomy.com"
YOUTUBE = "YouTube.com/Naztronomy"
TELESCOPES = [
    "ZWO Seestar S30",
    "ZWO Seestar S30 Pro",
    "ZWO Seestar S50",
    "Dwarf Mini",
    "Dwarf 3",
    "Dwarf 2",
    "Celestron Origin",
    "Unistellar eVscope 1 / eQuinox 1",
    "Unistellar eVscope 2 / eQuinox 2",
    "Unistellar Odyssey / Odyssey Pro",
]

FILTER_OPTIONS_MAP = {
    "ZWO Seestar S30": ["No Filter (Broadband)", "LP (Narrowband)"],
    "ZWO Seestar S30 Pro": ["No Filter (Broadband)", "LP (Narrowband)"],
    "ZWO Seestar S50": ["No Filter (Broadband)", "LP (Narrowband)"],
    "Dwarf Mini": ["Astro filter (UV/IR)", "Dual-Band"],
    "Dwarf 3": ["Astro filter (UV/IR)", "Dual-Band"],
    "Dwarf 2": ["Astro filter (UV/IR)"],
    "Celestron Origin": ["No Filter (Broadband)"],
    "Unistellar eVscope 1 / eQuinox 1": ["No Filter (Broadband)"],
    "Unistellar eVscope 2 / eQuinox 2": ["No Filter (Broadband)"],
    "Unistellar Odyssey / Odyssey Pro": ["No Filter (Broadband)"],
}

FILTER_COMMANDS_MAP = {
    "ZWO Seestar S30": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
        "LP (Narrowband)": ["-oscfilter=ZWO Seestar LP"],
    },
    "ZWO Seestar S30 Pro": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
        "LP (Narrowband)": ["-oscfilter=ZWO Seestar LP"],
    },
    "ZWO Seestar S50": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
        "LP (Narrowband)": ["-oscfilter=ZWO Seestar LP"],
    },
    "Dwarf Mini": {
        "Astro filter (UV/IR)": ["-oscfilter=UV/IR Block"],
        "Dual-Band": [
            "-narrowband",
            "-rwl=656.28",
            "-rbw=18",
            "-gwl=500.70",
            "-gbw=30",
            "-bwl=500.70",
            "-bbw=30",
        ],
    },
    "Dwarf 3": {
        "Astro filter (UV/IR)": ["-oscfilter=UV/IR Block"],
        "Dual-Band": [
            "-narrowband",
            "-rwl=656.28",
            "-rbw=18",
            "-gwl=500.70",
            "-gbw=30",
            "-bwl=500.70",
            "-bbw=30",
        ],
    },
    "Dwarf 2": {"Astro filter (UV/IR)": ["-oscfilter=UV/IR Block"]},
    "Celestron Origin": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
    },
}


class WorkerThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.kwargs["progress_callback"] = self.progress.emit
            self.kwargs["check_cancel"] = self.isInterruptionRequested
            self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


UI_DEFAULTS = {
    "feather_amount": 20,
    "drizzle_amount": 1.0,
    "pixel_fraction": 1.0,
    "max_files_per_batch": 2000,
    "win_max_files_per_batch": 2000,
    "mac_max_files_per_batch": 25000,
    "linux_max_files_per_batch": 25000,
}


class PreprocessingInterface(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")

        self.siril = s.SirilInterface()

        # Flags for mosaic mode and drizzle status
        # if drizzle is off, images will be debayered on convert
        self.drizzle_status = False
        self.drizzle_factor = 0
        self.filters_status = False
        self.initialization_successful = False

        # Detect OS and set appropriate max files per batch
        self.max_files_per_batch = 2000  # default
        if sys.platform.startswith("win"):
            self.max_files_per_batch = UI_DEFAULTS["win_max_files_per_batch"]
        elif sys.platform.startswith("linux"):
            self.max_files_per_batch = UI_DEFAULTS["linux_max_files_per_batch"]
        elif sys.platform.startswith("darwin"):
            self.max_files_per_batch = UI_DEFAULTS["mac_max_files_per_batch"]
        else:
            self.max_files_per_batch = UI_DEFAULTS["max_files_per_batch"]

        self.spcc_section = None
        self.spcc_checkbox = None
        self.chosen_telescope = "ZWO Seestar S30"
        self.telescope_options = TELESCOPES
        self.target_coords = None
        self.telescope_combo = None
        self.filter_combo = None

        self.filter_options_map = FILTER_OPTIONS_MAP
        self.current_filter_options = self.filter_options_map["ZWO Seestar S50"]

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

        self.astrometry_gaia_available = False
        try:
            self.astrometry_gaia_status = self.siril.get_siril_config(
                "core", "catalogue_gaia_astro"
            )
            if (
                self.astrometry_gaia_status
                and self.astrometry_gaia_status != "(not set)"
                and os.path.isfile(self.astrometry_gaia_status)
            ):
                self.astrometry_gaia_available = True
        except s.CommandError:
            pass

        self.photometry_gaia_available = False
        try:
            self.photometry_gaia_status = self.siril.get_siril_config(
                "core", "catalogue_gaia_photo"
            )
            if (
                self.photometry_gaia_status
                and self.photometry_gaia_status != "(not set)"
                and os.path.isdir(self.photometry_gaia_status)
            ):
                self.photometry_gaia_available = True
        except s.CommandError:
            pass

        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label_text = ""

        self.initial_message()

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
            if answer == QMessageBox.StandardButton.Yes:
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
        # Initialize fits_files_count before creating widgets
        self.fits_files_count = 0
        self.set_telescope_from_fits()

        # self.setup_shortcuts()
        self.initialization_successful = True

    def initial_message(self):
        msg = f"""Welcome to {APP_NAME} v{VERSION}!
        Please watch latest demos on https://youtube.com/Naztronomy which can answer most questions.
        Here are some Frequently Asked Questions:
        Q: Can it handle telescopes not listed in the dropdown?
        A: Yes, but it will not mosaic them. It will do regular star registration. 
        Q: How do I get support?
        A: Join the Naztronomy Discord server for support and discussion. Please have your logs handy.
        Q: Where can I find the logs?
        A: You can export logs by clicking the download button on the lower right hand side of the console.\n
        """
        self.siril_log_long(msg, LogColor.BLUE)

    def siril_log_long(self, message: str, color=LogColor.DEFAULT):
        """
        Args:
            message: The message to log (can be longer than 1022 bytes)
            color: LogColor enum value for text color (default: LogColor.DEFAULT)
        """
        lines = message.split("\n")
        for line in lines:
            stripped_line = line.lstrip()  # Remove leading whitespace
            if stripped_line:  # Skip empty lines
                self.siril.log(stripped_line, color)

    def set_telescope_from_fits(self):
        """Reads the first FITS file in lights directory and sets telescope based on TELESCOP header."""
        # Mapping from FITS header values to UI telescope names
        # Note: Order matters! Put more specific/longer strings first
        telescope_map = {
            "ZWO Seestar S30 Pro": "ZWO Seestar S30 Pro",
            "ZWO Seestar S30": "ZWO Seestar S30",
            "Seestar S50": "ZWO Seestar S50",
            "Seestar S30": "ZWO Seestar S30",
            "S50": "ZWO Seestar S50",
            "DWARF mini": "Dwarf Mini",
            "DWARFIII": "Dwarf 3",
            "DWARF 3": "Dwarf 3",
            "DWARFII": "Dwarf 2",
            "Origin": "Celestron Origin",
            "eVscope v1.0": "Unistellar eVscope 1 / eQuinox 1",
            "eVscope v2.0": "Unistellar eVscope 2 / eQuinox 2",
            "odyssey": "Unistellar Odyssey / Odyssey Pro",
        }

        try:
            lights_dir = os.path.join(self.current_working_directory, "lights")
            fits_files = [
                f
                for f in os.listdir(lights_dir)
                if f.lower().endswith((".fits", ".fit", ".fits.fz", ".fit.fz"))
            ]

            if not fits_files:
                return

            # Store fits files count to use later
            self.fits_files_count = len(fits_files)
            self.siril.log(
                f"Found {self.fits_files_count} FITS files in lights directory.",
                LogColor.BLUE,
            )
            # Update the label if it exists
            if hasattr(self, "files_found_label"):
                self.files_found_label.setText(
                    f"Fit(s) in lights directory: {self.fits_files_count}"
                )

            first_file = os.path.join(lights_dir, fits_files[0])
            with fits.open(first_file) as hdul:
                header = hdul[0].header
                telescop = header.get("TELESCOP", "")
                creator = header.get("CREATOR", "")
                camera = header.get("CAMERA", "")
                origin = header.get("ORIGIN", "")

                # Try to map telescope name, using startswith for partial matches
                mapped_telescope = "ZWO Seestar S30"  # default
                found_match = False

                # Filter out empty header values
                header_values = [v for v in [telescop, creator, camera] if v]

                # Check map against available headers
                for telescope_local_name, ui_name in telescope_map.items():
                    # Check if any of the effective header values start with this key
                    if any(
                        val.startswith(telescope_local_name) for val in header_values
                    ):
                        mapped_telescope = ui_name
                        found_match = True
                        #  print(f"Matched FITS header to '{mapped_telescope}' using key '{telescope_local_name}'")
                        break

                if origin.startswith("Unistellar"):
                    instrume = header.get("INSTRUME", "NULL")
                    # Dict for Unistellar
                    unistellar_instruments = {
                        "IMX224": "Unistellar eVscope 1 / eQuinox 1",
                        "IMX347": "Unistellar eVscope 2 / eQuinox 2",
                        "IMX415": "Unistellar Odyssey / Odyssey Pro",
                    }
                    for instrument, name in unistellar_instruments.items():
                        if instrume.startswith(instrument):
                            mapped_telescope = name
                            found_match = True
                            break

                if not found_match:
                    self.siril.log(
                        "Couldn't find Telescope info, setting default:", LogColor.BLUE
                    )

                self.telescope_combo.setCurrentText(mapped_telescope)
                self.chosen_telescope = mapped_telescope
                self.siril.log(
                    f"Set telescope to {mapped_telescope} from FITS header",
                    LogColor.BLUE,
                )

        except Exception as e:
            self.siril.log(f"Error reading telescope from FITS: {e}", LogColor.SALMON)

    def fixUnistellarHeaders(self, dir_name):
        dir = os.path.join(self.current_working_directory, dir_name)
        for file in os.listdir(dir):
            if file.upper().endswith("STACKINPUT.FITS") or file.upper().endswith(
                "STACKINPUT.FIT"
            ):
                data, hdr = fits.getdata(os.path.join(dir, file), header=True)
                hdr.set(
                    "RA", hdr["FOVRA"]
                )  # add a RA header based on the FOVRA unistellar header
                hdr.set(
                    "DEC", hdr["FOVDEC"]
                )  # add a DEC header based on the FOVDEC unistellar header
                telescope = None
                if hdr["INSTRUME"].startswith("IMX224"):  # eVscope1 or eQuinox1
                    hdr.set("FOCALLEN", 450.0)  # add a FOCALLEN header
                    hdr.set("XPIXSZ", 3.75)  # add a XPIXSZ header
                    hdr.set("YPIXSZ", 3.75)  # add a YPIXSZ header
                    telescope = "eVscope v1.0"
                if hdr["INSTRUME"].startswith("IMX347"):  # eVscope2 or eQuinox2
                    hdr.set("FOCALLEN", 450.0)  # add a FOCALLEN header
                    hdr.set("XPIXSZ", 2.9)  # add a XPIXSZ header
                    hdr.set("YPIXSZ", 2.9)  # add a YPIXSZ header
                    telescope = "eVscope v2.0"
                if hdr["INSTRUME"].startswith("IMX415"):  # Odyssey or Odyssey Pro
                    hdr.set("FOCALLEN", 320.0)  # add a FOCALLEN header
                    hdr.set("XPIXSZ", 2.9)  # add a XPIXSZ header
                    hdr.set("YPIXSZ", 2.9)  # add a YPIXSZ header
                    telescope = "Odyssey"

                if hdr["SOFTVER"].startswith("4.2") and telescope.startswith(
                    "eVscope"
                ):  # fix for bayer issue with latest FW 4.2
                    hdr.set("XBAYROFF", 0)  # add a XPIXSZ header
                    hdr.set("YBAYROFF", 1)  # add a YPIXSZ header
                else:
                    hdr.set("XBAYROFF", 0)  # add a XPIXSZ header
                    hdr.set("YBAYROFF", 0)  # add a YPIXSZ header

                if hdr.get("TELESCOP") is None and telescope is not None:
                    hdr.set(
                        "TELESCOP", telescope
                    )  # add a TELESCOP header for older FW version

                fits.writeto(os.path.join(dir, file), data, hdr, overwrite=True)
                # print(file)
        self.siril.log("Unistellar headers fixed!", LogColor.GREEN)

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
                    and not name.startswith(".")
                    and (
                        name.lower().endswith(".fit")
                        or name.lower().endswith(".fits")
                        or name.lower().endswith(".fit.fz")
                        or name.lower().endswith(".fits.fz")
                    )
                ]
            )
            self.siril.log(
                f"Found {file_count} files in {dir_name} directory.", LogColor.BLUE
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
                # args = ["convert", dir_name, "-out=../process"]
                # Switched to `link` command to only get fits files
                args = ["link", dir_name, "-out=../process"]
                # If there are no calibration frames or drizzle is off, debayer on convert, otherwise you get a monochrome image
                # if "lights" in dir_name.lower():
                #     if not self.darks_checkbox.isChecked() or not self.flats_checkbox.isChecked() or not self.drizzle_status:
                #             args.append("-debayer")
                self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                self.siril.cmd(*args)
            except (s.DataError, s.CommandError, s.SirilError) as e:
                # turn off compression if error (if checked)
                if self.compression_checkbox.isChecked():
                    self.siril.cmd("setcompress", "0")
                self.siril.log(f"File conversion failed: {e}", LogColor.RED)
                self.close_dialog()

            self.siril.cmd("cd", "../process")
            self.siril.log(
                f"Converted {file_count} {dir_name} files for processing!",
                LogColor.GREEN,
            )
            return True
        else:
            # self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            # raise NoImageError(
            #     (
            #         f'No directory named "{dir_name}" at this location. Make sure the working directory is correct.'
            #     )
            # )
            self.siril.log(
                f'No directory named "{dir_name}" at this location. Make sure the working directory is correct. Skipping.',
                LogColor.SALMON,
            )

    # Plate solve on sequence runs when file count < 2048
    def seq_plate_solve(self, seq_name):
        """Runs the siril command 'seqplatesolve' to plate solve the converted files."""
        # self.siril.cmd("cd", "process")
        args = ["seqplatesolve", seq_name]

        # If origin or D2, need to pass in the focal length, pixel size, and target coordinates
        # if self.chosen_telescope == "Celestron Origin":
        #     args.append(self.target_coords)
        #     focal_len = 335
        #     pixel_size = 2.4
        #     args.append(f"-focal={focal_len}")
        #     args.append(f"-pixelsize={pixel_size}")
        if self.chosen_telescope == "Dwarf 2":
            args.append(self.target_coords)
            focal_len = 100
            pixel_size = 1.45
            args.append(f"-focal={focal_len}")
            args.append(f"-pixelsize={pixel_size}")

        args.extend(
            ["-nocache", "-force", "-disto=ps_distortion", "-order=4", "-radius=25"]
        )
        # args = ["platesolve", seq_name, "-disto=ps_distortion", "-force"]

        try:
            self.siril.cmd(*args)
            self.siril.log(f"Platesolved {seq_name}", LogColor.GREEN)
            return True
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"seqplatesolve failed: {e}", LogColor.RED)
            return True  # TODO: disabling fallback because Siril seems to be throwing a false error

    # Regular registration if plate solve not available - No Mosaics
    def regular_register_seq(self, seq_name, drizzle_amount, pixel_fraction):
        """Registers the sequence using the 'register' command."""
        cmd_args = ["register", seq_name, "-2pass"]
        if self.drizzle_status:
            cmd_args.extend(
                ["-drizzle", f"-scale={drizzle_amount}", f"-pixfrac={pixel_fraction}"]
            )
        self.siril.log(
            "Regular Registration (Global Star Alignment) Done: " + " ".join(cmd_args),
            LogColor.BLUE,
        )

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Data error occurred: {e}", LogColor.RED)

        self.siril.log("Registered Sequence", LogColor.GREEN)

    def seq_bg_extract(self, seq_name):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        try:
            self.siril.cmd("seqsubsky", seq_name, "1", "-samples=10", "-tolerance=2.0")
            self.siril.cmd("cd", ".")  # Refresh current directory
            self.siril.cmd("close")  # Close and reopen to flush cache
            self.siril.cmd("cd", ".")  # Re-establish working directory
            time.sleep(10)  # Wait for Siril to flush cache
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Seq BG Extraction failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Background extracted from Sequence", LogColor.GREEN)

    def seq_apply_reg(
        self,
        seq_name,
        drizzle_amount,
        pixel_fraction,
        filter_roundness,
        filter_fwhm,
        filter_bg,
        filter_star_count,
    ):
        """Apply Existing Registration to the sequence."""
        cmd_args = [
            "seqapplyreg",
            seq_name,
            "-kernel=square",
            "-framing=max",
        ]

        if self.filters_group.isChecked():
            cmd_args.extend(
                [
                    f"-filter-round={filter_roundness}%",
                    f"-filter-wfwhm={filter_fwhm}%",
                    f"-filter-bkg={filter_bg}%",
                    f"-filter-nbstars={filter_star_count}%",
                ]
            )

        if self.drizzle_status:
            cmd_args.extend(
                ["-drizzle", f"-scale={drizzle_amount}", f"-pixfrac={pixel_fraction}"]
            )
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
            if filename.startswith(seq_name) and (
                filename.lower().endswith(self.fits_extension + ".fz")
                or filename.lower().endswith(self.fits_extension)
            ):
                filepath = os.path.join(folder, filename)
                try:
                    with fits.open(filepath) as hdul:
                        # Try to get data from HDU 1 for compressed files, fall back to HDU 0
                        data = None
                        if len(hdul) > 1:
                            data = hdul[1].data
                        else:
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
        ]

        # Check if darks_stacked exists before adding to command
        if use_darks and os.path.exists(
            os.path.join(
                self.current_working_directory,
                "process",
                f"darks_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-dark=darks_stacked")
            cmd_args.append("-cc=dark")

        if use_flats and os.path.exists(
            os.path.join(
                self.current_working_directory,
                "process",
                f"flats_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-flat=flats_stacked")

        cmd_args.extend(["-cfa", "-equalize_cfa"])

        # Calibrate with -debayer if drizle is not set
        self.siril.log(f"Drizzle status: {self.drizzle_status}", LogColor.BLUE)
        if not self.drizzle_status:
            cmd_args.append("-debayer")

        self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Command execution failed: {e}", LogColor.RED)
            self.close_dialog()

        if "eVscope 1" in self.chosen_telescope:
            # crop files for evscope1/equinox1 IMX224
            cmd_args = ["seqcrop", f"pp_{seq_name}", "7 0 1296 976"]

            self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

            try:
                self.siril.cmd(*cmd_args)
            except (s.DataError, s.CommandError, s.SirilError) as e:
                # turn off compression if error (if checked)
                if self.compression_checkbox.isChecked():
                    self.siril.cmd("setcompress", "0")
                self.siril.log(f"Command execution failed: {e}", LogColor.RED)
                self.close_dialog()

    def seq_stack(
        self,
        seq_name,
        feather,
        feather_amount,
        rejection=False,
        output_name=None,
        overlap_norm=False,
        stack_weighted=False,
        weighting_method="Noise",
    ):
        """Stack it all, and feather if it's provided"""
        out = "result" if output_name is None else output_name

        cmd_args = [
            "stack",
            f"{seq_name}",
            " rej 3 3" if rejection else " rej none",
            "-norm=addscale",
            "-output_norm",
            "-overlap_norm" if overlap_norm else "",
            "-rgb_equal",
            "-maximize",
            "-filter-included",
            "-32b",
            f"-out={out}",
        ]

        if stack_weighted:
            # Map weighting method to siril command option
            weighting_map = {
                "Number of Stars": "nbstars",
                "Weighted FWHM": "wfwhm",
                "Noise": "noise",
            }
            weight_option = weighting_map.get(weighting_method, "nbstars")
            cmd_args.append(f"-weight={weight_option}")
            cmd_args.append(f"-feather={feather_amount}")

        self.siril.log(
            f"Running seq_stack with arguments:\n"
            f"seq_name={seq_name}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}",
            LogColor.BLUE,
        )

        self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

        try:
            # Turn off compression for stacking
            self.siril.cmd("setcompress", "0")
            self.siril.cmd(*cmd_args)
            # Turn compression back on after stacking
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "1 -type=rice 16")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Stacking failed: {e}", LogColor.RED)
            self.close_dialog()

        self.siril.log(f"Completed stacking {seq_name}!", LogColor.GREEN)

    def save_image(self, suffix):
        """Saves the image as a FITS file."""

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M")

        # Default filename
        drizzle_str = str(round(self.drizzle_factor, 2)).replace(".", "-")
        file_name = f"result_drizzle-{drizzle_str}x_{current_datetime}{suffix}"

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
        if self.drizzle_status:
            file_name += f"_drizzle-{drizzle_str}x"

        file_name += f"_{current_datetime}{suffix}"

        try:
            self.siril.cmd("setcompress", "0")
            # self.siril.cmd("rotate", "180")
            self.siril.cmd(
                "save",
                f"{file_name}",
            )
            return file_name
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
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

    def image_plate_solve(self):
        """Plate solve the loaded image with the '-force' argument."""
        try:
            self.siril.cmd("platesolve", "-force")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Plate Solve command execution failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Platesolved image", LogColor.GREEN)

    def spcc(
        self,
        oscsensor="ZWO Seestar S30",
        filter="No Filter (Broadband)",
        catalog="localgaia",
        whiteref="Average Spiral Galaxy",
    ):

        recoded_sensor = oscsensor
        """SPCC with oscsensor, filter, catalog, and whiteref."""
        if oscsensor in ["ZWO Seestar S30 Pro"]:
            recoded_sensor = "Sony IMX585"
        elif oscsensor in ["Dwarf 3"]:
            recoded_sensor = "Sony IMX678"
        elif oscsensor in ["Dwarf Mini"]:
            recoded_sensor = "Sony IMX662"
        elif "eVscope 1" in oscsensor:
            recoded_sensor = "Sony IMX224"
        elif "eVscope 2" in oscsensor:
            recoded_sensor = "Sony IMX415"  # very similar to IMX347
        elif "Odyssey" in oscsensor:
            recoded_sensor = "Sony IMX415"
        else:
            recoded_sensor = oscsensor

        args = [
            f"-oscsensor={recoded_sensor}",
            f"-catalog={catalog}",
            f"-whiteref={whiteref}",
        ]

        # Add filter-specific arguments
        filter_args = FILTER_COMMANDS_MAP.get(oscsensor, {}).get(filter)
        if filter_args:
            args.extend(filter_args)
        else:
            # Default to UV/IR Block
            args.append("-oscfilter=UV/IR Block")

        # Double Quote each argument due to potential spaces
        quoted_args = [f'"{arg}"' for arg in args]
        try:
            self.siril.cmd("spcc", *quoted_args)
        except (s.CommandError, s.DataError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"SPCC execution failed: {e}", LogColor.RED)
            self.close_dialog()

        img = self.save_image("_spcc")
        self.siril.log(f"Saved SPCC'd image: {img}", LogColor.GREEN)
        return img

    def stacking_details(self):
        """Logs stacking details like number of frames, rejection method, feathering, and drizzle."""
        try:
            headers = self.siril.get_image_fits_header(return_as="dict")
            object_name = headers.get("OBJECT", "Unknown")
            exposure_time = headers.get("EXPTIME", "Unknown")
            telescope = headers.get("TELESCOP") or headers.get("INSTRUME", "Unknown")
            total_integration = headers.get("LIVETIME", "Unknown")
            num_frames = headers.get("STACKCNT", "Unknown")
            filter = headers.get("FILTER", "Unknown")
            focallen = headers.get("FOCALLEN", "Unknown")
            aperture = headers.get("APERTURE", "Unknown")
            date_obs = headers.get("DATE-OBS", "Unknown")
            date = headers.get("DATE", "Unknown")
            pixel_size = headers.get("XPIXSZ", "Unknown")
            feathering = (
                self.feather_amount_spinbox.value()
                if self.feather_group.isChecked()
                else "Off"
            )
            drizzle = f"{self.drizzle_factor}x" if self.drizzle_status else "Off"

            details_msg = (
                f"Stacking Details:\n"
                f"Object: {object_name}\n"
                f"Telescope: {telescope}\n"
                f"Observation Date: {date_obs}\n"
                f"Processing Date: {date}\n"
                f"Filter: {filter}\n"
                f"Number of Frames: {num_frames}\n"
                f"Exposure Time: {exposure_time}s\n"
                f"Total Integration: {total_integration}s\n"
                f"Focal Length: {round(focallen, 2)}mm\n"
                f"Aperture: {aperture}\n"
                f"Pixel Size: {round(pixel_size, 2)}µm\n"
                f"Feathering: {feathering}\n"
                f"Drizzle: {drizzle}"
            )
            self.siril.log(details_msg, LogColor.BLUE)
        except Exception as e:
            self.siril.log(f"Error retrieving stacking details: {e}", LogColor.SALMON)

    def load_image(self, image_name):
        """Loads the result."""
        try:
            self.siril.cmd("load", image_name)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
            self.siril.log(f"Load image failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log(f"Loaded image: {image_name}", LogColor.GREEN)

    def autostretch(self, do_spcc):
        """Autostretch as a way to preview the final result"""
        try:
            self.siril.cmd("autostretch", *(["-linked"] if do_spcc else []))
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Autostretch command execution failed: {e}", LogColor.RED)

            self.close_dialog()
        self.siril.log(
            "Autostretched image."
            + (" You may want to open the _spcc file instead." if do_spcc else ""),
            LogColor.GREEN,
        )

    def clean_up(self, prefix=None):
        """Cleans up all files in the process directory."""
        if not self.current_working_directory.endswith("process"):
            process_dir = os.path.join(self.current_working_directory, "process")
        else:
            process_dir = self.current_working_directory
        try:
            if not os.path.isdir(process_dir):
                self.siril.log(
                    f"Process directory not found: {process_dir}", LogColor.SALMON
                )
                return

            for f in os.listdir(process_dir):
                # Skip the stacked file
                name, ext = os.path.splitext(f.lower())
                if name in (f"{prefix}_stacked", "result") and ext in (
                    self.fits_extension,
                    self.fits_extension + ".fz",
                ):
                    continue

                # Check if file starts with prefix_ or pp_flats_
                if (
                    f.startswith(prefix)
                    or f.startswith(f"{prefix}_")
                    or f.startswith("pp_flats_")
                ):
                    file_path = os.path.join(process_dir, f)
                    if os.path.isfile(file_path):
                        # Retry loop for safe deletion
                        for i in range(3):
                            try:
                                os.remove(file_path)
                                break
                            except OSError:
                                time.sleep(0.5)
                        else:
                            # If loop completes without break, deletion failed
                            self.siril.log(
                                f"Failed to delete {file_path}", LogColor.SALMON
                            )
        except Exception as e:
            self.siril.log(f"Error during cleanup: {e}", LogColor.SALMON)
        self.siril.log(f"Cleaned up {prefix}", LogColor.BLUE)

    @Slot(str)
    def update_filter_options(self, selected_scope):
        """Update filter options when telescope selection changes"""
        new_options = self.filter_options_map.get(selected_scope, [])
        self.chosen_telescope = selected_scope
        self.siril.log(f"Chosen Telescope: {selected_scope}", LogColor.BLUE)

        # Clear and update filter combo
        self.filter_combo.clear()
        self.filter_combo.addItems(new_options)

        # Set default selection
        if new_options:
            self.filter_combo.setCurrentText(new_options[0])

        # If photometry Gaia is not available, disable SPCC
        # if not self.photometry_gaia_available:
        #     self.spcc_checkbox.setChecked(False)
        #     self.spcc_checkbox.setEnabled(False)
        #     print("diabled")
        # Disable SPCC for Celestron Origin
        if selected_scope == "Celestron Origin":
            self.spcc_checkbox.setChecked(False)
            self.spcc_checkbox.setEnabled(False)
            self.siril.log(
                "SPCC cannot be run on Celestron Origin automatically. It must be done manually.",
                LogColor.SALMON,
            )
        else:
            self.spcc_checkbox.setEnabled(True)
        # Update enabled state based on SPCC checkbox
        self.filter_combo.setEnabled(self.spcc_checkbox.isChecked())

    def show_help(self):
        help_text = (
            f"Author: {AUTHOR} ({WEBSITE}); Youtube: {YOUTUBE}\n"
            "Discord: https://discord.gg/yXKqrawpjr\n"
            "Patreon: https://www.patreon.com/c/naztronomy\n"
            "Buy me a Coffee: https://www.buymeacoffee.com/naztronomy\n\n"
            "Info:\n"
            '1. Must have a "lights" subdirectory inside of the working directory.\n'
            "2. For Calibration frames, you can optionally have one or more of the following types: darks, flats, biases.\n"
            "3. If only one calibration frame is present, it will be treated as a master frame.\n"
            "4. Local Astrometry Gaia catalog is required for mosaics!\n"
            f"5. If you have more than the default {self.max_files_per_batch} files, this script will automatically split them into batches. You can change the batching count from 50 to {self.max_files_per_batch}.\n"
            "6. If batching, intermediary files are cleaned up automatically even if 'clean up files' is unchecked.\n"
            "7. If batching, the frames are automatically feathered during the final stack even if 'feather' is unchecked.\n"
            "8. Drizzle increases processing time. Higher the drizzle the longer it takes.\n"
            "9. If you get an error with feathering, turn it off and try again.\n"
            "10. If the logs show a 'normalization' error, please check the 'black frames bug' checkbox and try again.\n"
            "11. When asking for help, please have the logs handy."
        )

        # Show help in Qt message box
        QMessageBox.information(self, "Help", help_text)
        # self.siril.log(help_text, LogColor.BLUE)
        self.siril_log_long(help_text, LogColor.BLUE)

    def _get_title_font(self):
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        return font

    def _create_gaia_status_section(self):
        gaia_status_section = QGroupBox("Local Gaia Catalog Status")
        gaia_status_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        gaia_status_layout = QHBoxLayout(gaia_status_section)
        gaia_status_layout.setSpacing(15)
        gaia_status_layout.setContentsMargins(10, 12, 10, 10)

        if self.astrometry_gaia_available:
            astrometry_gaia_label = QLabel("✓ Local Astrometry Gaia Available")
            astrometry_gaia_label.setStyleSheet("color: green;")
            astrometry_gaia_label.setToolTip(
                f"Local Astrometry Gaia found at: {self.astrometry_gaia_status}"
            )
        else:
            astrometry_gaia_label = QLabel("✗ Local Astrometry Gaia")
            astrometry_gaia_label.setStyleSheet("color: red;")
            astrometry_gaia_label.setToolTip(
                "Local Astrometry Gaia not available - mosaics will not be generated"
            )
        gaia_status_layout.addWidget(astrometry_gaia_label)

        if self.photometry_gaia_available:
            photometry_gaia_label = QLabel("✓ Local Photometry Gaia Available")
            photometry_gaia_label.setStyleSheet("color: green;")
            photometry_gaia_label.setToolTip(
                f"Local Photometry Gaia found at: {self.photometry_gaia_status}"
            )
        else:
            photometry_gaia_label = QLabel("✗ Local Photometry Gaia")
            photometry_gaia_label.setStyleSheet("color: orange;")
            photometry_gaia_label.setToolTip(
                "Local Photometry Gaia not available, will default to Online Gaia."
            )
        gaia_status_layout.addWidget(photometry_gaia_label)

        return gaia_status_section

    def _create_telescope_section(self):
        title_font = self._get_title_font()
        telescope_section = QGroupBox("Telescope")
        telescope_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        telescope_layout = QGridLayout(telescope_section)
        telescope_layout.setSpacing(8)
        telescope_layout.setContentsMargins(12, 12, 12, 12)
        telescope_layout.setHorizontalSpacing(12)
        telescope_layout.setVerticalSpacing(12)

        telescope_label = QLabel("Telescope:")
        telescope_label.setFont(title_font)
        telescope_label.setToolTip(
            "Select your telescope model to ensure proper color calibration and processing settings."
        )
        telescope_layout.addWidget(telescope_label, 0, 0)

        self.telescope_combo = QComboBox()
        self.telescope_combo.addItems(self.telescope_options)
        self.telescope_combo.setCurrentText("ZWO Seestar S30")
        self.telescope_combo.setToolTip(
            "Select your telescope model to ensure proper color calibration and processing settings."
        )
        telescope_layout.addWidget(self.telescope_combo, 0, 1, 1, 3)

        self.telescope_combo.currentTextChanged.connect(self.update_filter_options)

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

        cleanup_files_label = QLabel("Clean Up Files:")
        cleanup_files_label.setFont(title_font)
        cleanup_tooltip = "Enable this option to delete all intermediary files after they are done processing. This saves space on your hard drive.\nNote: If your session is batched, this option is automatically enabled even if it's unchecked!"
        cleanup_files_label.setToolTip(cleanup_tooltip)
        telescope_layout.addWidget(cleanup_files_label, 2, 0)

        self.cleanup_files_checkbox = QCheckBox("")
        self.cleanup_files_checkbox.setToolTip(cleanup_tooltip)
        telescope_layout.addWidget(self.cleanup_files_checkbox, 2, 1)

        # Use compression checkbox
        compression_label = QLabel("Use Compression:")
        compression_label.setFont(title_font)
        compression_tooltip = "Enable FITS compression to reduce file sizes during processing. Please note that if the script crashes OR you cancel the run, compression may remain on in your Siril settings."
        compression_label.setToolTip(compression_tooltip)
        telescope_layout.addWidget(compression_label, 2, 2)

        self.compression_checkbox = QCheckBox("")
        self.compression_checkbox.setToolTip(compression_tooltip)
        telescope_layout.addWidget(self.compression_checkbox, 2, 3)

        return telescope_section

    def _create_drizzle_group(self):
        title_font = self._get_title_font()
        drizzle_group_tooltip = "Drizzle integration can improve resolution but increases processing time and file size."
        self.drizzle_group = QGroupBox("Drizzle")
        self.drizzle_group.setCheckable(True)
        self.drizzle_group.setChecked(False)
        self.drizzle_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.drizzle_group.setToolTip(drizzle_group_tooltip)

        drizzle_layout = QGridLayout(self.drizzle_group)
        drizzle_layout.setSpacing(8)
        drizzle_layout.setContentsMargins(12, 12, 12, 12)
        drizzle_layout.setHorizontalSpacing(15)
        drizzle_layout.setVerticalSpacing(12)

        drizzle_amount_label_tooltip = "Scale factor for drizzle integration. Values between 1.0 and 3.0 are typical. \nNote: Higher values increase processing time and file size."
        drizzle_amount_label = QLabel("Drizzle amount:")
        drizzle_amount_label.setFont(title_font)
        drizzle_amount_label.setToolTip(drizzle_amount_label_tooltip)
        drizzle_layout.addWidget(drizzle_amount_label, 0, 0)

        self.drizzle_amount_spinbox = QDoubleSpinBox()
        self.drizzle_amount_spinbox.setRange(0.1, 3.0)
        self.drizzle_amount_spinbox.setSingleStep(0.1)
        self.drizzle_amount_spinbox.setValue(UI_DEFAULTS["drizzle_amount"])
        self.drizzle_amount_spinbox.setDecimals(1)
        self.drizzle_amount_spinbox.setMinimumWidth(80)
        self.drizzle_amount_spinbox.setSuffix(" x")
        self.drizzle_amount_spinbox.setToolTip(drizzle_amount_label_tooltip)
        drizzle_layout.addWidget(self.drizzle_amount_spinbox, 0, 1)

        pixel_fraction_label_tooltip = "Controls how much pixels overlap in drizzle integration. Lower values can reduce artifacts but may increase noise."
        pixel_fraction_label = QLabel("Pixel Fraction:")
        pixel_fraction_label.setFont(title_font)
        pixel_fraction_label.setToolTip(pixel_fraction_label_tooltip)
        drizzle_layout.addWidget(pixel_fraction_label, 0, 2)

        self.pixel_fraction_spinbox = QDoubleSpinBox()
        self.pixel_fraction_spinbox.setDecimals(2)
        self.pixel_fraction_spinbox.setRange(0.1, 10.0)
        self.pixel_fraction_spinbox.setSingleStep(0.01)
        self.pixel_fraction_spinbox.setValue(UI_DEFAULTS["pixel_fraction"])
        self.pixel_fraction_spinbox.setMinimumWidth(80)
        self.pixel_fraction_spinbox.setSuffix(" px")
        self.pixel_fraction_spinbox.setToolTip(pixel_fraction_label_tooltip)
        drizzle_layout.addWidget(self.pixel_fraction_spinbox, 0, 3)
        return self.drizzle_group

    def _create_optional_stacking_options_group(self):
        title_font = self._get_title_font()
        stacking_options_group = QGroupBox("Optional Stacking Options")
        stacking_options_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        stacking_options_layout = QVBoxLayout(stacking_options_group)
        stacking_options_layout.setSpacing(10)
        stacking_options_layout.setContentsMargins(12, 12, 12, 12)

        # Stack Weighting Subsection
        stack_weighting_subsection = QGroupBox("Stack Weighting")
        stack_weighting_subsection.setCheckable(True)
        stack_weighting_subsection.setChecked(False)
        stack_weighting_subsection.setStyleSheet("QGroupBox { font-weight: bold; }")
        stack_weighting_subsection_tooltip = (
            "Applies weighting to frames during stacking based on selected criteria."
        )
        stack_weighting_subsection.setToolTip(stack_weighting_subsection_tooltip)

        stack_weighting_layout = QGridLayout(stack_weighting_subsection)
        stack_weighting_layout.setSpacing(8)
        stack_weighting_layout.setContentsMargins(12, 12, 12, 12)
        stack_weighting_layout.setHorizontalSpacing(15)
        stack_weighting_layout.setVerticalSpacing(12)

        weighting_method_label = QLabel("Weighting Method:")
        weighting_method_label.setFont(title_font)
        weighting_method_tooltip = (
            "Select the criteria to use for weighting frames during stacking."
        )
        weighting_method_label.setToolTip(weighting_method_tooltip)
        stack_weighting_layout.addWidget(weighting_method_label, 0, 0)

        self.weighting_method_combo = QComboBox()
        self.weighting_method_combo.addItems(
            ["Noise", "Number of Stars", "Weighted FWHM"]
        )
        self.weighting_method_combo.setEnabled(False)
        self.weighting_method_combo.setToolTip(weighting_method_tooltip)
        stack_weighting_layout.addWidget(self.weighting_method_combo, 0, 1)

        stack_weighting_subsection.toggled.connect(
            self.weighting_method_combo.setEnabled
        )
        stacking_options_layout.addWidget(stack_weighting_subsection)

        # Filters Subsection
        filters_subsection = QGroupBox("Filters (Optional)")
        filters_subsection.setCheckable(True)
        filters_subsection.setChecked(False)
        filters_subsection.setStyleSheet("QGroupBox { font-weight: bold; }")
        filters_subsection_tooltip = (
            "Options for filtering images based on various criteria."
        )
        filters_subsection.setToolTip(filters_subsection_tooltip)

        filters_layout = QGridLayout(filters_subsection)
        filters_layout.setSpacing(8)
        filters_layout.setContentsMargins(12, 12, 12, 12)
        filters_layout.setHorizontalSpacing(15)
        filters_layout.setVerticalSpacing(12)

        # Roundness Filter
        roundness_label_tooltip = "Filters images by star roundness, calculated using the second moments of detected stars. \nA lower percentage keeps only frames with more circular stars. Higher percentages allow more variation in star shapes."
        roundness_label = QLabel("Roundness:")
        roundness_label.setFont(title_font)
        roundness_label.setToolTip(roundness_label_tooltip)
        filters_layout.addWidget(roundness_label, 0, 0)

        self.roundness_spinbox = QDoubleSpinBox()
        self.roundness_spinbox.setRange(1.0, 100.0)
        self.roundness_spinbox.setSingleStep(0.1)
        self.roundness_spinbox.setDecimals(2)
        self.roundness_spinbox.setValue(100.0)
        self.roundness_spinbox.setMinimumWidth(80)
        self.roundness_spinbox.setSuffix(" %")
        self.roundness_spinbox.setToolTip(roundness_label_tooltip)
        filters_layout.addWidget(self.roundness_spinbox, 0, 1)

        # FWHM Filter
        fwhm_label_tooltip = "Filters images by weighted Full Width at Half Maximum (FWHM), calculated using star sharpness. \nA lower percentage keeps only frames with consistent FWHM values. Higher percentages allow more variation."
        fwhm_label = QLabel("FWHM:")
        fwhm_label.setFont(title_font)
        fwhm_label.setToolTip(fwhm_label_tooltip)
        filters_layout.addWidget(fwhm_label, 0, 2)

        self.fwhm_spinbox = QDoubleSpinBox()
        self.fwhm_spinbox.setRange(1.0, 100.0)
        self.fwhm_spinbox.setSingleStep(0.1)
        self.fwhm_spinbox.setDecimals(2)
        self.fwhm_spinbox.setValue(100.0)
        self.fwhm_spinbox.setMinimumWidth(80)
        self.fwhm_spinbox.setSuffix(" %")
        self.fwhm_spinbox.setToolTip(fwhm_label_tooltip)
        filters_layout.addWidget(self.fwhm_spinbox, 0, 3)

        # Background Filter
        bg_filter_label = QLabel("Background:")
        bg_filter_label.setFont(title_font)
        bg_filter_tooltip = "Filter frames by background value. Lower percentages keep frames with lower background levels."
        bg_filter_label.setToolTip(bg_filter_tooltip)
        filters_layout.addWidget(bg_filter_label, 1, 0)

        self.bg_filter_spinbox = QDoubleSpinBox()
        self.bg_filter_spinbox.setRange(1.0, 100.0)
        self.bg_filter_spinbox.setSingleStep(0.1)
        self.bg_filter_spinbox.setDecimals(2)
        self.bg_filter_spinbox.setValue(100.0)
        self.bg_filter_spinbox.setMinimumWidth(80)
        self.bg_filter_spinbox.setSuffix(" %")
        self.bg_filter_spinbox.setToolTip(bg_filter_tooltip)
        filters_layout.addWidget(self.bg_filter_spinbox, 1, 1)

        # Star Count Filter
        star_count_filter_label = QLabel("Star Count:")
        star_count_filter_label.setFont(title_font)
        star_count_filter_tooltip = "Filter frames by star count. Lower percentages keep frames with fewer stars."
        star_count_filter_label.setToolTip(star_count_filter_tooltip)
        filters_layout.addWidget(star_count_filter_label, 1, 2)

        self.star_count_filter_spinbox = QDoubleSpinBox()
        self.star_count_filter_spinbox.setRange(1.0, 100.0)
        self.star_count_filter_spinbox.setSingleStep(0.1)
        self.star_count_filter_spinbox.setDecimals(2)
        self.star_count_filter_spinbox.setValue(100.0)
        self.star_count_filter_spinbox.setMinimumWidth(80)
        self.star_count_filter_spinbox.setSuffix(" %")
        self.star_count_filter_spinbox.setToolTip(star_count_filter_tooltip)
        filters_layout.addWidget(self.star_count_filter_spinbox, 1, 3)

        # Connect the filters group checkbox to enable/disable all filter controls
        filters_subsection.toggled.connect(self.roundness_spinbox.setEnabled)
        filters_subsection.toggled.connect(self.fwhm_spinbox.setEnabled)
        filters_subsection.toggled.connect(self.bg_filter_spinbox.setEnabled)
        filters_subsection.toggled.connect(self.star_count_filter_spinbox.setEnabled)
        stacking_options_layout.addWidget(filters_subsection)

        # Feathering Subsection
        feather_subsection = QGroupBox("Feather")
        feather_subsection.setCheckable(True)
        feather_subsection.setChecked(False)
        feather_subsection.setStyleSheet("QGroupBox { font-weight: bold; }")
        feather_subsection_tooltip = "Blends the edges of stacked frames to reduce edge artifacts in the final image."
        feather_subsection.setToolTip(feather_subsection_tooltip)

        feather_layout = QGridLayout(feather_subsection)
        feather_layout.setSpacing(8)
        feather_layout.setContentsMargins(12, 12, 12, 12)
        feather_layout.setHorizontalSpacing(15)
        feather_layout.setVerticalSpacing(12)

        feather_amount_label_tooltip = "Size of the feathering blend in pixels. Larger values create smoother transitions but may affect more of the image edge."
        feather_amount_label = QLabel("Feather amount:")
        feather_amount_label.setFont(title_font)
        feather_amount_label.setToolTip(feather_amount_label_tooltip)
        feather_layout.addWidget(feather_amount_label, 0, 0)

        self.feather_amount_spinbox = QSpinBox()
        self.feather_amount_spinbox.setRange(5, 2000)
        self.feather_amount_spinbox.setSingleStep(5)
        self.feather_amount_spinbox.setValue(UI_DEFAULTS["feather_amount"])
        self.feather_amount_spinbox.setMinimumWidth(80)
        self.feather_amount_spinbox.setSuffix(" px")
        self.feather_amount_spinbox.setToolTip(feather_amount_label_tooltip)
        feather_layout.addWidget(self.feather_amount_spinbox, 0, 1)
        stacking_options_layout.addWidget(feather_subsection)

        # Store references to subsections for the feather warning in SPCC section
        self.feather_group = feather_subsection
        self.stack_weighting_group = stack_weighting_subsection
        self.filters_group = filters_subsection

        return stacking_options_group

    def _create_preprocessing_section(self):
        title_font = self._get_title_font()
        preprocessing_section = QGroupBox("Optional Preprocessing Steps")
        preprocessing_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        preprocessing_layout = QGridLayout(preprocessing_section)
        preprocessing_layout.setSpacing(8)
        preprocessing_layout.setContentsMargins(12, 12, 12, 12)
        preprocessing_layout.setHorizontalSpacing(15)
        preprocessing_layout.setVerticalSpacing(12)

        # Batch size spinbox
        batch_size_label = QLabel("Max Files per Batch:")
        batch_size_label.setFont(title_font)
        batch_size_tooltip = (
            "Maximum number of files to process in each batch. Windows only. This is ignored on Mac/Linux."
            "This is an advanced option. Only change if you are comfortable with it.\\n"
            "Valid range: 50–2000."
        )
        batch_size_label.setToolTip(batch_size_tooltip)
        preprocessing_layout.addWidget(batch_size_label, 0, 0)

        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setToolTip(batch_size_tooltip)
        self.batch_size_spinbox.setRange(50, self.max_files_per_batch)
        self.batch_size_spinbox.setValue(self.max_files_per_batch)
        self.batch_size_spinbox.setSingleStep(50)
        preprocessing_layout.addWidget(self.batch_size_spinbox, 0, 1)

        # Files found label
        self.files_found_label = QLabel()
        self.files_found_label.setToolTip(
            "Number of Fit(s) files found in the lights directory."
        )
        preprocessing_layout.addWidget(self.files_found_label, 0, 2, 1, 4)

        bg_extract_label = QLabel("Background Extraction:")
        bg_extract_label.setFont(title_font)
        bg_extract_tooltip = "Removes background gradients from your images before stacking. Uses Polynomial value 1 and 10 samples."
        bg_extract_label.setToolTip(bg_extract_tooltip)
        preprocessing_layout.addWidget(bg_extract_label, 1, 0)

        self.bg_extract_checkbox = QCheckBox("")
        self.bg_extract_checkbox.setToolTip(bg_extract_tooltip)
        preprocessing_layout.addWidget(self.bg_extract_checkbox, 1, 1)

        # Add subsections
        drizzle_group = self._create_drizzle_group()
        preprocessing_layout.addWidget(drizzle_group, 2, 0, 1, 6)

        return preprocessing_section

    def _create_spcc_section(self):
        title_font = self._get_title_font()
        self.spcc_section = QGroupBox("Post-Stacking")
        self.spcc_section.setStyleSheet("QGroupBox { font-weight: bold; }")
        spcc_layout = QGridLayout(self.spcc_section)
        spcc_layout.setSpacing(8)
        spcc_layout.setContentsMargins(12, 12, 12, 12)
        spcc_layout.setHorizontalSpacing(12)
        spcc_layout.setVerticalSpacing(12)

        spcc_tooltip = "SPCC uses star colors to calibrate the image colors. Recommended for accurate color reproduction."
        self.spcc_checkbox = QCheckBox(
            "Enable Spectrophotometric Color Calibration (SPCC)"
        )
        self.spcc_checkbox.setToolTip(spcc_tooltip)
        spcc_layout.addWidget(self.spcc_checkbox, 0, 0, 1, 2)

        osc_filter_label = QLabel("OSC Filter:")
        osc_filter_label.setFont(title_font)
        osc_filter_label.setToolTip(
            "Select the filter used during image acquisition for proper color calibration."
        )
        spcc_layout.addWidget(osc_filter_label, 1, 0)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.current_filter_options)
        self.filter_combo.setCurrentText("No Filter (Broadband)")
        self.filter_combo.setEnabled(False)
        self.filter_combo.setToolTip(
            "Select the filter used during image acquisition for proper color calibration."
        )
        spcc_layout.addWidget(self.filter_combo, 1, 1)

        self.spcc_checkbox.toggled.connect(self.filter_combo.setEnabled)

        self.scan_blackframes_checkbox = QCheckBox("Black Frames Bug?")
        self.scan_blackframes_checkbox.setToolTip(
            "Enable this option to automatically scan for black frames in your image sequence ONLY If you see black frames as a result of drizzle."
            "\\nWhen the bug is confirmed fixed, this option and check will be removed."
        )
        spcc_layout.addWidget(self.scan_blackframes_checkbox, 3, 0, 1, 2)

        feather_warning = QLabel(
            "⚠ You enabled feather, this can cause slow processing and memory issues. If you get an error, turn it off and try again.\\nSupport will not be provided for feather-related issues. ⚠"
        )
        feather_warning.setStyleSheet("color: red;")
        feather_warning.setWordWrap(True)
        feather_warning.setVisible(False)
        spcc_layout.addWidget(feather_warning, 4, 0, 1, 2)

        self.feather_group.toggled.connect(feather_warning.setVisible)

        return self.spcc_section

    def _create_buttons_layout(self):
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(12, 10, 12, 12)
        button_layout.setSpacing(8)

        help_button = QPushButton("Help")
        help_button.setMinimumWidth(50)
        help_button.setMinimumHeight(35)
        help_button.setToolTip("Show help information and frequently asked questions")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        save_presets_button = QPushButton("Save Presets")
        save_presets_button.setMinimumWidth(80)
        save_presets_button.setMinimumHeight(35)
        save_presets_button.setToolTip(
            'Save current settings to a "naztronomy_smart_scope_presets.json" file in the presets directory'
        )
        save_presets_button.clicked.connect(self.save_presets)
        button_layout.addWidget(save_presets_button)

        load_presets_button = QPushButton("Load Presets")
        load_presets_button.setMinimumWidth(80)
        load_presets_button.setMinimumHeight(35)
        load_presets_button.setToolTip(
            'Load previously saved presets. If "presets/naztronomy_smart_scope_presets.json" exists, it will load first, otherwise it\'ll prompt you to find a proper .json file.'
        )
        load_presets_button.clicked.connect(self.load_presets)
        button_layout.addWidget(load_presets_button)

        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setMinimumHeight(35)
        close_button.setStyleSheet(
            "QPushButton { background-color: #c70306; color: white; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #fc3437; }"
        )
        close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(close_button)

        button_layout.addSpacing(10)

        self.run_button = QPushButton("Run")
        self.run_button.setMinimumWidth(100)
        self.run_button.setMinimumHeight(35)
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #0078cc; color: white; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #33abff; }"
        )
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)

        return button_layout

    def create_widgets(self):
        """Creates the UI widgets."""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Set default window size (larger by default)
        self.resize(950, 850)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        main_layout.addWidget(scroll_area)

        # Create content widget for scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(12, 8, 12, 12)
        content_layout.setSpacing(10)

        # Title and version
        title_label = QLabel(f"{APP_NAME}")
        title_font = self._get_title_font()
        title_label.setFont(title_font)
        content_layout.addWidget(title_label)

        # Current working directory label
        self.cwd_label = QLabel(self.cwd_label_text)
        content_layout.addWidget(self.cwd_label)

        # Catalog status section
        content_layout.addWidget(self._create_gaia_status_section())

        # Telescope section
        content_layout.addWidget(self._create_telescope_section())

        # Optional Preprocessing Steps
        content_layout.addWidget(self._create_preprocessing_section())

        # Optional Stacking Options
        content_layout.addWidget(self._create_optional_stacking_options_group())

        # SPCC Section
        content_layout.addWidget(self._create_spcc_section())

        # Add stretch to content layout to push buttons down
        content_layout.addStretch()

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(10)  # Make it slim
        # Remove border and make it span full width
        main_layout.addWidget(self.progress_bar)

        # Buttons section
        button_layout = self._create_buttons_layout()

        # Wrap button layout in a widget to easily disable/enable all buttons
        self.buttons_widget = QWidget()
        self.buttons_widget.setLayout(button_layout)
        # Add buttons to bottom of main layout (after scrollable area)
        main_layout.addWidget(self.buttons_widget)

    # def setup_shortcuts(self):
    #     """Setup keyboard shortcuts"""
    #     # Cmd+W on macOS, Ctrl+W on other platforms
    #     close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
    #     close_shortcut.activated.connect(self.close_dialog)

    #     # Escape key as alternative to close
    #     escape_shortcut = QShortcut(QKeySequence.StandardKey.Cancel, self)
    #     escape_shortcut.activated.connect(self.close_dialog)

    #     # Enter/Return key to run
    #     run_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
    #     run_shortcut.activated.connect(self.on_run_clicked)

    #     # Cmd+R on macOS, Ctrl+R on other platforms for run
    #     run_shortcut2 = QShortcut(QKeySequence("Ctrl+R"), self)
    #     run_shortcut2.activated.connect(self.on_run_clicked)

    def on_run_clicked(self):
        """Handle the Run button click"""
        # If currently running, request cancellation
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.requestInterruption()
            self.run_button.setText("Cancelling...")
            self.run_button.setEnabled(False)
            return

        # Disable all buttons EXCEPT Run (which becomes Cancel)

        # Disable inputs
        self.centralWidget().setEnabled(False)
        # Re-enable the buttons widget so we can click Cancel
        self.buttons_widget.setEnabled(True)
        self.buttons_widget.parentWidget().setEnabled(True)

        # disable the scroll area content
        self.findChild(QScrollArea).widget().setEnabled(False)

        # Disable other buttons
        for btn in self.buttons_widget.findChildren(QPushButton):
            if btn != self.run_button:
                btn.setEnabled(False)

        self.run_button.setText("Cancel")

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        # self.progress_bar.setValue(0)

        # Collect parameters
        params = {
            "do_spcc": self.spcc_checkbox.isChecked(),
            "filter": self.filter_combo.currentText(),
            "telescope": self.telescope_combo.currentText(),
            "use_darks": self.darks_checkbox.isChecked(),
            "use_flats": self.flats_checkbox.isChecked(),
            "use_biases": self.biases_checkbox.isChecked(),
            "max_files_per_batch": self.batch_size_spinbox.value(),
            "bg_extract": self.bg_extract_checkbox.isChecked(),
            "drizzle": self.drizzle_group.isChecked(),
            "drizzle_amount": round(self.drizzle_amount_spinbox.value(), 2),
            "pixel_fraction": round(self.pixel_fraction_spinbox.value(), 2),
            "filter_roundness": round(self.roundness_spinbox.value(), 2),
            "filter_fwhm": round(self.fwhm_spinbox.value(), 2),
            "filter_bg": round(self.bg_filter_spinbox.value(), 2),
            "filter_star_count": round(self.star_count_filter_spinbox.value(), 2),
            "stack_weighting": self.stack_weighting_group.isChecked(),
            "weighting_method": self.weighting_method_combo.currentText(),
            "feather": self.feather_group.isChecked(),
            "feather_amount": self.feather_amount_spinbox.value(),
            "clean_up_files": self.cleanup_files_checkbox.isChecked(),
        }

        # Run checks in main thread
        if not self.run_pre_checks():
            # Re-enable if checks fail
            self.findChild(QScrollArea).widget().setEnabled(True)
            for btn in self.buttons_widget.findChildren(QPushButton):
                btn.setEnabled(True)
            self.run_button.setText("Run")
            self.progress_bar.setVisible(False)
            return

        # Start background thread
        self.worker = WorkerThread(self.run_processing_logic, **params)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.error.connect(self.on_processing_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def on_processing_finished(self):
        self.progress_bar.setVisible(False)
        self.findChild(QScrollArea).widget().setEnabled(True)
        for btn in self.buttons_widget.findChildren(QPushButton):
            btn.setEnabled(True)
        self.run_button.setText("Run")
        self.close_dialog()

    def on_processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.findChild(QScrollArea).widget().setEnabled(True)
        for btn in self.buttons_widget.findChildren(QPushButton):
            btn.setEnabled(True)
        self.run_button.setText("Run")
        self.siril.log(f"Error during processing: {error_msg}", LogColor.RED)
        QMessageBox.critical(
            self, "Processing Error", f"An error occurred:\n{error_msg}"
        )

    def closeEvent(self, event):
        """Handle the window close event (clicking the X button)."""
        if hasattr(self, "worker") and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Processing in Progress",
                "The script is currently processing. Please wait for it to finish. To stop the script, click the 'stop' button under the console and stop the python process",
            )
            event.ignore()
        else:
            self.siril.disconnect()
            event.accept()

    def run_pre_checks(self):
        self.siril.log("Starting pre-checks...", LogColor.BLUE)

        if self.fits_files_count == 0:
            QMessageBox.warning(
                self,
                "No FITS Files Found",
                "No FITS files found in the lights directory. Please add files and try again.",
            )
            return False

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
                        self.siril.log(
                            "Cleaned up old sessions directories", LogColor.BLUE
                        )
                    if os.path.exists("process"):
                        shutil.rmtree("process")
                        self.siril.log(
                            "Cleaned up old process directory", LogColor.BLUE
                        )
                    if os.path.exists("final_stack"):
                        shutil.rmtree("final_stack")
                        self.siril.log(
                            "Cleaned up old final_stack directory", LogColor.BLUE
                        )
                except Exception as e:
                    self.siril.log(
                        "Error cleaning up old processing files in one or more of these directories: sessions, process, final_stack.",
                        LogColor.RED,
                    )
                    QMessageBox.warning(
                        self,
                        "Error",
                        "Error cleaning up old processing files in one or more of these directories: sessions, process, final_stack.\nPlease remove them manually and try again.\n\n",
                    )
                    return False
            else:
                self.siril.log(
                    "User chose to preserve old processing files. Stopping script.",
                    LogColor.BLUE,
                )
                return False
        return True

    def run_processing_logic(
        self,
        progress_callback=None,
        check_cancel=None,
        do_spcc: bool = False,
        filter: str = "broadband",
        telescope: str = "ZWO Seestar S30",
        catalog: str = "localgaia",
        use_darks: bool = False,
        use_flats: bool = False,
        use_biases: bool = False,
        max_files_per_batch: float = UI_DEFAULTS["max_files_per_batch"],
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        filter_roundness: float = 100.0,
        filter_fwhm: float = 100.0,
        filter_bg: float = 100.0,
        filter_star_count: float = 100.0,
        stack_weighting: bool = False,
        weighting_method: str = "Noise",
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
        clean_up_files: bool = False,
    ):
        self.siril.log(
            f"Running script version {VERSION} with arguments:\n"
            f"do_spcc={do_spcc}\n"
            f"filter={filter}\n"
            f"telescope={telescope}\n"
            f"catalog={catalog}\n"
            f"use_darks={use_darks}\n"
            f"use_flats={use_flats}\n"
            f"use_biases={use_biases}\n"
            f"batch_size={max_files_per_batch}\n"
            f"bg_extract={bg_extract}\n"
            f"drizzle={drizzle}\n"
            f"drizzle_amount={drizzle_amount}\n"
            f"filter_roundness={filter_roundness}\n"
            f"filter_fwhm={filter_fwhm}\n"
            f"filter_bg={filter_bg}\n"
            f"filter_star_count={filter_star_count}\n"
            f"pixel_fraction={pixel_fraction}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}\n"
            f"stack_weighting={stack_weighting}\n"
            f"weighting_method={weighting_method}\n"
            f"clean_up_files={clean_up_files}\n"
            f"compression={self.compression_checkbox.isChecked()}\n"
            f"black_frames_bug={self.scan_blackframes_checkbox.isChecked()}\n"
            f"build={VERSION}-{BUILD}",
            LogColor.BLUE,
        )
        self.siril.cmd("close")

        def check_interruption():
            if check_cancel and check_cancel():
                raise Exception("Operation cancelled by user.")

        check_interruption()

        self.drizzle_status = drizzle
        self.drizzle_factor = drizzle_amount

        # TODO: Stack calibration frames and copy to the various batch dirs
        if use_biases:
            converted = self.convert_files("biases")
            if converted:
                self.calibration_stack("biases")
            if clean_up_files:
                self.clean_up("biases")
            check_interruption()
        if use_flats:
            converted = self.convert_files("flats")
            if converted:
                self.calibration_stack("flats")
            if clean_up_files:
                self.clean_up("flats")
            check_interruption()
        if use_darks:
            converted = self.convert_files("darks")
            if converted:
                self.calibration_stack("darks")
            if clean_up_files:
                self.clean_up("darks")
            check_interruption()

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

        # only one batch will be run if less than max_files_per_batch OR not windows.
        if num_files <= max_files_per_batch:
            self.siril.log(
                f"{num_files} files found in the lights directory which is less than or equal to {max_files_per_batch} files allowed per batch - no batching needed.",
                LogColor.BLUE,
            )
            check_interruption()
            file_name = self.batch(
                output_name=lights_directory,
                use_darks=use_darks,
                use_flats=use_flats,
                use_biases=use_biases,
                bg_extract=bg_extract,
                drizzle=drizzle,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
                filter_roundness=filter_roundness,
                filter_fwhm=filter_fwhm,
                feather=feather,
                feather_amount=feather_amount,
                stack_weighting=stack_weighting,
                weighting_method=weighting_method,
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
            for i, filename in enumerate(all_files):
                batch_index = i // max_files_per_batch
                batch_dir = f"batch_lights{batch_index + 1}"
                src_path = os.path.join(lights_directory, filename)
                dest_path = os.path.join(batch_dir, filename)

                # try:
                #     # Try creating symlink first
                #     os.symlink(src_path, dest_path)
                # except (OSError, NotImplementedError):
                #     # Fall back to copying if symlink fails
                shutil.copy2(src_path, dest_path)

            # Send each of the new lights dir into batch directory
            for i in range(num_batches):
                check_interruption()
                batch_dir = f"batch_lights{i+1}"
                self.siril.log(f"Processing batch: {batch_dir}", LogColor.BLUE)
                self.batch(
                    output_name=batch_dir,
                    use_darks=use_darks,
                    use_flats=use_flats,
                    use_biases=use_biases,
                    bg_extract=bg_extract,
                    drizzle=drizzle,
                    drizzle_amount=drizzle_amount,
                    pixel_fraction=pixel_fraction,
                    filter_roundness=filter_roundness,
                    filter_fwhm=filter_fwhm,
                    feather=feather,
                    feather_amount=feather_amount,
                    stack_weighting=stack_weighting,
                    weighting_method=weighting_method,
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
            self.seq_plate_solve(seq_name=final_stack_seq_name)
            # turn off drizzle for this
            self.drizzle_status = False
            # Force filters to 3 sigma
            self.seq_apply_reg(
                seq_name=final_stack_seq_name,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
                filter_roundness=100.0,
                filter_fwhm=100.0,
                filter_bg=100.0,
                filter_star_count=100.0,
            )
            self.clean_up(prefix=final_stack_seq_name)
            check_interruption()
            registered_final_stack_seq_name = f"r_{final_stack_seq_name}"
            # final stack needs feathering and amount
            self.drizzle_status = drizzle  # Turn drizzle back to selected option
            self.seq_stack(
                seq_name=registered_final_stack_seq_name,
                feather=True,
                rejection=False,
                feather_amount=100,
                output_name="final_result",
                overlap_norm=True,
                stack_weighted=stack_weighting,
                weighting_method=weighting_method,
            )
            self.load_image(image_name="final_result")

            # cleanup final_stack directory
            # shutil.rmtree(final_stack_seq_name, ignore_errors=True)
            if clean_up_files:
                self.clean_up(prefix=registered_final_stack_seq_name)

            # Go back to working dir
            self.siril.cmd("cd", "../")

            # Save og image in WD - might have drizzle factor in name
            file_name = self.save_image("_batched")
            self.load_image(image_name=file_name)

        # Spcc as a last step
        if do_spcc:
            img = self.spcc(
                oscsensor=telescope,
                filter=filter,
                catalog=catalog,
                whiteref="Average Spiral Galaxy",
            )

            # self.autostretch(do_spcc=do_spcc)
            if drizzle:
                img = os.path.basename(img) + self.fits_extension
            else:
                img = os.path.basename(img)
            self.load_image(
                image_name=os.path.basename(img)
            )  # Load either og or spcc image

        # Get some stacking deets
        self.stacking_details()
        # self.clean_up()

        self.siril.cmd("setcompress", "0")  # Disable compression after processing

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
        # self.close_dialog() # Now handled by on_finished

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
        self.siril.log(f"Extracting Coordinates from file: {first_file}", LogColor.BLUE)
        file_path = os.path.join(process_dir, first_file)

        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                ra = header.get("RA")
                dec = header.get("DEC")

                if ra is None:
                    ra = header.get("FOVRA")
                    dec = header.get("FOVDEC")

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

    def batch(
        self,
        output_name: str,
        use_darks: bool = False,
        use_flats: bool = False,
        use_biases: bool = False,
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        filter_roundness: float = 100.0,
        filter_fwhm: float = 100.0,
        filter_bg: float = 100.0,
        filter_star_count: float = 100.0,
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
        stack_weighting: bool = False,
        weighting_method: str = "Noise",
        clean_up_files: bool = False,
    ):
        # If we're batching, force cleanup files so we don't collide with existing files
        self.siril.cmd("close")

        try:
            if self.compression_checkbox.isChecked():
                self.siril.log("Enabling FITS compression (Rice 16-bit)", LogColor.BLUE)
                self.siril.cmd("setcompress", "1 -type=rice 16")
            else:
                self.siril.log(
                    "Compression not set, disabling in case it's turned on from a previous run/crash",
                    LogColor.BLUE,
                )
                self.siril.cmd("setcompress", "0")
        except s.CommandError:
            # turn off compression if error (if checked)
            if self.compression_checkbox.isChecked():
                self.siril.cmd("setcompress", "0")
                self.siril.log(
                    "Disabling compression due to command error", LogColor.SALMON
                )

        if output_name.startswith("batch_lights"):
            clean_up_files = True

        self.drizzle_status = drizzle
        self.drizzle_factor = drizzle_amount

        if self.chosen_telescope.startswith("Unistellar"):
            self.fixUnistellarHeaders(dir_name=output_name)

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
            if "eVscope 1" in self.chosen_telescope:
                seq_name = "cropped_" + seq_name

        if bg_extract:
            self.seq_bg_extract(seq_name=seq_name)
            if clean_up_files:
                self.clean_up(
                    prefix=seq_name
                )  # Remove "pp_lights_" or just "lights_" if not flat calibrated
            seq_name = "bkg_" + seq_name

        if self.chosen_telescope in ["Celestron Origin", "Dwarf 2"]:
            self.extract_coords_from_fits(prefix=seq_name)

        # Only do plate solve if local gaia is available!
        if not self.astrometry_gaia_available:
            self.siril.log(
                "Local Gaia catalogue not available, skipping plate solving. Mosaics will NOT be automatically created.",
                LogColor.SALMON,
            )
            self.regular_register_seq(
                seq_name=seq_name,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
            )
        else:
            individual_plate_solve_status = self.seq_plate_solve(seq_name=seq_name)
            if not individual_plate_solve_status:
                self.siril.log(
                    "Plate solving failed, falling back to regular registration.",
                    LogColor.SALMON,
                )
                self.regular_register_seq(
                    seq_name=seq_name,
                    drizzle_amount=drizzle_amount,
                    pixel_fraction=pixel_fraction,
                )

        # seq_name stays the same after plate solve
        self.seq_apply_reg(
            seq_name=seq_name,
            drizzle_amount=drizzle_amount,
            pixel_fraction=pixel_fraction,
            filter_roundness=filter_roundness,
            filter_fwhm=filter_fwhm,
            filter_bg=filter_bg,
            filter_star_count=filter_star_count,
        )
        if clean_up_files:
            self.clean_up(
                prefix=seq_name
            )  # Clean up bkg_ files or pp_ if flat calibrated, otherwise lights_
        seq_name = f"r_{seq_name}"

        try:
            if drizzle:
                if self.scan_blackframes_checkbox.isChecked():
                    self.scan_black_frames(seq_name=seq_name)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(
                f"Data error occurred during black frame scan: {e}", LogColor.RED
            )

        try:
            self.seq_stack(
                seq_name=seq_name,
                feather=feather,
                feather_amount=feather_amount,
                rejection=True,
                output_name=output_name,
                overlap_norm=False,
                stack_weighted=stack_weighting,
                weighting_method=weighting_method,
            )
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Error occurred during stacking: {e}", LogColor.RED)
            if feather:
                QMessageBox.warning(
                    self,
                    "Stacking Error",
                    "There was an error during the stacking process which could have been caused by feathering. Please uncheck the feather option and try again.",
                )
            return None

        if clean_up_files:
            self.clean_up(prefix=seq_name)  # clean up r_ files
            try:
                shutil.rmtree(os.path.join(self.siril.get_siril_wd(), "cache"))
                shutil.rmtree(os.path.join(self.siril.get_siril_wd(), "drizztmp"))
            except Exception as e:
                self.siril.log(
                    f"Error cleaning up temporary files, continuing with the script: {e}",
                    LogColor.SALMON,
                )

        # Load the result (e.g. batch_lights_001.fits)
        self.load_image(image_name=output_name)

        # Go back to working dir
        self.siril.cmd("cd", "../")

        # Save og image in WD - might have drizzle factor in name
        if output_name.startswith("batch_lights"):
            out = output_name
        else:
            out = "og"
        file_name = self.save_image(f"_{out}")

        return file_name

    # Save and Load Presets code
    def save_presets(self):
        """Save current UI settings to a JSON file in the working directory."""
        presets = {
            "telescope": self.telescope_combo.currentText(),
            "filter": self.filter_combo.currentText(),
            # "catalog": self.catalog_combo.currentText(),
            "darks": self.darks_checkbox.isChecked(),
            "flats": self.flats_checkbox.isChecked(),
            "biases": self.biases_checkbox.isChecked(),
            "cleanup": self.cleanup_files_checkbox.isChecked(),
            "batch_size": self.batch_size_spinbox.value(),
            "bg_extract": self.bg_extract_checkbox.isChecked(),
            "drizzle": self.drizzle_group.isChecked(),
            "drizzle_amount": self.drizzle_amount_spinbox.value(),
            "pixel_fraction": self.pixel_fraction_spinbox.value(),
            "filters": self.filters_group.isChecked(),
            "roundness": self.roundness_spinbox.value(),
            "fwhm": self.fwhm_spinbox.value(),
            "star_count_filter": self.star_count_filter_spinbox.value(),
            "bg_filter": self.bg_filter_spinbox.value(),
            "feather": self.feather_group.isChecked(),
            "feather_amount": self.feather_amount_spinbox.value(),
            "stack_weighting": self.stack_weighting_group.isChecked(),
            "weighting_method": self.weighting_method_combo.currentText(),
            "spcc": self.spcc_checkbox.isChecked(),
            "compression": self.compression_checkbox.isChecked(),
        }

        presets_dir = os.path.join(self.current_working_directory, "presets")
        os.makedirs(presets_dir, exist_ok=True)
        presets_file = os.path.join(presets_dir, "naztronomy_smart_scope_presets.json")

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
                "naztronomy_smart_scope_presets.json",
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
            self.telescope_combo.setCurrentText(
                presets.get("telescope", "ZWO Seestar S50")
            )
            self.filter_combo.setCurrentText(
                presets.get("filter", "No Filter (Broadband)")
            )
            # self.catalog_combo.setCurrentText(presets.get("catalog", "localgaia"))
            self.darks_checkbox.setChecked(presets.get("darks", False))
            self.flats_checkbox.setChecked(presets.get("flats", False))
            self.biases_checkbox.setChecked(presets.get("biases", False))
            self.cleanup_files_checkbox.setChecked(presets.get("cleanup", False))
            self.batch_size_spinbox.setValue(
                presets.get("batch_size", self.max_files_per_batch)
            )
            self.bg_extract_checkbox.setChecked(presets.get("bg_extract", False))
            self.drizzle_group.setChecked(presets.get("drizzle", False))
            self.drizzle_amount_spinbox.setValue(
                presets.get("drizzle_amount", UI_DEFAULTS["drizzle_amount"])
            )
            self.pixel_fraction_spinbox.setValue(
                presets.get("pixel_fraction", UI_DEFAULTS["pixel_fraction"])
            )
            self.filters_group.setChecked(presets.get("filters", False))
            self.roundness_spinbox.setValue(presets.get("roundness", 3.0))
            self.fwhm_spinbox.setValue(presets.get("fwhm", 3.0))
            self.star_count_filter_spinbox.setValue(
                presets.get("star_count_filter", 100.0)
            )
            self.bg_filter_spinbox.setValue(presets.get("bg_filter", 100.0))
            self.feather_group.setChecked(presets.get("feather", False))
            self.feather_amount_spinbox.setValue(
                presets.get("feather_amount", UI_DEFAULTS["feather_amount"])
            )
            self.stack_weighting_group.setChecked(presets.get("stack_weighting", False))
            self.weighting_method_combo.setCurrentText(
                presets.get("weighting_method", "Noise")
            )
            self.spcc_checkbox.setChecked(presets.get("spcc", False))
            self.compression_checkbox.setChecked(presets.get("compression", False))

            self.siril.log(f"Loaded presets from {presets_file}", LogColor.GREEN)
        except Exception as e:
            self.siril.log(f"Failed to load presets: {e}", LogColor.RED)


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
