"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Naztronomy - OSC Image Preprocessing script
Version: 1.1.0
=====================================

The author of this script is Nazmus Nasir (Naztronomy) and can be reached at:
https://www.Naztronomy.com or https://www.YouTube.com/Naztronomy
Join discord for support and discussion: https://discord.gg/yXKqrawpjr
Support me on Patreon: https://www.patreon.com/c/naztronomy
Support me on Buy me a Coffee: https://www.buymeacoffee.com/naztronomy

This script is designed to process OSC images only at this time. Although mono images will process, they will be
incorrectly debayered and be saved as an RGB image.

If your images have the correct headers (RA/DEC coordinates, focal length, pixel size, etc.), this script can automatically
plate solve and stitch mosaics. If you are using data without the correct headers, it will do a star alignment on a reference frame (.e.g no mosaics).

This script can be run from any directory but recommended to create a blank directory.

All images are currently copied before processed so it can take up some disk space. This is to mitigate systems that don't allow symlinks. This also
allows you to choose files from any folder and drive and they will all be consolidated into a single location.

"""

"""
CHANGELOG:

1.1.0 - pyqt6 support
      - Save/Load presets
1.0.0 - initial release
      - Supports both Mosaics and star alignment for imags without proper headers
      - Cleans up all intermediate files BUT keeps all preprocessed lights so they can be combined later
"""


from operator import index
from pathlib import Path
import shutil
import sirilpy as s

s.ensure_installed("PyQt6", "numpy", "astropy")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFrame,
    QListWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTabWidget,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
)
from PyQt6.QtGui import QFont, QShortcut, QKeySequence
from datetime import datetime
import time
import os
import sys
import json
from sirilpy import LogColor, NoImageError
from astropy.io import fits
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


APP_NAME = "Naztronomy - OSC Image Preprocessor"
VERSION = "1.1.0"
AUTHOR = "Nazmus Nasir"
WEBSITE = "Naztronomy.com"
YOUTUBE = "YouTube.com/Naztronomy"


UI_DEFAULTS = {
    "feather_amount": 20,
    "filter_round": 3.0,
    "filter_wfwhm": 3.0,
    "drizzle_amount": 1.0,
    "pixel_fraction": 1.0,
    "max_files_per_batch": 2000,
}
FRAME_TYPES = ("lights", "darks", "flats", "biases")


@dataclass
class Session:
    lights: List[Path] = field(default_factory=list)
    darks: List[Path] = field(default_factory=list)
    flats: List[Path] = field(default_factory=list)
    biases: List[Path] = field(default_factory=list)

    def add_files(self, image_type: str, file_paths: List[Path]):
        if not hasattr(self, image_type):
            raise ValueError(f"Unknown frame type: {image_type}")
        getattr(self, image_type).extend(file_paths)

    def get_file_lists(self) -> Dict[str, List[Path]]:
        return {
            "lights": self.lights,
            "darks": self.darks,
            "flats": self.flats,
            "biases": self.biases,
        }

    def get_files_by_type(self, image_type: str) -> List[Path]:
        if not hasattr(self, image_type):
            raise ValueError(f"Unknown frame type: {image_type}")
        return getattr(self, image_type)

    def get_file_count(self) -> Dict[str, int]:
        return {
            "lights": len(self.lights),
            "darks": len(self.darks),
            "flats": len(self.flats),
            "biases": len(self.biases),
        }

    def __str__(self):
        counts = self.get_file_count()
        return f"Session(L: {counts['lights']}, D: {counts['darks']}, F: {counts['flats']}, B: {counts['biases']})"

    def reset(self):
        self.lights.clear()
        self.darks.clear()
        self.flats.clear()
        self.biases.clear()


class PreprocessingInterface(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        self.initialization_successful = False

        self.siril = s.SirilInterface()

        # if drizzle is off, images will be debayered on convert
        self.drizzle_status = False
        self.drizzle_factor = 0

        self.target_coords = None

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

        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label = self.current_working_directory

        # Assigns collected_lights directory to store all pp_lights files
        self.collected_lights_dir = os.path.join(
            self.current_working_directory, "collected_lights"
        )

        # Sessions
        self.sessions = self.create_sessions(1)  # Start with one session
        self.chosen_session = self.sessions[0]

        self.session_dropdown = QComboBox()
        self.update_dropdown()  # Fill it with sessions
        self.session_dropdown.setCurrentIndex(0)
        self.session_dropdown.currentIndexChanged.connect(self.on_session_selected)

        # self.current_session = "Session 1"  # optional, just for logging/debug
        # self.current_session = tk.StringVar(value=f"Session {len(self.sessions)}")

        # End Session
        self.create_widgets()
        self.initialization_successful = True  # Flag to track successful initialization

    # Start session methods
    def create_sessions(self, n_sessions: int) -> list[Session]:
        """
        Create a list of Sessions of length n_sessions.

        Args:
            n_sessions: The number of sessions to create.

        Returns:
            A list of Session objects.
        """

        return [Session() for _ in range(n_sessions)]

    def get_session_count(self) -> int:
        """
        Return the number of sessions.

        Returns:
            int: The number of sessions.
        """

        return len(self.sessions)

    def get_session_by_index(self, index: int) -> Session:
        """
        Return the session at the given index.

        Args:
            index: The index of the session to return.

        Returns:
            Session: The session at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.sessions):
            return self.sessions[index]
        else:
            raise IndexError("Session index out of range.")

    def get_all_sessions(self) -> List[Session]:
        """
        Return a copy of the list of all sessions.

        Returns:
            List[Session]: A copy of the list of all sessions.
        """
        return self.sessions.copy()

    def clear_all_sessions(self):
        """
        Clear all sessions by resetting each session.

        Resets the lights, darks, flats, and biases of each session to empty lists.

        Returns:
            List[Session]: The list of sessions after being cleared.
        """

        for session in self.sessions:
            session.reset()
        return self.sessions

    def remove_session_by_index(self, index: int) -> List[Session]:
        """
        Remove the session at the given index from the list of sessions.

        Args:
            index: The index of the session to remove.

        Returns:
            List[Session]: The list of sessions after the session at the given index has been removed.

        Raises:
            IndexError: If the index is out of range.
        """

        if 0 <= index < len(self.sessions):
            return self.sessions[:index] + self.sessions[index + 1 :]
        else:
            raise IndexError("Session index out of range.")

    def add_session(self, session: Session) -> List[Session]:
        """
        Add a session to the list of sessions.

        Args:
            session: The session to add to the list of sessions.

        Returns:
            List[Session]: The list of sessions after adding the given session.
        """
        self.sessions.append(session)
        return self.sessions

    def update_session(self, index: int, session: Session) -> List[Session]:
        """
        Update the session at the given index in the list of sessions.

        Args:
            index: The index of the session to update.
            session: The new session to replace the one at the given index.

        Returns:
            List[Session]: The list of sessions after the session at the given index has been updated.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.sessions):
            self.sessions[index] = session
            return self.sessions
        else:
            raise IndexError("Session index out of range.")

    def add_files_to_session(
        self, session: Session, file_type: str, file_paths: List[Path]
    ) -> None:
        if file_type not in FRAME_TYPES:
            raise ValueError(f"Unknown frame type: {file_type}")
        session.add_files(file_type, file_paths)

    # Session UI methods
    def on_session_selected(self, index: int):
        if index < 0 or index >= len(self.sessions):
            return
        # Only update if actually changing sessions
        if self.chosen_session != self.sessions[index]:
            self.chosen_session = self.get_session_by_index(index)
            self.current_session = f"Session {index+1}"
            self.refresh_file_list()

    def add_dropdown_session(self):
        self.add_session(Session())
        self.update_dropdown()
        new_index = len(self.sessions) - 1
        self.session_dropdown.setCurrentIndex(new_index)  # selects new session
        self.chosen_session = self.sessions[new_index]
        self.current_session = f"Session {new_index+1}"
        self.refresh_file_list()

    def remove_session(self):
        if len(self.sessions) <= 1:
            self.siril.log("Cannot remove the last session.", LogColor.BLUE)
            return

        current_index = self.session_dropdown.currentIndex()
        if 0 <= current_index < len(self.sessions):
            self.sessions.pop(current_index)

        self.update_dropdown()
        self.session_dropdown.setCurrentIndex(0)
        self.chosen_session = self.sessions[0]
        self.current_session = "Session 1"
        self.refresh_file_list()

    def update_dropdown(self):
        session_names = [f"Session {i+1}" for i in range(len(self.sessions))]
        self.session_dropdown.clear()  # remove old items
        self.session_dropdown.addItems(session_names)  # add new items

    def load_files(self, filetype: str):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setWindowTitle(f"Select {filetype} Files")

        if sys.platform.startswith("linux"):
            file_dialog.setDirectory(self.siril.get_siril_wd())

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_paths = file_dialog.selectedFiles()
            if not file_paths:
                return

            paths = list(map(Path, file_paths))

            match filetype.lower():
                case "lights":
                    self.chosen_session.lights.extend(paths)
                case "darks":
                    self.chosen_session.darks.extend(paths)
                case "flats":
                    self.chosen_session.flats.extend(paths)
                case "biases":
                    self.chosen_session.biases.extend(paths)

            self.siril.log(
                f"> Added {len(paths)} {filetype} files to {self.session_dropdown.currentText()}",
                LogColor.BLUE,
            )

            self.refresh_file_list()

    def copy_session_files(self, session: Session, session_name: str):
        """Copies all files from the session to the specified destination directory.
        Attempts to create symlinks first, falls back to copying if not supported."""
        destination = Path("sessions")
        if not destination.exists():
            os.mkdir(destination)
        session_dir = destination / session_name
        if not session_dir.exists():
            os.mkdir(session_dir)

        file_counts = session.get_file_count()
        for image_type in FRAME_TYPES:
            if file_counts.get(image_type, 0) > 0:
                type_dir = session_dir / image_type
                if not type_dir.exists():
                    os.mkdir(type_dir)
                files = session.get_files_by_type(image_type)
                for file in files:
                    dest_path = session_dir / image_type / file.name

                    try:
                        # Convert to absolute paths for reliable symlinks
                        src_abs = Path(file).resolve()
                        dest_abs = dest_path.resolve()

                        # Attempt to create symlink
                        os.symlink(src_abs, dest_abs)
                        self.siril.log(
                            f"Symlinked {file} to {dest_path}", LogColor.BLUE
                        )

                    except (OSError, NotImplementedError):
                        # Fall back to copying if symlink fails
                        # OSError covers permission issues and unsupported filesystems
                        # NotImplementedError covers platforms that don't support symlinks
                        shutil.copy(file, dest_path)
                        self.siril.log(f"Copied {file} to {dest_path}", LogColor.BLUE)
            else:
                self.siril.log(
                    f"Skipping {image_type}: no files found", LogColor.SALMON
                )

    def refresh_file_list(self):
        self.file_listbox.clear()  # clear QListWidget instead of delete()
        self.siril.log(f"Switched to session {self.chosen_session}", LogColor.BLUE)

        if self.chosen_session:
            for file_type in FRAME_TYPES:
                files = self.chosen_session.get_files_by_type(file_type)
                if files:
                    for index, file in enumerate(files):
                        self.file_listbox.addItem(
                            f"{index + 1:>4}. {file_type.capitalize():^20}  {str(file.resolve())}"
                        )

    # Debug code
    def show_all_sessions(self):
        for session in self.sessions:
            for file_type in FRAME_TYPES:
                files = session.get_files_by_type(file_type)
                if files:
                    self.siril.log(f"--- {file_type.upper()} ---", LogColor.BLUE)
                    for index, file in enumerate(files):
                        print(
                            f"{index + 1:>4}. {file_type.capitalize():^20}  {str(file.resolve())}"
                        )

    def remove_selected_files(self):
        selected_items = self.file_listbox.selectedItems()
        if not selected_items:
            return

        msg = f"Are you sure you want to delete {len(selected_items)} files? (Note: This will only remove them from the session, not delete them from disk.)"
        reply = QMessageBox.question(
            self,
            "Delete Selected Files?",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Build flat list of all files with type tracking
            all_files = (
                [("lights", f) for f in self.chosen_session.lights]
                + [("darks", f) for f in self.chosen_session.darks]
                + [("flats", f) for f in self.chosen_session.flats]
                + [("biases", f) for f in self.chosen_session.biases]
            )

            for item in selected_items:
                row = self.file_listbox.row(item)  # Get the row index
                filetype, path = all_files[row]
                getattr(self.chosen_session, filetype).remove(path)

            self.refresh_file_list()

    def reset_everything(self):
        msg = "Are you sure you want to reset all sessions? This will delete all file lists and reset the session count to 1."
        reply = QMessageBox.question(
            self,
            "Reset all sessions?",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            for session in self.sessions:
                session.reset()
            self.sessions = [Session()]  # reset to one new session
            self.update_dropdown()
            self.session_dropdown.setCurrentIndex(0)
            self.chosen_session = self.sessions[0]
            self.current_session = "Session 1"
            self.refresh_file_list()

    # end session methods

    # Start Siril processing methods
    # image_type: lights, darks, biases, flats
    def convert_files(self, image_type):
        directory = os.path.join(self.current_working_directory, image_type)
        self.siril.log(f"Converting files in {directory}", LogColor.BLUE)
        if os.path.isdir(directory):
            self.siril.cmd("cd", image_type)
            file_count = len(
                [
                    name
                    for name in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, name))
                ]
            )
            if file_count == 0:
                self.siril.log(
                    f"No files found in {image_type} directory. Skipping conversion.",
                    LogColor.SALMON,
                )
                return
            else:
                try:
                    args = ["convert", image_type, "-out=../process"]
                    if "lights" in image_type.lower():
                        if not self.drizzle_status:
                            args.append("-debayer")
                    else:
                        if not self.drizzle_status:
                            # flats, darks, bias: only debayer if drizzle is not set
                            args.append("-debayer")

                    self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                    self.siril.cmd(*args)
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(f"File conversion failed: {e}", LogColor.RED)
                    self.close_dialog()

                self.siril.cmd("cd", "../process")
                self.siril.log(
                    f"Converted {file_count} {image_type} files for processing!",
                    LogColor.GREEN,
                )
        else:
            self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            raise NoImageError(
                (
                    f'No directory named "{image_type}" at this location. Make sure the working directory is correct.'
                )
            )

    # Plate solve on sequence runs when file count < 2048
    def seq_plate_solve(self, seq_name):
        """Runs the siril command 'seqplatesolve' to plate solve the converted files."""
        # self.siril.cmd("cd", "process")
        args = ["seqplatesolve", seq_name]

        args.extend(["-nocache", "-force", "-disto=ps_distortion"])

        try:
            self.siril.cmd(*args)
            self.siril.log(f"Platesolved {seq_name}", LogColor.GREEN)
            return True
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(
                f"seqplatesolve failed, going to try regular registration: {e}",
                LogColor.SALMON,
            )
            return False
            # self.siril.error_messagebox(f"seqplatesolve failed: {e}")
            # self.close_dialog()

    def seq_bg_extract(self, seq_name):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        try:
            self.siril.cmd("seqsubsky", seq_name, "1", "-samples=10")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Seq BG Extraction failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Background extracted from Sequence", LogColor.GREEN)

    def seq_apply_reg(
        self, seq_name, drizzle_amount, pixel_fraction, filter_wfwhm=3, filter_round=3
    ):
        """Apply Existing Registration to the sequence."""
        cmd_args = [
            "seqapplyreg",
            seq_name,
            f"-filter-round={filter_round}k",
            f"-filter-wfwhm={filter_wfwhm}k",
            "-kernel=square",
            "-framing=max",
        ]
        if self.drizzle_status:
            cmd_args.extend(
                ["-drizzle", f"-scale={drizzle_amount}", f"-pixfrac={pixel_fraction}"]
            )
        self.siril.log("Command arguments: " + " ".join(cmd_args), LogColor.BLUE)

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Data error occurred: {e}", LogColor.RED)

        self.siril.log(
            f"Applied existing registration to seq {seq_name}", LogColor.GREEN
        )

    def regular_register_seq(self, seq_name, drizzle_amount, pixel_fraction):
        """Registers the sequence using the 'register' command."""
        cmd_args = ["register", seq_name, "-2pass"]
        if self.drizzle_status:
            cmd_args.extend(
                ["-drizzle", f"-scale={drizzle_amount}", f"-pixfrac={pixel_fraction}"]
            )
        self.siril.log(
            "Regular Registration Done: " + " ".join(cmd_args), LogColor.BLUE
        )

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
        if seq_name == "flats":
            if os.path.exists(
                os.path.join(
                    self.current_working_directory,
                    f"process/biases_stacked{self.fits_extension}",
                )
            ):
                # Saves as pp_flats
                self.siril.cmd("calibrate", "flats", "-bias=biases_stacked")
                self.siril.cmd(
                    "stack", "pp_flats rej 3 3", "-norm=mul", f"-out={seq_name}_stacked"
                )
                self.siril.cmd("cd", "..")
                return
            else:
                self.siril.cmd(
                    "stack",
                    f"{seq_name} rej 3 3",
                    "-norm=mul",
                    f"-out={seq_name}_stacked",
                )
                self.siril.cmd("cd", "..")
                return
        else:
            # Don't run code below for flats
            # biases and darks
            cmd_args = [
                "stack",
                f"{seq_name} rej 3 3 -nonorm",
                f"-out={seq_name}_stacked",
            ]
            self.siril.log(f"Running command: {' '.join(cmd_args)}", LogColor.BLUE)

            try:
                self.siril.cmd(*cmd_args)
            except (s.DataError, s.CommandError, s.SirilError) as e:
                self.siril.log(f"Command execution failed: {e}", LogColor.RED)
                self.close_dialog()

        self.siril.log(f"Completed stacking {seq_name}!", LogColor.GREEN)
        self.siril.cmd("cd", "..")

    def calibrate_lights(self, seq_name, use_darks=False, use_flats=False):
        # TODO: prefix for each session
        cmd_args = [
            "calibrate",
            f"{seq_name}",
        ]
        if os.path.exists(
            os.path.join(
                self.current_working_directory,
                f"process/darks_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-dark=darks_stacked")
        if os.path.exists(
            os.path.join(
                self.current_working_directory,
                f"process/flats_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-flat=flats_stacked")
        cmd_args.extend(["-cfa", "-equalize_cfa"])
        # cmd_args = [
        #     "calibrate",
        #     f"{seq_name}",
        #     "-dark=darks_stacked" if use_darks else "",
        #     "-flat=flats_stacked" if use_flats else "",
        #     "-cfa -equalize_cfa",
        # ]

        try:
            self.siril.cmd(*cmd_args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Command execution failed: {e}", LogColor.RED)
            self.close_dialog()

    def seq_stack(
        self, seq_name, feather, feather_amount, rejection=False, output_name=None
    ):
        """Stack it all, and feather if it's provided"""
        out = "result" if output_name is None else output_name

        cmd_args = [
            "stack",
            f"{seq_name}",
            " rej 3 3" if rejection else " rej none",
            "-norm=addscale",
            "-output_norm",
            "-rgb_equal",
            "-maximize",
            "-filter-included",
            "-weight=wfwhm",
            f"-out={out}",
        ]
        if feather:
            cmd_args.append(f"-feather={feather_amount}")

        self.siril.log(
            f"Running seq_stack with arguments:\n"
            f"seq_name={seq_name}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}\n"
            f"output_name={out}",
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
        drizzle_str = str(self.drizzle_factor).replace(".", "-")
        file_name = f"result__drizzle-{drizzle_str}x__{current_datetime}{suffix}"

        # Get header info from loaded image for filename
        current_fits_headers = self.siril.get_image_fits_header(return_as="dict")

        object_name = current_fits_headers.get("OBJECT", "Unknown").replace(" ", "_")
        exptime = int(current_fits_headers.get("EXPTIME", 0))
        livetime = int(current_fits_headers.get("LIVETIME", 0))
        stack_count = int(current_fits_headers.get("STACKCNT", 0))

        file_name = f"{object_name}_{stack_count:03d}x{exptime}sec_{livetime}s_"  # {date_obs_str}"
        if self.drizzle_status:
            file_name += f"_drizzle-{drizzle_str}x_"

        file_name += f"{current_datetime}{suffix}"

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
            "1. Recommended to use a blank working directory to have a clean setup.\n"
            "2. You can run this with or without calibration frames.\n"
            f"3. You can have as many sessions as you'd like. Each individual session currently has a limit of 2048 files on Windows machines.\n"
            "4. All preprocessed lights (pp_lights) are saved in the collected_lights directory and are not removed.\n"
            "5. This script uses symbolic links if available, otherwise it makes a copy of all of your images so that the originals are not modified.\n"
            "6. Drizzle increases processing time. Higher the drizzle the longer it takes.\n"
            "7. When asking for help, please have the logs handy."
        )
        QMessageBox.information(self, "Help", help_text)
        self.siril.log(help_text, LogColor.BLUE)

    def create_widgets(self):
        """Creates the UI widgets using PyQt6."""

        # Main layout
        main_widget = QWidget()
        self.setMinimumSize(700, 600)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 10, 15, 15)
        main_layout.setSpacing(8)

        # Title and working directory
        title_label = QLabel(f"{APP_NAME}")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        cwd_label = QLabel(f"Current working directory: {self.cwd_label}")
        main_layout.addWidget(cwd_label)

        # Tab widget
        tab_widget = QTabWidget()

        # Files tab
        files_tab = QWidget()
        files_layout = QVBoxLayout(files_tab)

        # Session selection row
        session_row = QHBoxLayout()
        session_label = QLabel("Session:")
        # self.session_dropdown = QComboBox()
        # self.session_dropdown.addItems([f"Session {i+1}" for i in range(len(self.sessions))])
        # self.session_dropdown.currentIndexChanged.connect(self.on_session_selected)
        # Now populate the dropdown here
        self.update_dropdown()  # Add this line
        self.session_dropdown.setCurrentIndex(0)  # Set initial selection

        add_session_btn = QPushButton("+ Add Session")
        add_session_btn.clicked.connect(self.add_dropdown_session)
        remove_session_btn = QPushButton("– Remove Session")
        remove_session_btn.clicked.connect(self.remove_session)

        session_row.addWidget(session_label)
        session_row.addWidget(self.session_dropdown)
        session_row.addWidget(add_session_btn)
        session_row.addWidget(remove_session_btn)
        files_layout.addLayout(session_row)

        # Frame buttons
        frame_buttons = QHBoxLayout()
        lights_btn = QPushButton("Add Lights")
        lights_btn.clicked.connect(lambda: self.load_files("Lights"))
        darks_btn = QPushButton("Add Darks")
        darks_btn.clicked.connect(lambda: self.load_files("Darks"))
        flats_btn = QPushButton("Add Flats")
        flats_btn.clicked.connect(lambda: self.load_files("Flats"))
        biases_btn = QPushButton("Add Biases")
        biases_btn.clicked.connect(lambda: self.load_files("Biases"))
        biases_btn.setToolTip("Bias frames or Dark Flats can be used.")

        for btn in [lights_btn, darks_btn, flats_btn, biases_btn]:
            frame_buttons.addWidget(btn)
        files_layout.addLayout(frame_buttons)

        # Files list
        list_group = QGroupBox("Files in Current Session")
        list_layout = QVBoxLayout()
        self.file_listbox = QListWidget()
        self.file_listbox.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        list_layout.addWidget(self.file_listbox)

        file_buttons = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected File(s)")
        remove_btn.clicked.connect(self.remove_selected_files)
        reset_btn = QPushButton("Reset Everything")
        reset_btn.clicked.connect(self.reset_everything)
        reset_btn.setToolTip("Warning: This will remove all sessions and files!")

        file_buttons.addWidget(remove_btn)
        file_buttons.addWidget(reset_btn)
        list_layout.addLayout(file_buttons)

        list_group.setLayout(list_layout)
        files_layout.addWidget(list_group)

        # Processing tab
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)

        # Drizzle settings
        drizzle_group = QGroupBox("Optional Preprocessing Steps")
        drizzle_layout = QVBoxLayout()

        bg_extract_tooltip = "Removes background gradients from your images before stacking. Uses Polynomial value 1 and 10 samples."

        self.bg_extract_check = QCheckBox("Background Extraction")
        self.bg_extract_check.setToolTip(bg_extract_tooltip)
        drizzle_layout.addWidget(self.bg_extract_check)

        drizzle_tooltip = "Drizzle integration can improve resolution but increases processing time and file size. Use values above 1.0 with caution."
        self.drizzle_checkbox = QCheckBox("Enable Drizzle")
        self.drizzle_checkbox.setToolTip(drizzle_tooltip)
        drizzle_layout.addWidget(self.drizzle_checkbox)

        drizzle_amount_tooltip = "Scale factor for drizzle integration. Values between 1.0 and 3.0 are typical. \nNote: Higher values increase processing time and file size."
        drizzle_amount_layout = QHBoxLayout()
        drizzle_amount_label = QLabel("Drizzle Amount:")
        drizzle_amount_label.setToolTip(drizzle_amount_tooltip)

        self.drizzle_amount_spinbox = QDoubleSpinBox()
        self.drizzle_amount_spinbox.setRange(0.1, 3.0)
        self.drizzle_amount_spinbox.setSingleStep(0.1)
        self.drizzle_amount_spinbox.setValue(UI_DEFAULTS["drizzle_amount"])
        self.drizzle_amount_spinbox.setDecimals(1)
        self.drizzle_amount_spinbox.setMinimumWidth(80)
        self.drizzle_amount_spinbox.setSuffix(" x")
        self.drizzle_amount_spinbox.setEnabled(False)
        self.drizzle_amount_spinbox.setToolTip(drizzle_amount_tooltip)
        drizzle_amount_layout.addWidget(drizzle_amount_label)
        drizzle_amount_layout.addWidget(self.drizzle_amount_spinbox)
        drizzle_layout.addLayout(drizzle_amount_layout)

        self.drizzle_checkbox.toggled.connect(self.drizzle_amount_spinbox.setEnabled)

        pixel_fraction_label_tooltip = "Controls how much pixels overlap in drizzle integration. Lower values can reduce artifacts but may increase noise."
        pixel_fraction_layout = QHBoxLayout()
        pixel_fraction_label = QLabel("Pixel Fraction:")
        pixel_fraction_label.setToolTip(pixel_fraction_label_tooltip)
        self.pixel_fraction_spinbox = QDoubleSpinBox()
        self.pixel_fraction_spinbox.setRange(0.1, 10.0)
        self.pixel_fraction_spinbox.setSingleStep(0.1)
        self.pixel_fraction_spinbox.setValue(UI_DEFAULTS["pixel_fraction"])
        self.pixel_fraction_spinbox.setMinimumWidth(80)
        self.pixel_fraction_spinbox.setSuffix(" px")
        self.pixel_fraction_spinbox.setEnabled(False)
        self.pixel_fraction_spinbox.setToolTip(pixel_fraction_label_tooltip)
        pixel_fraction_layout.addWidget(pixel_fraction_label)
        pixel_fraction_layout.addWidget(self.pixel_fraction_spinbox)
        drizzle_layout.addLayout(pixel_fraction_layout)

        self.drizzle_checkbox.toggled.connect(self.pixel_fraction_spinbox.setEnabled)

        drizzle_group.setLayout(drizzle_layout)
        processing_layout.addWidget(drizzle_group)

        # Registration settings
        reg_group = QGroupBox("Optional Filter Settings")
        reg_layout = QVBoxLayout()

        # Registration controls
        roundness_label_tooltip = "Filters images by star roundness, calculated using the second moments of detected stars. \nA lower roundness value applies a stricter filter, keeping only frames with well-defined, circular stars. Higher roundness values allow more variation in star shapes."
        roundness_layout = QHBoxLayout()
        roundness_label = QLabel("Filter Roundness:")
        roundness_label.setToolTip(roundness_label_tooltip)
        self.roundness_spinbox = QDoubleSpinBox()
        self.roundness_spinbox.setRange(1, 4)
        self.roundness_spinbox.setSingleStep(0.1)
        self.roundness_spinbox.setValue(UI_DEFAULTS["filter_round"])
        self.roundness_spinbox.setDecimals(1)
        self.roundness_spinbox.setMinimumWidth(80)
        self.roundness_spinbox.setSuffix(" σ")
        self.roundness_spinbox.setToolTip(roundness_label_tooltip)
        roundness_layout.addWidget(roundness_label)
        roundness_layout.addWidget(self.roundness_spinbox)
        reg_layout.addLayout(roundness_layout)

        reg_group.setLayout(reg_layout)
        processing_layout.addWidget(reg_group)

        # FWHM filter
        fwhm_label_tooltip = "Filters images by weighted Full Width at Half Maximum (FWHM), calculated using star sharpness. \nA lower sigma value applies a stricter filter, keeping only frames close to the median FWHM. Higher sigma allows more variation."
        fwhm_layout = QHBoxLayout()
        fwhm_label = QLabel("Filter FWHM:")
        self.fwhm_spinbox = QDoubleSpinBox()
        fwhm_label.setToolTip(fwhm_label_tooltip)
        self.fwhm_spinbox.setRange(1, 4)
        self.fwhm_spinbox.setSingleStep(0.1)
        self.fwhm_spinbox.setValue(UI_DEFAULTS["filter_wfwhm"])
        self.fwhm_spinbox.setDecimals(1)
        self.fwhm_spinbox.setMinimumWidth(80)
        self.fwhm_spinbox.setSuffix(" σ")
        self.fwhm_spinbox.setToolTip(fwhm_label_tooltip)
        fwhm_layout.addWidget(fwhm_label)
        fwhm_layout.addWidget(self.fwhm_spinbox)
        reg_layout.addLayout(fwhm_layout)

        # Stacking settings
        stack_group = QGroupBox("Stacking Settings")
        stack_layout = QVBoxLayout()

        self.feather_checkbox = QCheckBox("Enable Feather")
        stack_layout.addWidget(self.feather_checkbox)

        feather_tooltip = "Blends the edges of stacked frames to reduce edge artifacts in the final image."
        feather_amount_label_tooltip = "Size of the feathering blend in pixels. Larger values create smoother transitions but may affect more of the image edge."
        feather_amount_layout = QHBoxLayout()
        feather_amount_label = QLabel("Feather Amount:")
        feather_amount_label.setToolTip(feather_tooltip)
        self.feather_amount_spinbox = QSpinBox()
        self.feather_amount_spinbox.setRange(5, 2000)
        self.feather_amount_spinbox.setSingleStep(5)
        self.feather_amount_spinbox.setValue(UI_DEFAULTS["feather_amount"])
        self.feather_amount_spinbox.setMinimumWidth(80)
        self.feather_amount_spinbox.setSuffix(" px")
        self.feather_amount_spinbox.setEnabled(False)
        self.feather_amount_spinbox.setToolTip(feather_amount_label_tooltip)
        feather_amount_layout.addWidget(feather_amount_label)
        feather_amount_layout.addWidget(self.feather_amount_spinbox)
        stack_layout.addLayout(feather_amount_layout)

        cleanup_tooltip = "Enable this option to delete all intermediary files after they are done processing. This saves space on your hard drive.\nNote: If your session is batched, this option is automatically enabled even if it's unchecked!"

        self.cleanup_check = QCheckBox("Clean up intermediate files")
        self.cleanup_check.setToolTip(cleanup_tooltip)
        stack_layout.addWidget(self.cleanup_check)

        self.feather_checkbox.toggled.connect(self.feather_amount_spinbox.setEnabled)

        stack_group.setLayout(stack_layout)
        processing_layout.addWidget(stack_group)

        # Process button
        process_btn = QPushButton("Process Files")
        process_btn.clicked.connect(
            lambda: self.run_script(
                bg_extract=self.bg_extract_check.isChecked(),
                drizzle=self.drizzle_checkbox.isChecked(),
                drizzle_amount=self.drizzle_amount_spinbox.value(),
                pixel_fraction=self.pixel_fraction_spinbox.value(),
                feather=self.feather_checkbox.isChecked(),
                feather_amount=self.feather_amount_spinbox.value(),
                filter_round=self.roundness_spinbox.value(),
                filter_wfwhm=self.fwhm_spinbox.value(),
                clean_up_files=self.cleanup_check.isChecked(),
            )
        )
        processing_layout.addWidget(process_btn)

        # Add tabs
        tab_widget.addTab(files_tab, "1. Files")
        tab_widget.addTab(processing_tab, "2. Processing")
        main_layout.addWidget(tab_widget)

        # Bottom buttons
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

        # button_layout.addWidget(help_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        # main_layout.addLayout(button_layout)

        # Convert the root window
        # self.root.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        # self.root.resize(900, 700)

    def close_dialog(self):
        self.siril.disconnect()
        self.close()

    def print_footer(self):
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

    def save_presets(self):
        """Save current UI settings and session data to a preset file"""
        # Collect settings
        presets = {
            "bg_extract": self.bg_extract_check.isChecked(),
            "drizzle": self.drizzle_checkbox.isChecked(),
            "drizzle_amount": round(self.drizzle_amount_spinbox.value(), 1),
            "pixel_fraction": round(self.pixel_fraction_spinbox.value(), 2),
            "feather": self.feather_checkbox.isChecked(),
            "feather_amount": round(self.feather_amount_spinbox.value(), 2),
            "filter_round": round(self.roundness_spinbox.value(), 1),
            "filter_wfwhm": round(self.fwhm_spinbox.value(), 1),
            "cleanup": self.cleanup_check.isChecked(),
            # Add session information
            "sessions": [],
        }

        # Collect data from all sessions
        for idx, session in enumerate(self.sessions):
            session_data = {
                "name": f"Session {idx + 1}",
                "lights": [str(path) for path in session.lights],
                "darks": [str(path) for path in session.darks],
                "flats": [str(path) for path in session.flats],
                "biases": [str(path) for path in session.biases],
            }
            presets["sessions"].append(session_data)

        # Create presets directory if it doesn't exist
        presets_dir = os.path.join(self.current_working_directory, "presets")
        os.makedirs(presets_dir, exist_ok=True)
        presets_file = os.path.join(presets_dir, "naztronomy_osc_pp_presets.json")

        try:
            with open(presets_file, "w") as f:
                json.dump(presets, f, indent=4)
            self.siril.log(
                f"Saved presets and session data to {presets_file}", LogColor.GREEN
            )
        except Exception as e:
            self.siril.log(f"Failed to save presets: {e}", LogColor.RED)

    def load_presets(self):
        """Load settings and session data from a preset file"""
        try:
            # Open file dialog to select presets file
            # First check for default presets file
            default_presets_file = os.path.join(
                self.current_working_directory,
                "presets",
                "naztronomy_osc_pp_presets.json",
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

                # Load UI settings
                self.bg_extract_check.setChecked(presets.get("bg_extract", False))
                self.drizzle_checkbox.setChecked(presets.get("drizzle", False))
                self.drizzle_amount_spinbox.setValue(presets.get("drizzle_amount", 1.0))
                self.pixel_fraction_spinbox.setValue(presets.get("pixel_fraction", 1.0))
                self.feather_checkbox.setChecked(presets.get("feather", False))
                self.feather_amount_spinbox.setValue(presets.get("feather_amount", 20))
                self.roundness_spinbox.setValue(presets.get("filter_round", 3.0))
                self.fwhm_spinbox.setValue(presets.get("filter_wfwhm", 3.0))
                self.cleanup_check.setChecked(presets.get("cleanup", False))

                # Load session data
                sessions_data = presets.get("sessions", [])
                if sessions_data:
                    # Clear existing sessions
                    self.sessions.clear()

                    # Create new sessions from loaded data
                    for session_data in sessions_data:
                        new_session = Session()
                        new_session.lights = [
                            Path(path) for path in session_data.get("lights", [])
                        ]
                        new_session.darks = [
                            Path(path) for path in session_data.get("darks", [])
                        ]
                        new_session.flats = [
                            Path(path) for path in session_data.get("flats", [])
                        ]
                        new_session.biases = [
                            Path(path) for path in session_data.get("biases", [])
                        ]
                        self.sessions.append(new_session)

                    # Update UI
                    self.update_dropdown()
                    self.session_dropdown.setCurrentIndex(0)
                    self.chosen_session = self.sessions[0]
                    self.refresh_file_list()

                self.siril.log(
                    f"Loaded presets and {len(sessions_data)} sessions from {presets_file}",
                    LogColor.GREEN,
                )
        except Exception as e:
            self.siril.log(f"Error loading presets: {str(e)}", LogColor.RED)

    def run_script(
        self,
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
        filter_round: float = UI_DEFAULTS["filter_round"],
        filter_wfwhm: float = UI_DEFAULTS["filter_wfwhm"],
        clean_up_files: bool = False,
    ):
        self.siril.log(
            f"Running script version {VERSION} with arguments:\n"
            f"bg_extract={bg_extract}\n"
            f"drizzle={drizzle}\n"
            f"drizzle_amount={drizzle_amount}\n"
            f"pixel_fraction={pixel_fraction}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}\n"
            f"filter_round={filter_round}\n"
            f"filter_wfwhm={filter_wfwhm}\n"
            f"clean_up_files={clean_up_files}",
            LogColor.BLUE,
        )
        self.siril.cmd("close")
        # Check files - if more than 2048, batch them:
        self.drizzle_status = drizzle
        self.drizzle_factor = drizzle_amount

        # Check files in working directory/lights.
        # create sub folders with more than 2048 divided by equal amounts

        # Get all sessions
        session_to_process = self.get_all_sessions()

        for idx, session in enumerate(
            session_to_process
        ):  # for session in session_to_process:
            # Copy session files to directories
            self.copy_session_files(session, f"session{idx + 1}")

        # e.g. CD sessions/session1
        for idx, session in enumerate(session_to_process):
            session_name = f"session{idx + 1}"
            self.siril.cmd("cd", f"sessions/{session_name}")
            self.current_working_directory = self.siril.get_siril_wd()
            session_file_counts = session.get_file_count()

            for image_type in ["darks", "biases", "flats"]:
                if session_file_counts.get(image_type, 0) > 0:
                    self.convert_files(image_type=image_type)
                    self.calibration_stack(seq_name=image_type)
                    self.clean_up(prefix=image_type)
                else:
                    self.siril.log(
                        f"Skipping {image_type}: no files found", LogColor.SALMON
                    )

            # Process lights
            # self.siril.cmd("cd", "lights")

            # Don't continue if no light frames
            total_lights = 0
            for session in self.sessions:
                file_counts = session.get_file_count()
                total_lights += file_counts.get("lights", 0)

            if total_lights == 0:
                self.siril.log(
                    "No light frames found. Only master calibration frames were created. Stopping script.",
                    LogColor.BLUE,
                )
                self.print_footer()
                self.siril.cmd("cd", "../..")
                return
            self.convert_files(image_type="lights")
            self.calibrate_lights(seq_name="lights", use_darks=True, use_flats=True)

            # Directory to move files to

            # Current directory where files are located
            current_dir = os.path.join(self.current_working_directory, "process")

            # Mitigate bug: If collected_lights doesn't exist, create it here because sometimes it doesn't get created earlier
            os.makedirs(self.collected_lights_dir, exist_ok=True)
            # Find and move all files starting with 'pp_lights'
            for file_name in os.listdir(current_dir):
                if file_name.startswith("pp_lights") and file_name.endswith(
                    self.fits_extension
                ):
                    src_path = os.path.join(current_dir, file_name)

                    # Prepend session_name to the filename
                    new_file_name = f"{session_name}_{file_name}"
                    dest_path = os.path.join(self.collected_lights_dir, new_file_name)

                    shutil.copy2(src_path, dest_path)
                    self.siril.log(
                        f"Moved {file_name} to {self.collected_lights_dir} as {new_file_name}",
                        LogColor.BLUE,
                    )

            # Go back to the previous directory
            self.siril.cmd("cd", "../../..")
            self.current_working_directory = self.siril.get_siril_wd()
            # If clean up is selected, delete the session# directories one after another.
            if clean_up_files:
                shutil.rmtree(
                    os.path.join(
                        self.current_working_directory, "sessions", session_name
                    )
                )

        self.siril.cmd("cd", f'"{self.collected_lights_dir}"')
        self.current_working_directory = self.siril.get_siril_wd()
        # Create a new sequence for each session
        for idx, session in enumerate(session_to_process):
            self.siril.create_new_seq(f"session{idx + 1}_pp_lights_")
        # Find all files starting with 'session' and ending with '.seq'

        if len(session_to_process) > 1:
            session_files = [
                file_name
                for file_name in os.listdir(self.current_working_directory)
                if file_name.startswith("session") and file_name.endswith(".seq")
            ]

            # Merge all session files
            seq_name = "pp_lights_merged_"
            if session_files:
                self.siril.cmd("merge", *session_files, seq_name)
                self.siril.log(
                    f"Merged session files: {', '.join(session_files)}", LogColor.GREEN
                )
            else:
                self.siril.log("No session files found to merge", LogColor.SALMON)
        else:
            seq_name = "session1_pp_lights_"

        if bg_extract:
            self.seq_bg_extract(seq_name=seq_name)
            seq_name = "bkg_" + seq_name

        plate_solve_status = self.seq_plate_solve(seq_name=seq_name)

        if plate_solve_status:
            self.seq_apply_reg(
                seq_name=seq_name,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
            )
        else:
            # If Siril can't plate solve, we apply regular registration with 2pass and then apply registration with max framing
            self.regular_register_seq(
                seq_name=seq_name,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
            )
            self.seq_apply_reg(
                seq_name=seq_name,
                drizzle_amount=drizzle_amount,
                pixel_fraction=pixel_fraction,
            )

        seq_name = f"r_{seq_name}"

        # Scans for black frames due to existing Siril bug.
        if drizzle:
            self.scan_black_frames(seq_name=seq_name, folder=self.collected_lights_dir)

        # Stacks the sequence with rejection
        stack_name = "merge_stacked" if len(session_to_process) > 1 else "final_stacked"
        self.seq_stack(
            seq_name=seq_name,
            feather=feather,
            feather_amount=feather_amount,
            rejection=True,
            output_name=stack_name,
        )

        self.load_image(image_name=stack_name)
        self.siril.cmd("cd", "../")
        self.current_working_directory = self.siril.get_siril_wd()
        file_name = self.save_image("_og")
        self.load_image(image_name=file_name)
        # Delete the blank sessions dir
        if clean_up_files:
            shutil.rmtree(os.path.join(self.current_working_directory, "sessions"))
            extension = self.fits_extension.lstrip(".")
            collected_lights_dir = os.path.join(
                self.current_working_directory, "collected_lights"
            )
            for filename in os.listdir(collected_lights_dir):
                file_path = os.path.join(collected_lights_dir, filename)

                if os.path.isfile(file_path) and not (
                    filename.startswith("session") and filename.endswith(extension)
                ):
                    os.remove(file_path)
            shutil.rmtree(os.path.join(collected_lights_dir, "cache"))

            self.siril.log("Cleaned up collected_lights directory", LogColor.BLUE)

        # self.clean_up()

        self.print_footer()

        # self.close_dialog()


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
