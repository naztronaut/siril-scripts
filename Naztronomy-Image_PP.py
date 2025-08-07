"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Telescope Preprocessing script
Version: 1.1.0
=====================================

The author of this script is Nazmus Nasir (Naztronomy) and can be reached at:
https://www.Naztronomy.com or https://www.YouTube.com/Naztronomy
Join discord for support and discussion: https://discord.gg/yXKqrawpjr
Support me on Patreon: https://www.patreon.com/c/naztronomy

TODO: Update below notes for non-smart scope setups
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
"""


import math
from pathlib import Path
import shutil
import sirilpy as s

s.ensure_installed("ttkthemes", "numpy", "astropy")
from datetime import datetime
import time
import os
import sys
import tkinter as tk
from tkinter import ttk
from sirilpy import LogColor, NoImageError, tksiril
from ttkthemes import ThemedTk
from astropy.io import fits
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# Beta 3
if sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog
# from tkinter import filedialog

APP_NAME = "Naztronomy - Image Preprocessor"
VERSION = "1.0.0"
AUTHOR = "Nazmus Nasir"
WEBSITE = "Naztronomy.com"
YOUTUBE = "YouTube.com/Naztronomy"


UI_DEFAULTS = {
    "feather_amount": 20.0,
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


class MacOSFriendlyDialog:
    def __init__(self, parent):
        self.parent = parent

    def askdirectory(self, **kwargs):
        """Dialogue de sélection de dossier optimisé pour macOS"""
        if sys.platform == "darwin":
            if self.parent:
                original_state = self.parent.state()

                self.parent.lift()
                self.parent.focus_force()
                self.parent.update_idletasks()

                kwargs_copy = kwargs.copy()
                if "parent" in kwargs_copy:
                    del kwargs_copy["parent"]

                result = filedialog.askdirectory(**kwargs_copy)

                if original_state == "normal":
                    self.parent.deiconify()
                self.parent.lift()

                return result

        return filedialog.askdirectory(**kwargs)


class PreprocessingInterface:

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} - v{VERSION}")

        self.root.geometry(
            f"900x700+{int(self.root.winfo_screenwidth()/5)}+{int(self.root.winfo_screenheight()/5)}"
        )
        self.root.resizable(True, True)

        self.style = tksiril.standard_style()

        self.siril = s.SirilInterface()

        # Flags for mosaic mode and drizzle status
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
        tksiril.match_theme_to_siril(self.root, self.siril)

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

        self.fits_extension = self.siril.get_siril_config("core", "extension")

        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label = self.current_working_directory

        # Create collected_lights directory to store all pp_lights files
        self.collected_lights_dir = os.path.join(
            self.current_working_directory, "collected_lights"
        )
        # if not os.path.exists(self.collected_lights_dir):
        #     os.makedirs(self.collected_lights_dir, exist_ok=True)

        # Sessions
        # self.sessions = []
        self.sessions = self.create_sessions(1)  # Start with one session
        self.chosen_session = self.sessions[0]
        self.current_session = tk.StringVar(value=f"Session {len(self.sessions)}")
        # self.session_dropdown = None
        # self.file_listbox = None

        # End Session
        self.create_widgets()

    # Start session methods
    def create_sessions(self, n_sessions: int) -> list[Session]:
        return [Session() for _ in range(n_sessions)]

    def get_session_count(self) -> int:
        return len(self.sessions)

    def get_session_by_index(self, index: int) -> Session:
        if 0 <= index < len(self.sessions):
            return self.sessions[index]
        else:
            raise IndexError("Session index out of range.")

    def get_all_sessions(self) -> List[Session]:
        return self.sessions.copy()

    def clear_all_sessions(self):
        for session in self.sessions:
            session.reset()
        return self.sessions

    def remove_session_by_index(self, index: int) -> List[Session]:
        if 0 <= index < len(self.sessions):
            return self.sessions[:index] + self.sessions[index + 1 :]
        else:
            raise IndexError("Session index out of range.")

    def add_session(self, session: Session) -> List[Session]:
        self.sessions.append(session)
        return self.sessions

    def update_session(self, index: int, session: Session) -> List[Session]:
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
    def on_session_selected(self, event):
        selected_index = int(self.current_session.get().split()[-1]) - 1
        self.chosen_session = self.get_session_by_index(selected_index)
        self.refresh_file_list()

    def add_dropdown_session(self):
        self.add_session(Session())
        self.update_dropdown()
        self.current_session.set(f"Session {len(self.sessions)}")
        self.chosen_session = self.sessions[
            len(self.sessions) - 1
        ]  # Set to the newly added session
        self.refresh_file_list()

    def remove_session(self):
        if len(self.sessions) <= 1:
            self.siril.log("Cannot remove the last session.", LogColor.BLUE)
            return  # don't allow removing the last session

        current = self.current_session.get()
        current_session_index = int(current.split()[-1]) - 1
        sess = self.get_session_by_index(current_session_index)  # Validate index
        self.sessions.remove(sess)
        self.update_dropdown()
        self.current_session.set("Session 1")  # fallback to first session
        self.chosen_session = self.sessions[0]  # Reset chosen_session
        self.refresh_file_list()

    def update_dropdown(self):
        session_names = [f"Session {i+1}" for i in range(len(self.sessions))]
        self.session_dropdown["values"] = session_names

    def load_files(self, filetype: str):
        file_paths = filedialog.askopenfilenames(title=f"Select {filetype} Files")
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
            f"> Added {len(paths)} {filetype} files to {self.current_session.get()}",
            LogColor.BLUE,
        )
        self.refresh_file_list()

    def copy_session_files(self, session: Session, session_name: str):
        """Copies all files from the session to the specified destination directory."""

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
                    shutil.copy(file, dest_path)
                    self.siril.log(f"Copied {file} to {dest_path}", LogColor.BLUE)
            else:
                self.siril.log(
                    f"Skipping {image_type}: no files found", LogColor.SALMON
                )

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        self.siril.log(f"Switched to session {self.chosen_session}", LogColor.BLUE)
        if self.chosen_session:
            for file_type in FRAME_TYPES:
                # file_list = chosen_session.get_file_lists().get(file_type, [])
                files = self.chosen_session.get_files_by_type(file_type)
                if files:
                    # file_listbox.insert(tk.END, f"--- {file_type.upper()} ---")
                    for index, file in enumerate(files):
                        self.file_listbox.insert(
                            tk.END,
                            f"{index + 1:>4}. {file_type.capitalize():^20}  {str(file.resolve())}",
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
            print(session)

    def remove_selected_files(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return

        msg = f"Are you sure you want to delete {len(selected_indices)} files? (Note: This will only remove them from the session, not delete them from disk.)"
        answer = tk.messagebox.askyesno("Delete Selected Files?", msg)
        if answer:
            # Build flat list of all files with type tracking
            all_files = (
                [("lights", f) for f in self.chosen_session.lights]
                + [("darks", f) for f in self.chosen_session.darks]
                + [("flats", f) for f in self.chosen_session.flats]
                + [("biases", f) for f in self.chosen_session.biases]
            )

            for index in reversed(
                selected_indices
            ):  # Remove from end to avoid reindexing issues
                filetype, path = all_files[index]
                getattr(self.chosen_session, filetype).remove(path)

            self.refresh_file_list()

    def reset_everything(self):
        msg = "Are you sure you want to reset all sessions? This will delete all file lists and reset the session count to 1."
        answer = tk.messagebox.askyesno("Reset all sessions?", msg)
        if answer:
            for session in self.sessions:
                session.reset()
            self.sessions = self.sessions[:1]
            self.update_dropdown()
            self.refresh_file_list()
            self.current_session.set("Session 1")

    # end session methods

    # Start Siril processing methods
    # Dirname: lights, darks, biases, flats
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
        # date_obs = current_fits_headers.get("DATE-OBS", current_datetime)

        # try:
        #     dt = datetime.fromisoformat(date_obs)
        #     date_obs_str = dt.strftime("%Y-%m-%d")
        # except ValueError:
        #     date_obs_str = datetime.now().strftime("%Y%m%d")

        file_name = f"{object_name}_{stack_count:03d}x{exptime}sec_{livetime}s_" #{date_obs_str}"
        if self.drizzle_status:
            file_name += f"__drizzle-{drizzle_str}x"

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

    def image_plate_solve(self):
        """Plate solve the loaded image with the '-force' argument."""
        try:
            self.siril.cmd("platesolve", "-force")
        except (s.DataError, s.CommandError, s.SirilError) as e:
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
        if oscsensor == "Unistellar Evscope 2":
            self.siril.cmd("pcc", f"-catalog={catalog}")
            self.siril.log(
                "PCC'd Image, SPCC Unavailable for Evscope 2", LogColor.GREEN
            )
        else:
            recoded_sensor = oscsensor
            """SPCC with oscsensor, filter, catalog, and whiteref."""
            if oscsensor in ["Dwarf 3"]:
                recoded_sensor = "Sony IMX678"
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
                self.siril.log(f"SPCC execution failed: {e}", LogColor.RED)
                self.close_dialog()

            img = self.save_image("_spcc")
            self.siril.log(f"Saved SPCC'd image: {img}", LogColor.GREEN)
            return img

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
            "Patreon: https://www.patreon.com/c/naztronomy\n\n"
            "Info:\n"
            "1. Recommended to use a blank working directory to have a clean setup.\n"
            "2. You can run this with or without calibration frames.\n"
            f"3. You can have as many sessions as you'd like. Each individual session currently has a limit of 2048 files on Windows machines.\n"
            "4. All preprocessed lights (pp_lights) are saved in the collected_lights directory and are not removed.\n"
            "5. This script makes a copy of all of your images so that the originals are not modified.\n"
            "6. Drizzle increases processing time. Higher the drizzle the longer it takes.\n"
            "7. When asking for help, please have the logs handy."
        )
        self.siril.info_messagebox(help_text, True)
        self.siril.log(help_text, LogColor.BLUE)

        tksiril.elevate(self.root)

    def create_widgets(self):
        """Creates the UI widgets."""
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True, anchor=tk.NW)

        # Define styles
        bold_label = ttk.Style()
        bold_label.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"), foreground="white")
        white_label = ttk.Style()
        white_label.configure("White.TLabel", font=("Segoe UI", 10), foreground="white")

        style = ttk.Style()
        style.configure("TButton", foreground="white")
        # style.configure("TLabel", foreground="white")
        style.configure("TCheckbutton", foreground="white")
        style.configure("TRadiobutton", foreground="white")
        style.configure("TMenubutton", foreground="white")
        style.configure("TEntry", foreground="white")
        style.configure("TCombobox", foreground="white")
        style.configure("TNotebook.Tab", foreground="white", font=("Segoe UI", 9, "bold"))
        style.configure("White.TSpinbox", foreground="white")

        # Title and version
        ttk.Label(
            main_frame,
            text=f"{APP_NAME}",
            style="Bold.TLabel",
        ).pack(pady=(10, 10))

        ttk.Label(
            main_frame,
            text=f"Current working directory: {self.cwd_label}",
        ).pack(anchor="w", pady=(0, 10))

        # tab 1

        tab = ttk.Notebook(main_frame)
        frame1 = ttk.Frame(tab)
        frame2 = ttk.Frame(tab)

        tab.add(frame1, text="1. Files")
        tab.add(frame2, text="2. Processing")
        tab.pack(fill=tk.BOTH, expand=True)

        # self.current_session = tk.StringVar(value=len(self.sessions))
        # Frame 1 Start

        # Session selection row
        session_row = ttk.Frame(frame1)
        session_row.pack(anchor="w", fill="x", pady=5)

        ttk.Label(session_row, text="Session:", style="White.TLabel").pack(side="left", padx=(0, 5))

        self.session_dropdown = ttk.Combobox(
            session_row,
            textvariable=self.current_session,
            values=[f"Session {i+1}" for i in range(len(self.sessions))],
            state="readonly",
            width=30,
        )
        self.session_dropdown.pack(side="left")
        self.session_dropdown.bind("<<ComboboxSelected>>", self.on_session_selected)

        ttk.Button(
            session_row, text="+ Add Session", command=self.add_dropdown_session
        ).pack(side="left", padx=5)

        ttk.Button(
            session_row, text="– Remove Session", command=self.remove_session
        ).pack(side="left", padx=5)

        # Separator (optional)
        ttk.Separator(frame1, orient="horizontal").pack(fill="x", pady=10)

        # Frame addition row
        frame_buttons = ttk.Frame(frame1)
        frame_buttons.pack(anchor="center", pady=5)

        ttk.Button(
            frame_buttons, text="Add Lights", command=lambda: self.load_files("Lights")
        ).pack(side="left", padx=10)

        ttk.Button(
            frame_buttons, text="Add Darks", command=lambda: self.load_files("Darks")
        ).pack(side="left", padx=10)

        ttk.Button(
            frame_buttons, text="Add Flats", command=lambda: self.load_files("Flats")
        ).pack(side="left", padx=10)

        add_bias_button = ttk.Button(
            frame_buttons, text="Add Biases", command=lambda: self.load_files("Biases")
        )
        add_bias_button.pack(side="left", padx=10)

        tksiril.create_tooltip(
            add_bias_button,
            "Bias frames or Dark Flats can be used.")
        
        # LabelFrame container for the section
        self.list_frame = ttk.LabelFrame(frame1, text="Files in Current Session", style="White.TLabel")
        self.list_frame.pack(fill="both", expand=True, padx=5, pady=10)

        # Label above listbox
        # ttk.Label(self.list_frame, text="Files in Current Session:").pack(anchor="w", pady=(10, 0))

        # === Frame for Listbox and Scrollbar side-by-side ===
        list_container = ttk.Frame(self.list_frame)
        list_container.pack(fill="both", expand=True)

        # Listbox
        self.file_listbox = tk.Listbox(
            list_container,
            selectmode=tk.EXTENDED,
            yscrollcommand=lambda *args: scrollbar.set(*args),
        )
        self.file_listbox.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            list_container, orient="vertical", command=self.file_listbox.yview
        )
        scrollbar.pack(side="right", fill="y")

        # === Frame for buttons under the listbox ===
        file_button_row = ttk.Frame(self.list_frame)
        file_button_row.pack(pady=(10, 0))

        ttk.Button(
            file_button_row,
            text="Remove Selected File(s)",
            command=self.remove_selected_files,
        ).pack(side="left", padx=5)

        reset_button = ttk.Button(
            file_button_row, text="Reset Everything", command=self.reset_everything
        )
        reset_button.pack(side="left", padx=5)

        tksiril.create_tooltip(
            reset_button,
            "Warning: This will remove all sessions and files!")

        # ttk.Button(
        #     file_button_row, text="Debug Print", command=self.show_all_sessions
        # ).pack(side="left", padx=5)

        # Frame 1 End

        # Frame 2 Start

        # Stacking section
        registration_section = ttk.LabelFrame(
            frame2, text="Registration Settings", padding=10
        )
        registration_section.pack(fill=tk.X, pady=5)
        # ttk.Label(registration_section, text="Telescope:", style="Bold.TLabel").grid(
        #     row=0, column=0, sticky="w"
        # )

        roundness_variable = tk.DoubleVar(value=UI_DEFAULTS["filter_round"])
        roundness_label = ttk.Label(registration_section, text="Filter Roundness:", style="White.TLabel")
        roundness_label.grid(row=1, column=0, sticky="w")
        roundness_spinbox = ttk.Spinbox(
            registration_section,
            textvariable=roundness_variable,
            from_=1,
            to=4,
            increment=0.1,
            style="White.TSpinbox"
        )
        roundness_spinbox.grid(row=1, column=2, sticky="w")
        ttk.Label(registration_section, text="σ", style="White.TLabel").grid(row=1, column=3, sticky="w")
        roundess_tooltip_text = "Filter frames based on roundness. A value of 3 is recommended for most images. Decreasing the value will filter out more images based on their FWHM values. If you have a lot of bad frames, decrease this value to 2.5 or 2 (or lower)."
        tksiril.create_tooltip(
            roundness_spinbox,
            roundess_tooltip_text
        )

        tksiril.create_tooltip(
            roundness_label,
            roundess_tooltip_text
        )

        wfwhm_variable = tk.DoubleVar(value=UI_DEFAULTS["filter_wfwhm"])
        wfwhm_label = ttk.Label(registration_section, text="Filter Weighted FWHM:", style="White.TLabel")
        wfwhm_label.grid(row=2, column=0, sticky="w")
        wfwhm_spinbox = ttk.Spinbox(
            registration_section,
            textvariable=wfwhm_variable,
            from_=1,
            to=4,
            increment=0.1,
            style="White.TSpinbox"
        )
        wfwhm_spinbox.grid(row=2, column=2, sticky="w")
        ttk.Label(registration_section, text="σ", style="White.TLabel").grid(row=2, column=3, sticky="w")
        wfwhm_tooltip_text = "Filters images by weighted Full Width at Half Maximum (FWHM), calculated using star sharpness. A lower sigma value applies a stricter filter, keeping only frames close to the median FWHM. Higher sigma allows more variation."

        tksiril.create_tooltip(
            wfwhm_spinbox,
            wfwhm_tooltip_text,
        )
        tksiril.create_tooltip(
            wfwhm_label,
            wfwhm_tooltip_text,
        )

        drizzle_label = ttk.Label(registration_section, text="Enable Drizzle:", style="White.TLabel")
        drizzle_label.grid(row=3, column=0, sticky="w")

        drizzle_checkbox_variable = tk.BooleanVar()
        drizzle_checkbox = ttk.Checkbutton(registration_section, variable=drizzle_checkbox_variable)
        drizzle_checkbox.grid(
            row=3, column=1, sticky="w"
        )
        
        drizzle_checkbox_tooltip_text = "Works best with well dithered data."
        tksiril.create_tooltip(
            drizzle_checkbox,
            drizzle_checkbox_tooltip_text
        )

        tksiril.create_tooltip(
            drizzle_label,
            drizzle_checkbox_tooltip_text
        )

        drizzle_amount_label = ttk.Label(registration_section, text="Drizzle Factor:", style="White.TLabel")
        drizzle_amount_label.grid(row=4, column=1, sticky="w")
        drizzle_amount_variable = tk.DoubleVar(value=UI_DEFAULTS["drizzle_amount"])
        drizzle_amount_spinbox = ttk.Spinbox(
            registration_section,
            textvariable=drizzle_amount_variable,
            from_=0.1,
            to=3.0,
            increment=0.1,
            state=tk.DISABLED,
            style="White.TSpinbox"
        )
        drizzle_amount_spinbox.grid(row=4, column=2, sticky="w")
        ttk.Label(registration_section, text="x", style="White.TLabel").grid(row=4, column=3, sticky="w")
        drizzle_checkbox_variable.trace_add(
            "write",
            lambda *args: drizzle_amount_spinbox.config(
                state=tk.NORMAL if drizzle_checkbox_variable.get() else tk.DISABLED,
                textvariable=drizzle_amount_variable,
            ),
        )

        drizzle_spinbox_tooltip_text = "Higher the drizzle factor, the longer it will take the stack your session. Larger value will upscale the image."
        tksiril.create_tooltip(
            drizzle_amount_spinbox,
            drizzle_spinbox_tooltip_text
        )
        tksiril.create_tooltip(
            drizzle_amount_label,    
            drizzle_spinbox_tooltip_text
        )
        


        pixel_fraction_label = ttk.Label(registration_section, text="Pixel Fraction:", style="White.TLabel")
        pixel_fraction_label.grid(
            row=5, column=1, sticky="w"
        )
        pixel_fraction_variable = tk.DoubleVar(value=UI_DEFAULTS["pixel_fraction"])
        pixel_fraction_spinbox = ttk.Spinbox(
            registration_section,
            textvariable=pixel_fraction_variable,
            from_=0.1,
            to=10.0,
            increment=0.01,
            state=tk.DISABLED,
            style="White.TSpinbox"
        )
        pixel_fraction_spinbox.grid(row=5, column=2, sticky="w")
        ttk.Label(registration_section, text="px", style="White.TLabel").grid(row=5, column=3, sticky="w")
        drizzle_checkbox_variable.trace_add(
            "write",
            lambda *args: pixel_fraction_spinbox.config(
                state=tk.NORMAL if drizzle_checkbox_variable.get() else tk.DISABLED,
                textvariable=pixel_fraction_variable,
            ),
        )

        pixel_fraction_tooltip_text = "Pixel size controls the drizzle output. Lower value can increase sharpness but can also product artifacts."
        tksiril.create_tooltip(
            pixel_fraction_spinbox,
            pixel_fraction_tooltip_text
        )
        tksiril.create_tooltip(
            pixel_fraction_label,    
            pixel_fraction_tooltip_text
        )

        # Optional Preprocessing Steps
        calib_section = ttk.LabelFrame(frame2, text="Other Optional Steps", padding=10)

        calib_section.pack(fill=tk.X, pady=5)
        background_extraction_label = ttk.Label(
            calib_section, text="Background Extraction:", style="White.TLabel"
        )
        background_extraction_label.grid(row=2, column=0, sticky="w")

        bg_extract_checkbox_variable = tk.BooleanVar()
        bg_extract_checkbox = ttk.Checkbutton(
            calib_section,
            text="",
            variable=bg_extract_checkbox_variable,
        )
        bg_extract_checkbox.grid(row=2, column=1, sticky="w")

        bg_extract_tooltip_text = "Enable this option to extract the background from each individual image. Uses polynomial factor 1 for extraction."
        tksiril.create_tooltip(
            bg_extract_checkbox,
            bg_extract_tooltip_text
        )
        tksiril.create_tooltip(
            background_extraction_label,
            bg_extract_tooltip_text
        )
        # ttk.Label(calib_section, text="Registration:", style="Bold.TLabel").grid(
        #     row=3, column=0, sticky="w"
        # )

        feather_checkbox_label = ttk.Label(calib_section, text="Feather Frames:", style="White.TLabel")
        feather_checkbox_label.grid(
            row=5, column=0, sticky="w"
        )

        feather_checkbox_variable = tk.BooleanVar()
        feather_checkbox = ttk.Checkbutton(
            calib_section, variable=feather_checkbox_variable
        )
        feather_checkbox.grid(row=5, column=1, sticky="w")
        feather_tooltip_text = "Only check this box if you're doing a multi panel mosaic! This will help remove artifacts between panels."
        tksiril.create_tooltip(
            feather_checkbox,
            feather_tooltip_text
        )
        tksiril.create_tooltip(
            feather_checkbox_label,
            feather_tooltip_text
        )
        feather_amount_label = ttk.Label(calib_section, text="Feather amount:", style="White.TLabel")
        feather_amount_label.grid(row=6, column=1, sticky="w")
        feather_amount_variable = tk.DoubleVar(value=UI_DEFAULTS["feather_amount"])
        feather_amount_spinbox = ttk.Spinbox(
            calib_section,
            textvariable=feather_amount_variable,
            from_=5,
            to=2000,
            increment=5,
            state=tk.DISABLED,
            style="White.TSpinbox"
        )
        feather_amount_spinbox.grid(row=6, column=2, sticky="w")
        ttk.Label(calib_section, text="px", style="White.TLabel").grid(row=6, column=3, sticky="w")
        feather_checkbox_variable.trace_add(
            "write",
            lambda *args: feather_amount_spinbox.config(
                state=tk.NORMAL if feather_checkbox_variable.get() else tk.DISABLED,
                textvariable=feather_amount_variable,
            ),
        )

        feather_amount_tooltip_text = "Feather amount in pixels. "
        tksiril.create_tooltip(
            feather_amount_spinbox,
            feather_amount_tooltip_text
        )
        tksiril.create_tooltip(
            feather_amount_label,
            feather_amount_tooltip_text
        )
        cleanup_files_label =ttk.Label(calib_section, text="Clean Up Files:", style="White.TLabel")
        cleanup_files_label.grid(
            row=7, column=0, sticky="w"
        )

        cleanup_files_checkbox_variable = tk.BooleanVar()
        cleanup_checkbox = ttk.Checkbutton(
            calib_section, text="", variable=cleanup_files_checkbox_variable
        )
        cleanup_checkbox.grid(row=7, column=1, sticky="w")
        cleanup_checkbox_tooltip_text = "Enable this option to delete all intermediary session files. This will NOT delete the gathered pp_lights files which can be found in the 'collected_lights' folder."
        tksiril.create_tooltip(
            cleanup_checkbox,
            cleanup_checkbox_tooltip_text
        )
        tksiril.create_tooltip(
            cleanup_files_label,
            cleanup_checkbox_tooltip_text
        )

        # Run button
        ttk.Button(
            frame2,
            text="Run",
            width=10,
            command=lambda: self.run_script(
                bg_extract=bg_extract_checkbox_variable.get(),
                drizzle=drizzle_checkbox_variable.get(),
                drizzle_amount=drizzle_amount_spinbox.get(),
                pixel_fraction=pixel_fraction_spinbox.get(),
                feather=feather_checkbox_variable.get(),
                feather_amount=feather_amount_spinbox.get(),
                filter_round=roundness_variable.get(),
                filter_wfwhm=wfwhm_variable.get(),
                clean_up_files=cleanup_files_checkbox_variable.get(),
            ),
        ).pack(pady=(15, 0), side=tk.RIGHT)

        # Frame 2 end
        # Help button
        ttk.Button(main_frame, text="Help", width=10, command=self.show_help).pack(
            pady=(15, 0), side=tk.LEFT
        )
        # Close button
        ttk.Button(main_frame, text="Close", width=10, command=self.close_dialog).pack(
            pady=(15, 0), side=tk.RIGHT
        )

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

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
        """, LogColor.BLUE)

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
                self.siril.log("No light frames found. Only master calibration frames were created. Stopping script.", LogColor.BLUE)
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

        self.siril.cmd("cd", self.collected_lights_dir)
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
            collected_lights_dir = os.path.join(self.current_working_directory, "collected_lights")
            for filename in os.listdir(collected_lights_dir):
                file_path = os.path.join(collected_lights_dir, filename)

                if os.path.isfile(file_path) and not (filename.startswith("session") and filename.endswith(extension)):
                    os.remove(file_path)
            shutil.rmtree(os.path.join(collected_lights_dir, "cache"))

            self.siril.log("Cleaned up collected_lights directory", LogColor.BLUE)

        # self.clean_up()

        self.print_footer()

        # self.close_dialog()


def main():
    try:
        root = ThemedTk()
        app = PreprocessingInterface(root)
        root.mainloop()
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

##############################################################################
