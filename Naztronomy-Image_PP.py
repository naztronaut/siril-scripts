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
    "drizzle_amount": 1.0,
    "pixel_fraction": 1.0,
    "max_files_per_batch": 2000,
}
FRAME_TYPES = ("darks", "flats", "biases", "lights")


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
            f"900x700+{int(self.root.winfo_screenwidth()/3)}+{int(self.root.winfo_screenheight()/3)}"
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

        # self.root.withdraw()  # Hide the main window
        # changed_cwd = False  # a way not to run the prompting loop
        # initial_cwd = os.path.join(self.current_working_directory, "lights")
        # if os.path.isdir(initial_cwd):
        #     self.siril.log(
        #         f"Current working directory is valid: {self.current_working_directory}",
        #         LogColor.GREEN,
        #     )
        #     self.siril.cmd("cd", f'"{self.current_working_directory}"')
        #     self.cwd_label.set(
        #         f"Current working directory: {self.current_working_directory}"
        #     )
        #     changed_cwd = True
        #     # self.root.deiconify()
        # elif os.path.basename(self.current_working_directory.lower()) == "lights":
        #     msg = "You're currently in the 'lights' directory, do you want to select the parent directory?"
        #     answer = tk.messagebox.askyesno("Already in Lights Dir", msg)
        #     if answer:
        #         self.siril.cmd("cd", "../")
        #         os.chdir(os.path.dirname(self.current_working_directory))
        #         self.current_working_directory = os.path.dirname(
        #             self.current_working_directory
        #         )
        #         self.cwd_label.set(
        #             f"Current working directory: {self.current_working_directory}"
        #         )
        #         self.siril.log(
        #             f"Updated current working directory to: {self.current_working_directory}",
        #             LogColor.GREEN,
        #         )
        #         changed_cwd = True
        #         # self.root.deiconify()
        #     else:
        #         self.siril.log(
        #             f"Current working directory is invalid: {self.current_working_directory}, reprompting...",
        #             LogColor.SALMON,
        #         )
        #         changed_cwd = False

        # if not changed_cwd:
        #     dialog_helper = MacOSFriendlyDialog(self.root)

        #     while True:
        #         prompt_title = (
        #             "Select the parent directory containing the 'lights' directory"
        #         )

        #         if sys.platform == "darwin":
        #             self.root.lift()
        #             self.root.attributes("-topmost", True)
        #             self.root.update()
        #             self.root.attributes("-topmost", False)

        #         selected_dir = dialog_helper.askdirectory(
        #             initialdir=self.current_working_directory,
        #             title=prompt_title,
        #         )

        #         if not selected_dir:
        #             self.siril.log(
        #                 "Canceled selecting directory. Restart the script to try again.",
        #                 LogColor.SALMON,
        #             )
        #             self.siril.disconnect()
        #             self.root.quit()
        #             self.root.destroy()
        #             break

        #         lights_directory = os.path.join(selected_dir, "lights")
        #         if os.path.isdir(lights_directory):
        #             self.siril.cmd("cd", f'"{selected_dir}"')
        #             os.chdir(selected_dir)
        #             self.current_working_directory = selected_dir
        #             self.cwd_label.set(f"Current working directory: {selected_dir}")
        #             self.siril.log(
        #                 f"Updated current working directory to: {selected_dir}",
        #                 LogColor.GREEN,
        #             )
        #             break

        #         elif os.path.basename(selected_dir.lower()) == "lights":
        #             msg = "The selected directory is the 'lights' directory, do you want to select the parent directory?"
        #             answer = tk.messagebox.askyesno("Already in Lights Dir", msg)
        #             if answer:
        #                 parent_dir = os.path.dirname(selected_dir)
        #                 self.siril.cmd("cd", f'"{parent_dir}"')
        #                 os.chdir(parent_dir)
        #                 self.current_working_directory = parent_dir
        #                 self.cwd_label.set(f"Current working directory: {parent_dir}")
        #                 self.siril.log(
        #                     f"Updated current working directory to: {parent_dir}",
        #                     LogColor.GREEN,
        #                 )
        #                 break
        #         else:
        #             msg = f"The selected directory must contain a subdirectory named 'lights'.\nYou selected: {selected_dir}. Please try again."
        #             self.siril.log(msg, LogColor.SALMON)
        #             tk.messagebox.showerror("Invalid Directory", msg)
        #             continue

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
        print(self.chosen_session)
        print(selected_index)
        self.refresh_file_list()

    def add_dropdown_session(self):
        self.add_session(Session())
        # next_number = len(sessions) + 1
        # new_session = f"Session {next_number}"
        # sessions.append(new_session)
        self.update_dropdown()
        self.current_session.set(f"Session {len(self.sessions)}")
        self.chosen_session = self.sessions[
            len(self.sessions) - 1
        ]  # Set to the newly added session
        self.refresh_file_list()

    def remove_session(self):
        if len(self.sessions) <= 1:
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

        print(f"> Added {len(paths)} {filetype} files to {self.current_session.get()}")
        self.refresh_file_list()

    def copy_session_files(self, session: Session, session_name: str):
        """Copies all files from the session to the specified destination directory."""
        # if not os.path.isdir(destination):
        #     raise FileNotFoundError(f"Destination directory {destination} does not exist.")

        destination = Path("sessions")
        if not destination.exists():
            os.mkdir(destination)
        session_dir = destination / session_name
        if not session_dir.exists():
            os.mkdir(session_dir)

        for file_type in FRAME_TYPES:
            type_dir = session_dir / file_type
            if not type_dir.exists():
                os.mkdir(type_dir)

        for file_type in FRAME_TYPES:
            files = session.get_files_by_type(file_type)
            for file in files:
                dest_path = session_dir / file_type / file.name
                shutil.copy(file, dest_path)
                self.siril.log(f"Copied {file} to {dest_path}", LogColor.GREEN)

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        print(self.chosen_session)
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

    def show_all_sessions(self):
        for session in self.sessions:
            for file_type in FRAME_TYPES:
                files = session.get_files_by_type(file_type)
                if files:
                    print(f"--- {file_type.upper()} ---")
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
        print(directory)
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

        # If origin or D2, need to pass in the focal length, pixel size, and target coordinates
        if self.chosen_telescope == "Celestron Origin":
            args.append(self.target_coords)
            focal_len = 400
            pixel_size = 2.4
            args.append(f"-focal={focal_len}")
            args.append(f"-pixelsize={pixel_size}")
        if self.chosen_telescope == "Dwarf 2":
            args.append(self.target_coords)
            focal_len = 100
            pixel_size = 1.45
            args.append(f"-focal={focal_len}")
            args.append(f"-pixelsize={pixel_size}")

        args.extend(["-nocache", "-force", "-disto=ps_distortion"])
        # args = ["platesolve", seq_name, "-disto=ps_distortion", "-force"]

        try:
            self.siril.cmd(*args)
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"seqplatesolve failed: {e}", LogColor.RED)
            # self.siril.error_messagebox(f"seqplatesolve failed: {e}")
            # self.close_dialog()
        self.siril.log(f"Platesolved {seq_name}", LogColor.GREEN)

    def seq_bg_extract(self, seq_name):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        try:
            self.siril.cmd("seqsubsky", seq_name, "1", "-samples=10")
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Seq BG Extraction failed: {e}", LogColor.RED)
            self.close_dialog()
        self.siril.log("Background extracted from Sequence", LogColor.GREEN)

    def seq_apply_reg(self, seq_name, drizzle_amount, pixel_fraction):
        """Apply Existing Registration to the sequence."""
        cmd_args = [
            "seqapplyreg",
            seq_name,
            "-filter-round=2.5k",
            # "-filter-fwhm=2k",
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
        print(f"Sequence name in calibration stacks: {seq_name}")
        print(
            os.path.join(
                self.current_working_directory,
                f"process/biases_stacked{self.fits_extension}",
            )
        )
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
                print("DID THIS HERE")
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
            f"-out={out}",
        ]
        if feather:
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
        stack_count = int(current_fits_headers.get("STACKCNT", 0))
        date_obs = current_fits_headers.get("DATE-OBS", current_datetime)

        try:
            dt = datetime.fromisoformat(date_obs)
            date_obs_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_obs_str = datetime.now().strftime("%Y%m%d")

        file_name = f"{object_name}_{stack_count:03d}x{exptime}sec_{date_obs_str}"
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
            '1. Must have a "lights" subdirectory inside of the working directory.\n'
            "2. For Calibration frames, you can have one or more of the following types: darks, flats, biases.\n"
            f"3. If on Windows and you have more than {UI_DEFAULTS["max_files_per_batch"]} files, this script will automatically split them into batches.\n"
            "4. If batching, intermediary files are cleaned up automatically even if 'clean up files' is unchecked.\n"
            "5. If batching, the frames are automatically feathered during the final stack even if 'feather' is unchecked.\n"
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
        bold_label.configure("Bold.TLabel", font=("TkDefaultFont", 10, "bold"))

        # Title and version
        ttk.Label(
            main_frame,
            text=f"{APP_NAME}",
            style="Bold.TLabel",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=(10, 10))

        ttk.Label(
            main_frame,
            text=f"{self.cwd_label}",
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
        self.session_dropdown = ttk.Combobox(
            frame1,
            textvariable=self.current_session,
            values=[f"Session {i+1}" for i in range(len(self.sessions))],
            state="readonly",
        )
        self.session_dropdown.pack(anchor="w", fill="x")
        self.session_dropdown.bind("<<ComboboxSelected>>", self.on_session_selected)

        button_frame = ttk.Frame(frame1)
        button_frame.pack(anchor="w", pady=10)

        ttk.Button(
            button_frame, text="Add Session", command=self.add_dropdown_session
        ).pack(side="left", padx=5)
        ttk.Button(
            button_frame, text="Remove Session", command=self.remove_session
        ).pack(side="left", padx=5)

        button_frame = ttk.LabelFrame(frame1, text="Add Frames")
        button_frame.pack(pady=10, fill="x")

        ttk.Button(
            button_frame, text="Add Lights", command=lambda: self.load_files("Lights")
        ).pack(fill="x", pady=2)
        ttk.Button(
            button_frame, text="Add Darks", command=lambda: self.load_files("Darks")
        ).pack(fill="x", pady=2)
        ttk.Button(
            button_frame, text="Add Flats", command=lambda: self.load_files("Flats")
        ).pack(fill="x", pady=2)
        ttk.Button(
            button_frame, text="Add Biases", command=lambda: self.load_files("Biases")
        ).pack(fill="x", pady=2)

        # Frame for file list
        self.list_frame = ttk.LabelFrame(frame1, text="Files in Current Session")
        self.list_frame.pack(fill="both", expand=True, padx=5, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.list_frame)
        scrollbar.pack(side="right", fill="y")

        # Listbox to display files
        self.file_listbox = tk.Listbox(
            self.list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set
        )
        self.file_listbox.pack(side="left", fill="both", expand=True)

        scrollbar.config(command=self.file_listbox.yview)
        ttk.Button(
            self.list_frame, text="Remove Selected", command=self.remove_selected_files
        ).pack(pady=(5, 0))
        ttk.Button(
            self.list_frame, text="Reset Everything", command=self.reset_everything
        ).pack(pady=(5, 0))
        ttk.Button(
            self.list_frame,
            text="Print Sessions (Debug)",
            command=self.show_all_sessions,
        ).pack(pady=(5, 0))

        # Frame 1 End

        # Frame 2 Start

        # Stacking section
        stacking_section = ttk.LabelFrame(frame2, text="Stacking Settings", padding=10)
        stacking_section.pack(fill=tk.X, pady=5)
        ttk.Label(stacking_section, text="Telescope:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        # Optional Calibration Frames

        ttk.Label(stacking_section, text="Clean Up Files:", style="Bold.TLabel").grid(
            row=2, column=0, sticky="w"
        )

        cleanup_files_checkbox_variable = tk.BooleanVar()
        cleanup_checkbox = ttk.Checkbutton(
            stacking_section, text="", variable=cleanup_files_checkbox_variable
        )
        cleanup_checkbox.grid(row=2, column=1, sticky="w", padx=5)
        tksiril.create_tooltip(
            cleanup_checkbox,
            "Enable this option to delete all intermediary files after they are done processing. This saves space on your hard drive.\n"
            "Note: If your session is batched, this option is automatically enabled even if it's unchecked!",
        )

        # Optional Preprocessing Steps
        calib_section = ttk.LabelFrame(
            frame2, text="Optional Preprocessing Steps", padding=10
        )

        calib_section.pack(fill=tk.X, pady=5)
        ttk.Label(
            calib_section, text="Background Extraction:", style="Bold.TLabel"
        ).grid(row=2, column=0, sticky="w")

        bg_extract_checkbox_variable = tk.BooleanVar()
        ttk.Checkbutton(
            calib_section,
            text="",
            variable=bg_extract_checkbox_variable,
        ).grid(row=2, column=1, sticky="w")

        ttk.Label(calib_section, text="Registration:", style="Bold.TLabel").grid(
            row=3, column=0, sticky="w"
        )

        drizzle_checkbox_variable = tk.BooleanVar()
        ttk.Checkbutton(
            calib_section, text="Drizzle?", variable=drizzle_checkbox_variable
        ).grid(row=3, column=1, sticky="w")

        drizzle_amount_label = ttk.Label(calib_section, text="Drizzle amount:")
        drizzle_amount_label.grid(row=3, column=2, sticky="w")
        drizzle_amount_variable = tk.DoubleVar(value=UI_DEFAULTS["drizzle_amount"])
        drizzle_amount_spinbox = ttk.Spinbox(
            calib_section,
            textvariable=drizzle_amount_variable,
            from_=0.1,
            to=3.0,
            increment=0.1,
            state=tk.DISABLED,
        )
        drizzle_amount_spinbox.grid(row=3, column=3, sticky="w")
        drizzle_checkbox_variable.trace_add(
            "write",
            lambda *args: drizzle_amount_spinbox.config(
                state=tk.NORMAL if drizzle_checkbox_variable.get() else tk.DISABLED,
                textvariable=drizzle_amount_variable,
            ),
        )
        ttk.Label(calib_section, text="Pixel Fraction:").grid(
            row=4, column=2, sticky="w"
        )
        pixel_fraction_variable = tk.DoubleVar(value=UI_DEFAULTS["pixel_fraction"])
        pixel_fraction_spinbox = ttk.Spinbox(
            calib_section,
            textvariable=pixel_fraction_variable,
            from_=0.1,
            to=10.0,
            increment=0.01,
            state=tk.DISABLED,
        )
        pixel_fraction_spinbox.grid(row=4, column=3, sticky="w")
        drizzle_checkbox_variable.trace_add(
            "write",
            lambda *args: pixel_fraction_spinbox.config(
                state=tk.NORMAL if drizzle_checkbox_variable.get() else tk.DISABLED,
                textvariable=pixel_fraction_variable,
            ),
        )

        ttk.Label(calib_section, text="Stacking:", style="Bold.TLabel").grid(
            row=5, column=0, sticky="w"
        )

        feather_checkbox_variable = tk.BooleanVar()
        feather_checkbox = ttk.Checkbutton(
            calib_section, text="Feather?", variable=feather_checkbox_variable
        )
        feather_checkbox.grid(row=5, column=1, sticky="w")

        feather_amount_label = ttk.Label(calib_section, text="Feather amount:")
        feather_amount_label.grid(row=5, column=2, sticky="w")
        feather_amount_variable = tk.DoubleVar(value=UI_DEFAULTS["feather_amount"])
        feather_amount_spinbox = ttk.Spinbox(
            calib_section,
            textvariable=feather_amount_variable,
            from_=5,
            to=2000,
            increment=5,
            state=tk.DISABLED,
        )
        feather_amount_spinbox.grid(row=5, column=3, sticky="w")
        feather_checkbox_variable.trace_add(
            "write",
            lambda *args: feather_amount_spinbox.config(
                state=tk.NORMAL if feather_checkbox_variable.get() else tk.DISABLED,
                textvariable=feather_amount_variable,
            ),
        )

        ttk.Button(frame2, text="Help", width=10, command=self.show_help).pack(
            pady=(15, 0), side=tk.LEFT
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
                clean_up_files=cleanup_files_checkbox_variable.get(),
            ),
        ).pack(pady=(15, 0), side=tk.RIGHT)

        # Frame 2 end

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

    def batch(
        self,
        output_name: str,
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
        clean_up_files: bool = False,
    ):
        # If we're batching, force cleanup files so we don't collide with existing files
        self.siril.cmd("close")
        if output_name.startswith("batch_lights"):
            clean_up_files = True

        self.drizzle_status = drizzle
        self.drizzle_factor = drizzle_amount

        # Output name is actually the name of the batched working directory
        self.convert_files(image_type=output_name)
        self.unselect_bad_fits(seq_name=output_name)

        seq_name = f"{output_name}_"

        # self.siril.cmd("cd", batch_working_dir)

        # Using calibration frames puts pp_ prefix in process directory
        if use_flats or use_darks:
            self.calibrate_lights(
                seq_name=seq_name, use_darks=use_darks, use_flats=use_flats
            )
            seq_name = "pp_" + seq_name

        if bg_extract:
            self.seq_bg_extract(seq_name=seq_name)
            if clean_up_files:
                self.clean_up(
                    prefix=seq_name
                )  # Remove "pp_lights_" or just "lights_" if not flat calibrated
            seq_name = "bkg_" + seq_name

        self.seq_plate_solve(seq_name=seq_name)
        # seq_name stays the same after plate solve
        self.seq_apply_reg(
            seq_name=seq_name,
            drizzle_amount=drizzle_amount,
            pixel_fraction=pixel_fraction,
        )
        if clean_up_files:
            self.clean_up(
                prefix=seq_name
            )  # Clean up bkg_ files or pp_ if flat calibrated, otherwise lights_
        seq_name = f"r_{seq_name}"

        if drizzle:
            self.scan_black_frames(seq_name=seq_name)

        self.seq_stack(
            seq_name=seq_name,
            feather=feather,
            feather_amount=feather_amount,
            rejection=True,
            output_name=output_name,
        )

        if clean_up_files:
            self.clean_up(prefix=seq_name)  # clean up r_ files

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

    def run_script(
        self,
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
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

        # TODO: Go into each session and run convert, process flats, and process lights
        # CD sessions/session1
        for idx, session in enumerate(session_to_process):
            session_name = f"session{idx + 1}"
            self.siril.cmd("cd", f"sessions/{session_name}")
            self.current_working_directory = self.siril.get_siril_wd()

            # Convert files
            for image_type in ["darks", "biases", "flats"]:
                self.convert_files(image_type=image_type)
                self.calibration_stack(
                    seq_name=image_type,
                )
                self.clean_up(prefix=image_type)

            # Process lights
            self.siril.cmd("cd", "lights")
            self.convert_files(image_type="lights")
            self.calibrate_lights(seq_name="lights", use_darks=True, use_flats=True)

        # TODO: take the pp_lights from the above dir, and put it into cwd/calibrated_lights

        # TODO: run process to stack all pp lights

        # lights_directory = "lights"

        # Get list of all files in the lights directory
        # all_files = [
        #     name
        #     for name in os.listdir(lights_directory)
        #     if os.path.isfile(os.path.join(lights_directory, name))
        # ]
        # num_files = len(all_files)
        # is_windows = sys.platform.startswith("win")

        # # only one batch will be run if less than max_files_per_batch OR not windows.
        # if num_files <= UI_DEFAULTS["max_files_per_batch"] or not is_windows:
        #     self.siril.log(
        #         f"{num_files} files found in the lights directory which is less than or equal to {UI_DEFAULTS['max_files_per_batch']} files allowed per batch - no batching needed.",
        #         LogColor.BLUE,
        #     )
        #     file_name = self.batch(
        #         output_name=lights_directory,
        #         bg_extract=bg_extract,
        #         drizzle=drizzle,
        #         drizzle_amount=drizzle_amount,
        #         pixel_fraction=pixel_fraction,
        #         feather=feather,
        #         feather_amount=feather_amount,
        #         clean_up_files=clean_up_files,
        #     )

        #     self.load_image(image_name=file_name)
        # else:
        #     num_batches = math.ceil(num_files / UI_DEFAULTS["max_files_per_batch"])
        #     batch_size = math.ceil(num_files / num_batches)

        #     self.siril.log(
        #         f"{num_files} files found in the lights directory, splitting into {num_batches} batches...",
        #         LogColor.BLUE,
        #     )

        #     # Ensure temp folders exist and are empty
        #     for i in range(num_batches):
        #         batch_dir = f"batch_lights{i+1}"
        #         os.makedirs(batch_dir, exist_ok=True)
        #         # Optionally clean out existing files:
        #         for f in os.listdir(batch_dir):
        #             os.remove(os.path.join(batch_dir, f))

        #     # Split and copy files into batches
        #     for i, filename in enumerate(all_files):
        #         batch_index = i // UI_DEFAULTS["max_files_per_batch"]
        #         batch_dir = f"batch_lights{batch_index + 1}"
        #         src_path = os.path.join(lights_directory, filename)
        #         dest_path = os.path.join(batch_dir, filename)
        #         shutil.copy2(src_path, dest_path)

        #     # Send each of the new lights dir into batch directory
        #     for i in range(num_batches):
        #         batch_dir = f"batch_lights{i+1}"
        #         self.siril.log(f"Processing batch: {batch_dir}", LogColor.BLUE)
        #         self.batch(
        #             output_name=batch_dir,
        #             bg_extract=bg_extract,
        #             drizzle=drizzle,
        #             drizzle_amount=drizzle_amount,
        #             pixel_fraction=pixel_fraction,
        #             feather=feather,
        #             feather_amount=feather_amount,
        #             clean_up_files=clean_up_files,
        #         )
        #     self.siril.log("Batching complete.", LogColor.GREEN)

        #     # Create batched_lights directory
        #     final_stack_seq_name = "final_stack"
        #     batch_lights = "batch_lights"
        #     os.makedirs(final_stack_seq_name, exist_ok=True)
        #     source_dir = os.path.join(os.getcwd(), "process")
        #     # Move batch result files into batched_lights
        #     target_subdir = os.path.join(os.getcwd(), final_stack_seq_name)

        #     # Create the target subdirectory if it doesn't exist
        #     os.makedirs(target_subdir, exist_ok=True)

        #     # Loop through all files in the source directory
        #     for filename in os.listdir(source_dir):
        #         if f"{batch_lights}" in filename:
        #             full_src_path = os.path.join(source_dir, filename)
        #             full_dst_path = os.path.join(target_subdir, filename)

        #         # Only move files, skip directories
        #         if os.path.isfile(full_src_path):
        #             shutil.move(full_src_path, full_dst_path)
        #             self.siril.log(f"Moved: {filename}", LogColor.BLUE)

        #     # Clean up temp_lightsX directories
        #     for i in range(num_batches):
        #         batch_dir = f"{batch_lights}{i+1}"
        #         shutil.rmtree(batch_dir, ignore_errors=True)

        #     self.convert_files(final_stack_seq_name)
        #     self.seq_plate_solve(seq_name=final_stack_seq_name)
        #     # turn off drizzle for this
        #     self.drizzle_status = False
        #     self.seq_apply_reg(
        #         seq_name=final_stack_seq_name,
        #         drizzle_amount=drizzle_amount,
        #         pixel_fraction=pixel_fraction,
        #     )
        #     self.clean_up(prefix=final_stack_seq_name)
        #     registered_final_stack_seq_name = f"r_{final_stack_seq_name}"
        #     # final stack needs feathering and amount
        #     self.drizzle_status = drizzle  # Turn drizzle back to selected option
        #     self.seq_stack(
        #         seq_name=registered_final_stack_seq_name,
        #         feather=True,
        #         rejection=False,
        #         feather_amount=60,
        #         output_name="final_result",
        #     )
        #     self.load_image(image_name="final_result")

        #     # cleanup final_stack directory
        #     shutil.rmtree(final_stack_seq_name, ignore_errors=True)
        #     self.clean_up(prefix=registered_final_stack_seq_name)

        #     # Go back to working dir
        #     self.siril.cmd("cd", "../")

        #     # Save og image in WD - might have drizzle factor in name
        #     file_name = self.save_image("_batched")

        # Spcc as a last step

        # self.clean_up()
        import datetime

        self.siril.log(
            f"Finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
        """,
            LogColor.BLUE,
        )

        self.close_dialog()


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
