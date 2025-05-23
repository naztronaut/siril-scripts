"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Smart Telescope Preprocessing script
Version: 1.0.0
=====================================

The author of this script is Nazmus Nasir (Naztronomy) and can be reached at:
https://www.Naztronomy.com or https://www.YouTube.com/Naztronomy
Join discord for support and discussion: https://discord.gg/yXKqrawpjr

"""

from datetime import datetime
import os
import sys
import shutil
import tkinter as tk
from tkinter import ttk
import sirilpy as s
from sirilpy import LogColor, NoImageError, tksiril
from ttkthemes import ThemedTk
from astropy.io import fits
import numpy as np

s.ensure_installed("ttkthemes", "numpy", "astropy")

APP_NAME = "Smart Telescope Preprocessing"
VERSION = "1.0.0"
AUTHOR = "Nazmus Nasir"
WEBSITE = "https://www.Naztronomy.com (https://www.YouTube.com/Naztronomy)"
TELESCOPES = ["ZWO Seestar S30", "ZWO Seestar S50", "Dwarf 3", "Celestron Origin"]
FILTER_OPTIONS_MAP = {
    "ZWO Seestar S30": ["No Filter (Broadband)", "LP (Narrowband)"],
    "ZWO Seestar S50": ["No Filter (Broadband)", "LP (Narrowband)"],
    "Dwarf 3": ["Astro filter (UV/IR)", "Dual-Band"],
    "Celestron Origin": ["Broadband", "Origin Nebula Filter"],
    # "Evscope2": ["No Filter (Broadband)"],
}

FILTER_COMMANDS_MAP = {
    "ZWO Seestar S30": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
        "LP (Narrowband)": ["-oscfilter=ZWO Seestar LP"],
    },
    "ZWO Seestar S50": {
        "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
        "LP (Narrowband)": ["-oscfilter=ZWO Seestar LP"],
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
    "Celestron Origin": {
        "Broadband": ["-oscfilter=No Filter"],
        "Origin Nebula Filter": [
            "-narrowband",
            "-rwl=656.28",
            "-rbw=20",
            "-gwl=498",
            "-gbw=15",
            "-bwl=498",
            "-bbw=15",
        ],
    },
    # "Evscope2": {
    #     "No Filter (Broadband)": ["-oscfilter=UV/IR Block"],
    # },
}


UI_DEFAULTS = {
    "feather_amount": 20.0,
    "drizzle_amount": 1.0,
    "pixel_fraction": 1.0,
}


class PreprocessingInterface:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Smart Telescope Preprocessing - v{VERSION}")
        self.root.geometry("800x700")
        self.root.resizable(False, False)

        self.style = tksiril.standard_style()

        self.siril = s.SirilInterface()

        # Flags for mosaic mode and drizzle status
        # If mosaic mode, fitseq will be used but cannot use framing max
        # if drizzle is off, images will be debayered on convert
        self.fitseq_mode = False
        self.drizzle_status = False

        self.spcc_section = ttk.LabelFrame()
        self.spcc_checkbox_variable = None
        self.telescope_options = TELESCOPES
        self.telescope_variable = tk.StringVar(value="ZWO Seestar S50")
        self.filter_variable = tk.StringVar(value="broadband")

        self.filter_options_map = FILTER_OPTIONS_MAP
        self.current_filter_options = self.filter_options_map[
            self.telescope_variable.get()
        ]
        self.filter_menu = None
        try:
            self.siril.connect()
            self.siril.log("Connected to Siril", LogColor.GREEN)
        except s.SirilConnectionError:
            self.siril.log("Failed to connect to Siril", LogColor.RED)
            self.close_dialog()

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

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
        lights_directory = os.path.join(self.current_working_directory, "lights")
        if not os.path.isdir(lights_directory):
            raise self.siril.error_messagebox(
                "Directory 'lights' does not exist, please change current working directory and try again.",
                True,
            )
        self.create_widgets()

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

            if sys.platform.startswith("win") and not self.fitseq_mode and dir_name == "lights" and file_count > 2048:
                self.siril.error_messagebox(
                    "More than 2048 images found on this Windows machine. Please select '2048+ Mode' and try again."
                )
                self.close_dialog()

            try:
                args = ["convert", dir_name, "-out=../process"]
                if self.fitseq_mode:
                    if not self.drizzle_status:
                        args.extend(["-fitseq", "-debayer"])
                    else:  # If Drizzle is on, don't debayer on convert
                        args.append("-fitseq")
                else:
                    if not self.drizzle_status:
                        args.append("-debayer")
                self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                self.siril.cmd(*args)
            except s.CommandError as e:
                self.siril.error_messagebox(f"{e}")
                self.close_dialog()
            self.siril.cmd("cd", "../process")
            self.siril.log(
                f"Converted {file_count} files for processing!", LogColor.GREEN
            )
        else:
            self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            raise NoImageError(
                (
                    f'No directory named "{dir_name}" at this location. Make sure the working directory is correct.'
                )
            )

    # Plate solve on sequence runs when file count < 2048
    def seq_plate_solve(self, seq_name):
        """Runs the siril command 'seqplatesolve' to plate solve the converted files."""
        try:
            self.siril.cmd(
                "seqplatesolve", seq_name, "-nocache", "-force", "-disto=ps_distortion"
            )
        except s.DataError as e:
            self.siril.error_messagebox(f"seqplatesolve failed: {e}")
            self.close_dialog()
        self.siril.log(f"Platesolved {seq_name}", LogColor.GREEN)

    def seq_graxpert_bg_extract(self):
        """Runs Graxpert to extract bg for each file in the sequence. Uses GPU if available."""
        self.siril.cmd("seqgraxpert_bg", "lights_", "-gpu")

    def seq_bg_extract(self, seq_name):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        try:
            self.siril.cmd("seqsubsky", seq_name, "1", "-samples=10")
        except s.DataError as e:
            self.siril.error_messagebox(f"seqsubsky failed: {e}")
            self.close_dialog()
        self.siril.log("Background extracted from Sequence", LogColor.GREEN)

    def seq_apply_reg(self, seq_name, drizzle_amount, pixel_fraction):
        """Apply Existing Registration to the sequence."""
        if self.fitseq_mode:
            cmd_args = ["register", seq_name]

        else:
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
        except s.DataError as e:
            self.siril.log(f"Data error occurred: {e}", LogColor.RED)

        # Need the extra reg for fitseq
        if self.fitseq_mode:
            try:
                self.siril.cmd(
                    "seqapplyreg", seq_name, "-filter-round=2.5k", "-filter-fwhm=2k"
                )
            except s.DataError as e:
                self.siril.error_messagebox(f"seqapplyreg failed for fitseq: {e}")
                self.close_dialog()

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

        for idx, filename in enumerate(sorted(os.listdir(folder))):
            if filename.startswith(seq_name) and filename.lower().endswith(
                (".fit", ".fits")
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
                                dynamic_threshold = 0.004

                            is_black, median_val = self.is_black_frame(
                                data, dynamic_threshold, crop_fraction
                            )
                            all_frames_info.append((filename, median_val))

                            # Log for debugging
                            print(
                                f"{filename} | shape: {data.shape} | dtype: {data.dtype} | min: {np.min(data)} | max: {data_max} | median: {median_val} | threshold used: {dynamic_threshold}"
                            )

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

    def seq_stack(self, seq_name, feather, feather_amount):
        """Stack it all, and feather if it's provided"""
        cmd_args = [
            "stack",
            f"{seq_name} rej 3 3",
            "-norm=addscale",
            "-output_norm",
            "-rgb_equal",
            "-maximize",
            "-filter-included",
            "-out=result",
        ]
        if not self.fitseq_mode and feather and feather_amount is not None:
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
        except s.DataError as e:
            self.siril.error_messagebox(f"Command execution failed: {e}")
            self.close_dialog()

        self.siril.log("Completed stacking!", LogColor.GREEN)

    def save_image(self, suffix):
        """Saves the image as a FITS file."""
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
        # file_name = f"$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dt${current_datetime}{suffix}"
        file_name = f"{current_datetime}{suffix}"

        try:
            self.siril.cmd(
                "save",
                f"../{file_name}",
            )
        except s.CommandError as e:
            self.siril.error_messagebox(f"save command failed: {e}")
            self.close_dialog()
        self.siril.log(f"Saved file: {file_name}", LogColor.GREEN)

    def load_registered_image(self):
        """Loads the registered image. Currently unused"""
        try:
            self.siril.cmd("load", "result")
        except s.CommandError as e:
            self.siril.error_messagebox(f"Load command failed: {e}")
        self.save_image("_og")

    def image_plate_solve(self):
        """Plate solve the loaded image with the '-force' argument."""
        try:
            self.siril.cmd("platesolve", "-force")
        except s.CommandError as e:
            self.siril.error_messagebox(f"Plate solve command failed: {e}")
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
            if oscsensor == "Dwarf 3":
                recoded_sensor = "Sony IMX678"
            elif oscsensor == "Celestron Origin":
                recoded_sensor = "Sony IMX178C"
            elif oscsensor == "Unistellar Evscope 2":
                recoded_sensor = "Sony IMX294C"
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
            except (s.CommandError, s.DataError) as e:
                self.siril.error_messagebox(f"SPCC command failed: {e}")
                self.close_dialog()

            self.save_image("_spcc")
            self.siril.log("SPCC'd image", LogColor.GREEN)

    def load_image(self, image_name):
        """Loads the result."""
        try:
            self.siril.cmd("load", image_name)
        except (s.CommandError, s.DataError) as e:
            self.siril.error_messagebox(f"Load command failed: {e}")
            self.close_dialog()
        self.siril.log(f"Loaded image: {image_name}", LogColor.GREEN)

    def autostretch(self, do_spcc):
        """Autostretch as a way to preview the final result"""
        try:
            self.siril.cmd("autostretch", *(["-linked"] if do_spcc else []))
        except s.CommandError as e:
            self.siril.error_messagebox(f"Autostretch command failed: {e}")
            self.close_dialog()
        self.siril.log(
            "Autostretched image."
            + (" You may want to open the _spcc file instead." if do_spcc else ""),
            LogColor.GREEN,
        )

    def clean_up(self, prefix=None):
        """Cleans up all files in the process directory."""
        self.siril.cmd("cd", "../")
        # TODO: Delete each previous sequence to save space
        # process_directory = os.path.join(self.current_working_directory, 'process')
        # if os.path.isdir(process_directory):
        #     shutil.rmtree(process_directory)

    # Function to update filter options
    def update_filter_options(self, *args):
        print("here")

        selected_scope = self.telescope_variable.get()
        new_options = self.filter_options_map.get(selected_scope, [])
        print(selected_scope)
        # Clear current menu
        menu = self.filter_menu["menu"]
        menu.delete(0, "end")

        # Add new options
        for option in new_options:
            menu.add_command(
                label=option,
                command=lambda value=option: self.filter_variable.set(value),
            )

        # Set default and enable menu
        self.filter_variable.set(new_options[0])
        state = tk.NORMAL if self.spcc_checkbox_variable.get() else tk.DISABLED
        self.filter_menu["state"] = state

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
            text=f"{APP_NAME} v{VERSION}",
            style="Bold.TLabel",
        ).pack(anchor="w")
        ttk.Label(main_frame, text=f"Author: {AUTHOR}", style="Bold.TLabel").pack(
            anchor="w", pady=(0, 0)
        )
        ttk.Label(main_frame, text=f"Website: {WEBSITE}", style="Bold.TLabel").pack(
            anchor="w", pady=(0, 0)
        )

        ttk.Label(
            main_frame,
            text=f"Current Working Directory: {self.current_working_directory}",
        ).pack(anchor="w", pady=(0, 10))

        # Telescope section
        telescope_section = ttk.LabelFrame(main_frame, text="Telescope", padding=10)
        telescope_section.pack(fill=tk.X, pady=5)
        ttk.Label(telescope_section, text="Telescope:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w"
        )

        ttk.OptionMenu(
            telescope_section,
            self.telescope_variable,
            "ZWO Seestar S30",
            *self.telescope_options,
        ).grid(row=0, column=1, sticky="w")

        self.telescope_variable.trace_add("write", self.update_filter_options)

        # Mode to handle >2048 files
        telescope_section.pack(fill=tk.X, pady=5)

        ttk.Label(telescope_section, text="2048+ Files:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w"
        )

        fitseq_checkbox_variable = tk.BooleanVar()

        fitseq_checkbox = ttk.Checkbutton(
            telescope_section, text="Enable", variable=fitseq_checkbox_variable
        )
        fitseq_checkbox.grid(row=1, column=1, sticky="w")

        # Add tooltip to the one and only checkbox
        tksiril.create_tooltip(
            fitseq_checkbox,
            "Enable this option if you have more than 2048 images to process. A different workflow will be used and the sequence will not be plate solved and the framing method will be default."
            "This can affect how large mosaic sessions look.",
        )

        # Optional Preprocessing Steps
        calib_section = ttk.LabelFrame(
            main_frame, text="Optional Preprocessing Steps", padding=10
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
        fitseq_checkbox_variable.trace_add(
            "write",
            lambda *args: feather_checkbox.config(
                state=tk.DISABLED if fitseq_checkbox_variable.get() else tk.NORMAL
            ),
        )

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

        # SPCC Section
        self.spcc_section = ttk.LabelFrame(main_frame, text="Post-Stacking", padding=10)
        self.spcc_section.pack(fill=tk.X, pady=5)

        self.spcc_checkbox_variable = tk.BooleanVar()

        def toggle_filter_and_gaia():
            state = tk.NORMAL if self.spcc_checkbox_variable.get() else tk.DISABLED
            self.filter_menu["state"] = state
            catalog_menu["state"] = state

        ttk.Checkbutton(
            self.spcc_section,
            text="Enable Spectrophotometric Color Calibration (SPCC)",
            variable=self.spcc_checkbox_variable,
            command=toggle_filter_and_gaia,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(self.spcc_section, text="OSC Filter:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w"
        )

        # filter_options = ["broadband", "narrowband"]
        self.filter_menu = ttk.OptionMenu(
            self.spcc_section,
            self.filter_variable,
            "No Filter (Broadband)",
            *self.current_filter_options,
        )
        self.filter_menu.grid(row=1, column=1, sticky="w")
        self.filter_menu["state"] = tk.DISABLED

        ttk.Label(self.spcc_section, text="Catalog:", style="Bold.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        catalog_variable = tk.StringVar(value="localgaia")
        catalog_options = ["localgaia", "gaia"]
        catalog_menu = ttk.OptionMenu(
            self.spcc_section, catalog_variable, "localgaia", *catalog_options
        )
        catalog_menu.grid(row=2, column=1, sticky="w")
        catalog_menu["state"] = tk.DISABLED

        if self.gaia_catalogue_available:
            ttk.Label(
                self.spcc_section,
                text="✓ Local Gaia Available",
                foreground="green",
                style="Success.TLabel",
            ).grid(row=2, column=2, sticky="w")
        else:
            ttk.Label(
                self.spcc_section, text="✗ Local Gaia Not available", foreground="red"
            ).grid(row=2, column=2, sticky="w")

        # Run button
        ttk.Button(
            main_frame,
            text="Run",
            command=lambda: self.run_script(
                fitseq_mode=fitseq_checkbox_variable.get(),
                do_spcc=self.spcc_checkbox_variable.get(),
                filter=self.filter_variable.get(),
                telescope=self.telescope_variable.get(),
                catalog=catalog_variable.get(),
                # use_darks=darks_checkbox_variable.get(),
                # use_flats=flats_checkbox_variable.get(),
                # use_bias=bias_checkbox_variable.get(),
                # graxpert=graxpert_checkbox_variable.get(),
                bg_extract=bg_extract_checkbox_variable.get(),
                drizzle=drizzle_checkbox_variable.get(),
                drizzle_amount=drizzle_amount_spinbox.get(),
                pixel_fraction=pixel_fraction_spinbox.get(),
                feather=feather_checkbox_variable.get(),
                feather_amount=feather_amount_spinbox.get(),
            ),
        ).pack(pady=(15, 0), anchor="w")

        # SPCC Button for Debugging
        # ttk.Button(
        #     main_frame,
        #     text="SPCC",
        #     command=lambda: self.spcc(
        #         oscsensor=telescope_variable.get(),
        #         filter=filter_variable.get(),
        #         catalog=catalog_variable.get(),
        #         whiteref="Average Spiral Galaxy",
        #     ),
        # ).pack(pady=(15, 0), anchor="w")

        # Fit Sequence Extract Button for Debugging
        # ttk.Button(
        #     main_frame,
        #     text="FITS Extract",
        #     command=lambda: self.extract_fits(),
        # ).pack(pady=(15, 0), anchor="w")

        # ttk.Button(
        #     main_frame,
        #     text="Test Register",
        #     command=lambda: self.seq_apply_reg(),
        # ).pack(pady=(15, 0), anchor="w")
        # ttk.Button(
        #     main_frame,
        #     text="Remove Bad Frames",
        #     command=lambda: self.scan_black_frames(seq_name="r_bkg_lights_"),
        # ).pack(pady=(15, 0), anchor="w")

        # ttk.Button(
        #     main_frame,
        #     text="Stack Again",
        #     command=lambda: self.seq_stack(
        #         seq_name="r_bkg_lights_",feather=feather_checkbox_variable.get(), feather_amount=feather_amount_spinbox.get()),
        # ).pack(pady=(15, 0), anchor="w")

    def close_dialog(self):
        self.siril.disconnect()
        self.root.quit()
        self.root.destroy()

    def run_script(
        self,
        fitseq_mode: bool = False,
        do_spcc: bool = False,
        filter: str = "broadband",
        telescope: str = "ZWO Seestar S30",
        catalog: str = "localgaia",
        use_darks: bool = False,
        use_flats: bool = False,
        use_bias: bool = False,
        # graxpert: bool = False,
        bg_extract: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
    ):
        self.siril.log(
            f"Running script with arguments:\n"
            f"fitseq_mode={fitseq_mode}\n"
            f"do_spcc={do_spcc}\n"
            f"filter={filter}\n"
            f"telescope={telescope}\n"
            f"catalog={catalog}\n"
            f"use_darks={use_darks}\n"
            f"use_flats={use_flats}\n"
            f"use_bias={use_bias}\n"
            # f"graxpert={graxpert}\n"
            f"bg_extract={bg_extract}\n"
            f"drizzle={drizzle}\n"
            f"drizzle_amount={drizzle_amount}\n"
            f"pixel_fraction={pixel_fraction}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}",
            LogColor.BLUE,
        )
        self.fitseq_mode = fitseq_mode
        self.drizzle_status = drizzle
        self.convert_files("lights")
        if use_darks:
            self.convert_files("darks")
        if use_flats:
            self.convert_files("flats")
        if use_bias:
            self.convert_files("biases")

        # # TODO: Calibration frames processing

        seq_name = "lights" if fitseq_mode else "lights_"
        # if graxpert:
        #     self.seq_graxpert_bg_extract()
        #     seq_name = "gxbg_" + seq_name
        if bg_extract:
            self.seq_bg_extract(seq_name=seq_name)
            seq_name = "bkg_" + seq_name

        # Don't plate solve if 2048+ mode on, doesn't do anything but waste time
        if not self.fitseq_mode and telescope != "Evscope2":
            print(telescope)

            self.seq_plate_solve(seq_name=seq_name)
        # seq_name stays the same after plate solve
        self.seq_apply_reg(
            seq_name=seq_name,
            drizzle_amount=drizzle_amount,
            pixel_fraction=pixel_fraction,
        )
        seq_name = f"r_{seq_name}"

        if drizzle:
            self.scan_black_frames(seq_name=seq_name)

        self.seq_stack(
            seq_name=seq_name, feather=feather, feather_amount=feather_amount
        )
        self.load_image(image_name="result")

        # Force a plate solve if 2048+ mode so that SPCC works (if called)
        if self.fitseq_mode:
            self.image_plate_solve()

        self.save_image("_og")

        if do_spcc:
            self.spcc(
                oscsensor=telescope,
                filter=filter,
                catalog=catalog,
                whiteref="Average Spiral Galaxy",
            )

        self.autostretch(do_spcc=do_spcc)
        self.clean_up()
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

##############################################################################
