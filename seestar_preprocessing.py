import os
import sys
import shutil
import tkinter as tk
from tkinter import ttk
import sirilpy as s
from sirilpy import NoImageError, tksiril
from ttkthemes import ThemedTk
from astropy.io import fits

s.ensure_installed("ttkthemes")

APP_NAME = "Smart Telescope Preprocessing"
VERSION = "1.0.0"
AUTHOR = "Nazmus Nasir"
WEBSITE = "https://www.Naztronomy.com (https://www.YouTube.com/Naztronomy)"

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
        try:
            self.siril.connect()
            self.siril.log("Connected to Siril", self.siril.LogColor.GREEN)
        except s.SirilConnectionError:
            self.siril.log("Failed to connect to Siril", self.siril.LogColor.RED)   
            self.close_dialog()

        try:
            self.siril.cmd("requires", "1.3.6")
        except s.CommandError:
            self.close_dialog()
            return

        self.gaia_catalogue_available = False
        try:
            catalog_status = self.siril.get_siril_config("core", "catalogue_gaia_astro")
            if catalog_status and catalog_status != "(not set)" and os.path.isfile(catalog_status):
                self.gaia_catalogue_available = True

        except s.CommandError:
            pass
        self.current_working_directory = self.siril.get_siril_wd()
        lights_directory = os.path.join(self.current_working_directory, "lights")
        if not os.path.isdir(lights_directory):
            raise self.siril.error_messagebox("Directory 'lights' does not exist, please change current working directory and try again.", True)
        self.create_widgets()


    # Dirname: lights, darks, biases, flats
    def convert_files(self, dir_name):
        directory = os.path.join(self.current_working_directory, dir_name)
        if os.path.isdir(directory):
            self.siril.cmd("cd", dir_name)
            file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
            if file_count > 2048:
                self.siril.error_messagebox("More than 2048 images found. Please select '2048+ Mode' and try again.")
                self.close_dialog()

            try:
                if self.fitseq_mode:
                    if not self.drizzle_status:
                        self.siril.cmd("convert", dir_name, "-fitseq", "-debayer", "-out=../process")
                    else:
                        self.siril.cmd("convert", dir_name, "-fitseq", "-out=../process")
                else:
                    if not self.drizzle_status:
                        self.siril.cmd("convert", dir_name, "-debayer", "-out=../process")
                    else:
                        self.siril.cmd("convert", dir_name, "-out=../process")
            except s.CommandError as e:
                self.siril.error_messagebox(f"{e}")
                self.close_dialog()
            self.siril.cmd("cd", "../process")
            self.siril.log(f"Converted {file_count} files for processing!", self.siril.LogColor.GREEN)
        else:
            self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            raise NoImageError((f"No directory named \"{dir_name}\" at this location. Make sure the working directory is correct."))
            

    # Plate solve on sequence runs when file count < 2048
    def seq_plate_solve(self, seq_name):
        """Runs the siril command 'seqplatesolve' to plate solve the converted files."""
        self.siril.cmd(
            "seqplatesolve", seq_name, "-nocache", "-force", "-disto=ps_distortion"
        )
        self.siril.log(f"Platesolved {seq_name}", self.siril.LogColor.GREEN)

    def seq_graxpert_bg_extract(self):
        """Runs Graxpert to extract bg for each file in the sequence. Uses GPU if available."""
        self.siril.cmd("seqgraxpert_bg", "lights_", "-gpu")

    def seq_bg_extract(self):
        """Runs the siril command 'seqsubsky' to extract the background from the plate solved files."""
        self.siril.cmd("seqsubsky", "lights_", "1", "-samples=10")
        self.siril.log("Background extracted from Sequence", self.siril.LogColor.GREEN)

    def seq_apply_reg(self, seq_name, drizzle_amount, pixel_fraction):
        """Apply Existing Registration to the sequence. """
        if self.fitseq_mode:
            cmd_args = [
                "register",
                seq_name,
                "-2pass",
                "-selected"
            ]
            
        else:
            cmd_args = [
                "seqapplyreg",
                seq_name,
                "-filter-round=2.5k",
                "-kernel=square",
                "-framing=max",
            ]
        if self.drizzle_status:
            cmd_args.extend(
                ["-drizzle", f"-scale={drizzle_amount}", f"-pixfrac={pixel_fraction}"]
            )
        self.siril.cmd(*cmd_args)

        # Need the extra reg for fitseq
        if self.fitseq_mode:
            self.siril.cmd("seqapplyreg", seq_name,"-filter-round=2.5k")

        self.siril.log("Registered Sequence", self.siril.LogColor.GREEN)

    def seq_stack(self, seq_name, feather, feather_amount):
        """Stack it all, and feather if it's provided"""
        cmd_args = [
            "stack",
            f"{seq_name} rej 3 3",
            "-norm=addscale",
            "-output_norm",
            "-rgb_equal",
            "-maximize",
            "-out=result",
        ]
        if feather and feather_amount is not None:
            cmd_args.append(f"-feather={feather_amount}")
        self.siril.cmd(*cmd_args)

        self.siril.log("Complete stack!", self.siril.LogColor.GREEN)

    def save_image(self, suffix):
        """Saves the image as a FITS file."""
        file_name = f"$OBJECT:%s$_$STACKCNT:%d$x$EXPTIME:%d$sec_$DATE-OBS:dt${suffix}"
        self.siril.cmd(
            "save",
            f"../{file_name}",
        )
        self.siril.log(f"Saved file: {file_name}", self.siril.LogColor.GREEN)

    def load_registered_image(self):
        """Loads the registered image. Currently unused"""
        self.siril.cmd("load", "result")
        self.save_image("_og")

    def image_plate_solve(self):
        """Plate solve the loaded image with the '-force' argument."""
        self.siril.cmd("platesolve", "-force")
        self.siril.log("Platesolved image", self.siril.LogColor.GREEN)

    def spcc(self, oscsensor="ZWO Seestar S50", filter="broadband", catalog="localgaia", whiteref="Average Spiral Galaxy"):
        """SPCC with oscsensor, filter, catalog, and whiteref."""

        args = [
            f"-oscsensor={oscsensor}",
            f"-catalog={catalog}",
            f"-whiteref={whiteref}",
        ]
        
        if oscsensor in {"ZWO Seestar S30", "ZWO Seestar S50"}:
            if filter == "broadband":
                args.extend(
                    [
                        "-oscfilter=UV/IR Block",
                    ]
                )
            elif filter == "narrowband":
                args.extend(
                    [
                        "-oscfilter=ZWO Seestar LP",
                    ]
                )
        elif oscsensor == "Dwarf 3":
            if filter == "broadband":
                args.extend(
                    [
                        "-oscfilter=Sony IMX678",
                    ]
                )
            elif filter == "narrowband":
                args.extend(
                    [
                        "-narrowband",
                        "-rwl=656.28",
                        "-rbw=18",
                        "-gwl=500.70",
                        "-gbw=30",
                        "-bwl=500.70",
                        "-bbw=30",
                    ]
                )
        else:
            args.extend(
                    [
                        "-oscfilter=UV/IR Block",
                    ]
                )
            
        # Need to put double quotes around each argument, otherwise the spaces cause issues
        quoted_args = [f'"{arg}"' for arg in args]        
        self.siril.cmd("spcc", *quoted_args)

        self.save_image("_spcc")

        self.siril.log("SPCC'd image", self.siril.LogColor.GREEN)

    def load_image(self, image_name):
        """Loads the result."""
        self.siril.cmd("load", image_name)
        self.siril.log(f"Loaded image: {image_name}", self.siril.LogColor.GREEN)

    def autostretch(self):
        """Autostretch as a way to preview the final result"""
        self.siril.cmd("autostretch")
        self.siril.log("Autostretched image. You may want to open the _spcc file instead", self.siril.LogColor.GREEN)

    def clean_up(self, prefix):
        """Cleans up all files in the process directory."""
        self.siril.cmd("cd", "../")
        process_directory = os.path.join(self.current_working_directory, 'process')
        if os.path.isdir(process_directory):
            shutil.rmtree(process_directory)


    


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
        ttk.Label(main_frame, text=f"Author: {AUTHOR}", style="Bold.TLabel" ).pack(
            anchor="w", pady=(0, 0)
        )
        ttk.Label(
            main_frame, text=f"Website: {WEBSITE}", style="Bold.TLabel"
        ).pack(anchor="w", pady=(0, 0))

        ttk.Label(main_frame, text=f"Current Working Directory: {self.current_working_directory}").pack(
            anchor="w", pady=(0, 10)
        )

        # Telescope section
        telescope_section = ttk.LabelFrame(main_frame, text="Telescope", padding=10)
        telescope_section.pack(fill=tk.X, pady=5)
        ttk.Label(telescope_section, text="Telescope:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        telescope_variable = tk.StringVar(value="ZWO Seestar S30")
        telescope_options = ["ZWO Seestar S30", "ZWO Seestar S50", "Dwarf 3"]
        ttk.OptionMenu(
            telescope_section, telescope_variable, "ZWO Seestar S30", *telescope_options
        ).grid(row=0, column=1, sticky="w")

        # Mode to handle >2048 files
        telescope_section.pack(fill=tk.X, pady=5)

        ttk.Label(
            telescope_section, text="2048+ Files:", style="Bold.TLabel"
        ).grid(row=1, column=0, sticky="w")

        fitseq_checkbox_variable = tk.BooleanVar()

        fitseq_checkbox = ttk.Checkbutton(
            telescope_section,
            text="Enable",
            variable=fitseq_checkbox_variable
        )
        fitseq_checkbox.grid(row=1, column=1, sticky="w")

        # Add tooltip to the one and only checkbox
        tksiril.create_tooltip(
            fitseq_checkbox,
            "Enable this option if you have more than 2048 images to process. A different workflow will be used and the sequence will not be plate solved and the framing method will be default."
            "This can affect how large mosaic sessions look."
        )



        # Optional Preprocessing Steps

        # Optional Preprocessing Steps 
        calib_section = ttk.LabelFrame(
            main_frame, text="Optional Preprocessing Steps", padding=10
        )

        # Regular BG Extraction
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
        
        # Graxpert BG Extraction
        # calib_section.pack(fill=tk.X, pady=5)
        # ttk.Label(
        #     calib_section, text="Graxpert BG Extraction:", style="Bold.TLabel"
        # ).grid(row=2, column=0, sticky="w")

        # graxpert_checkbox_variable = tk.IntVar()
        # ttk.Checkbutton(
        #     calib_section,
        #     text="Background Extraction?",
        #     variable=graxpert_checkbox_variable,
        # ).grid(row=2, column=1, sticky="w")

        # Registering Frames
        calib_section.pack(fill=tk.X, pady=5)
        ttk.Label(calib_section, text="Registration:", style="Bold.TLabel").grid(
            row=3, column=0, sticky="w"
        )

        drizzle_checkbox_variable = tk.BooleanVar()
        ttk.Checkbutton(
            calib_section, text="Drizzle?", variable=drizzle_checkbox_variable
        ).grid(row=3, column=1, sticky="w")

        drizzle_amount_label = ttk.Label(calib_section, text="Drizzle amount:")
        drizzle_amount_label.grid(row=3, column=2, sticky="w")
        drizzle_amount_variable = tk.DoubleVar(value=1.0)
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
        pixel_fraction_variable = tk.DoubleVar(value=1.0)
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

        # Stacking Frames

        calib_section.pack(fill=tk.X, pady=5)
        ttk.Label(calib_section, text="Stacking:", style="Bold.TLabel").grid(
            row=5, column=0, sticky="w"
        )

        feather_checkbox_variable = tk.BooleanVar()
        ttk.Checkbutton(
            calib_section, text="Feather?", variable=feather_checkbox_variable
        ).grid(row=5, column=1, sticky="w")

        feather_amount_label = ttk.Label(calib_section, text="Feather amount:")
        feather_amount_label.grid(row=5, column=2, sticky="w")
        feather_amount_variable = tk.DoubleVar(value=5.0)
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
        spcc_section = ttk.LabelFrame(main_frame, text="Post-Stacking", padding=10)
        spcc_section.pack(fill=tk.X, pady=5)

        spcc_checkbox_variable = tk.BooleanVar()

        def toggle_filter_and_gaia():
            state = tk.NORMAL if spcc_checkbox_variable.get() else tk.DISABLED
            filter_menu["state"] = state
            catalog_menu["state"] = state

        ttk.Checkbutton(
            spcc_section,
            text="Enable Spectrophotometric Color Calibration (SPCC)",
            variable=spcc_checkbox_variable,
            command=toggle_filter_and_gaia,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(spcc_section, text="OSC Filter:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        filter_variable = tk.StringVar(value="broadband")
        filter_options = ["broadband", "narrowband"]
        filter_menu = ttk.OptionMenu(
            spcc_section, filter_variable, "broadband", *filter_options
        )
        filter_menu.grid(row=1, column=1, sticky="w")
        filter_menu["state"] = tk.DISABLED

        ttk.Label(spcc_section, text="Catalog:", style="Bold.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        catalog_variable = tk.StringVar(value="localgaia")
        catalog_options = ["localgaia", "gaia"]
        catalog_menu = ttk.OptionMenu(
            spcc_section, catalog_variable, "localgaia", *catalog_options
        )
        catalog_menu.grid(row=2, column=1, sticky="w")
        catalog_menu["state"] = tk.DISABLED

        if self.gaia_catalogue_available:
            ttk.Label(
                spcc_section, 
                text="✓ Local Gaia Available", 
                foreground="green", 
                style="Success.TLabel"
            ).grid(row=2, column=2, sticky="w")
        else:
            ttk.Label(
                spcc_section, 
                text="✗ Local Gaia Not available", 
                foreground="red"
            ).grid(row=2, column=2, sticky="w")
        

        # Run button
        ttk.Button(
            main_frame,
            text="Run",
            command=lambda: self.run_script(
                fitseq_mode=fitseq_checkbox_variable.get(),
                do_spcc=spcc_checkbox_variable.get(),
                filter=filter_variable.get(),
                telescope=telescope_variable.get(),
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
        drizzle_amount: float = 1.0,
        pixel_fraction: float = 1.0,
        feather: bool = False,
        feather_amount: float = 5.0,
    ):
        self.fitseq_mode = fitseq_mode
        self.drizzle_status = drizzle
        self.convert_files("lights")
        if use_darks:
            self.convert_files("darks")
        if use_flats:
            self.convert_files("flats")
        if use_bias:
            self.convert_files("biases")

        # TODO: Calibration frames processing 

        seq_name = "lights" if fitseq_mode else "lights_"
        # if graxpert:
        #     self.seq_graxpert_bg_extract()
        #     seq_name = "gxbg_" + seq_name
        if bg_extract:
            self.seq_bg_extract()
            seq_name = "bkg_" + seq_name
        
        # Don't plate solve if mosaic mode on, doesn't do anything but waste time
        if not self.fitseq_mode:
            self.seq_plate_solve(seq_name=seq_name)
        # seq_name stays the same after plate solve
        self.seq_apply_reg(
            seq_name=seq_name,
            drizzle_amount=drizzle_amount,
            pixel_fraction=pixel_fraction,
        )
        seq_name = f"r_{seq_name}"
        self.seq_stack(
            seq_name=seq_name, feather=feather, feather_amount=feather_amount
        )
        self.load_image(image_name="result")

        self.save_image("_og")

        if do_spcc:
            self.spcc(oscsensor=telescope, filter=filter, catalog=catalog, whiteref="Average Spiral Galaxy")
        
        self.autostretch()
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
