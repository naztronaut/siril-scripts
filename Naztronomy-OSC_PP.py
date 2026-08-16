"""
(c) Nazmus Nasir 2025
SPDX-License-Identifier: GPL-3.0-or-later

Naztronomy - OSC Image Preprocessing script
Version: 2.0.5
=====================================

The author of this script is Nazmus Nasir (Naztronomy) and can be reached at:
https://www.Naztronomy.com or https://www.YouTube.com/Naztronomy
Join discord for support and discussion: https://discord.gg/yXKqrawpjr
Support me on Patreon: https://www.patreon.com/c/naztronomy
Support me on Buy me a Coffee: https://www.buymeacoffee.com/naztronomy

This script is designed to process OSC images only at this time. An experimental monochrome feature is available in this script, however
there are no guarantees.

If your images have the correct headers (RA/DEC coordinates, focal length, pixel size, etc.), this script can automatically
plate solve and stitch mosaics. If you are using data without the correct headers, it will do a star alignment on a reference frame (.e.g no mosaics).

This script can be run from any directory but recommended to create a blank directory.

All images are currently copied before processed so it can take up some disk space. This is to mitigate systems that don't allow symlinks. This also
allows you to choose files from any folder and drive and they will all be consolidated into a single location.

"""

"""
CHANGELOG:

2.0.4 - Discard rejected and bad files coming from NINA
      - Files tab: file list is now a sortable, grouped tree (#, Filter, Object,
        File name) with per-type counts in the group title. Ported from Mono PP.
      - Export Sessions to CSV: writes an AstroBin acquisition CSV (date, filter,
        frame counts, exposure, gain, binning, cooling, f-number) with a Bortle
        prompt applied to every row.
      - Master Config: configure a reusable master darks folder and a single
        master bias file, remembered between runs in the shared scripts config.
      - Import Masters: apply configured master darks (matched by exposure time
        and temperature) and the master bias to all open sessions.
      - Reset Calibration: remove darks/flats/biases from chosen sessions
        (per-session selection when multiple sessions exist); lights untouched.
      - Filter settings: added "abs" (absolute value) mode alongside sigma and % for
        Roundness, FWHM, Star Count and Background filters. When "abs" is chosen
        the raw value is passed to Siril with no suffix (issue #99).
      - allow astrometry.net fallback (PR#107 @tophrchris)
2.0.3 - Files tab UI overhaul
      - Drag and drop files or folders directly onto the file list
      - Folder drop: named folders (lights/darks/flats/biases/dark flats) auto-detected
      - Folder drop: parent directory containing named subfolders also accepted
      - Stacked calibration folders (darks_stacked, flats_stacked, biases_stacked) also accepted
      - Stacked folder drop: prompts current vs all sessions when multiple sessions exist
      - Frame type selection dialog on drop (Lights, Darks, Flats, Biases)
      - Master calibration frame support via drag & drop (Master Dark, Master Flat, Master Bias)
      - Master frames can be applied to the current session, selected sessions (checkboxes), or all sessions
      - File list rows color-coded by frame type (green=Lights, blue=Darks, amber=Flats, purple=Biases)
      - Custom item delegate: visible selection highlight and hover effect on file list rows
      - Green circle indicator on Add buttons when that frame type has files loaded
      - Empty-state placeholder text on the file list
      - Drag-hover border highlight on the file list drop zone
      - Files tab split into two group boxes: Session Management (top) and Files in Session N (bottom)
      - Session content group box title updates dynamically to show current session number
      - Next / Back single toggle button for navigating between tabs (styled white with dark border)
2.0.2 - Bug fixes - bias and BGE colliding
2.0.1 - Single/Multi/Paneled mosaic workflows 
      - Allow stacking multiple targets at the same time (without combining them at the end)
      - Single target session can combine everything at once or do it by session/panel
      - Paneled mosaic will crop down to reference frame size to avoid noisy edges and speed up processing
      - Paneled mosaic automatically applies overlap normalization 
      - Final Stack Checkbox
      - Delay between stacks to reduce IO errors
2.0.0 - pyqt6 support
      - Save/Load presets
      - Monochrome support (experimental)
      - Improved session management
1.0.0 - initial release
      - Supports both Mosaics and star alignment for images without proper headers
      - Cleans up all intermediate files BUT keeps all preprocessed lights so they can be combined later
"""


from pathlib import Path
import shutil
import sirilpy as s

# Themes requirements: qt_themes, pyside6, qtpy
s.ensure_installed("PyQt6", "numpy", "astropy", "qt_themes", "pyside6", "qtpy")


from PyQt6.QtCore import Qt, QUrl
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
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QLineEdit,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
    QToolButton,
    QMenu,
    QDialog,
    QTextBrowser,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtGui import (
    QFont,
    QShortcut,
    QKeySequence,
    QAction,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QPainter,
    QColor,
    QBrush,
)
from datetime import datetime, timedelta
import csv
import re
import time
import os
import sys
import json
import qt_themes
from sirilpy import LogColor, NoImageError
from astropy.io import fits
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

APP_NAME = "Naztronomy - OSC Image Preprocessor"
VERSION = "2.0.5"
BUILD = "20260430"
AUTHOR = "Nazmus Nasir"
WEBSITE = "https://www.Naztronomy.com"
YOUTUBE = "https://www.YouTube.com/Naztronomy"
DISCORD = "https://discord.gg/yXKqrawpjr"
PATREON = "https://www.patreon.com/c/naztronomy"
BUY_ME_A_COFFEE = "https://www.buymeacoffee.com/naztronomy"


UI_DEFAULTS = {
    "feather_amount": 20,
    "filter_round": 3.0,
    "filter_wfwhm": 3.0,
    "filter_stars": 3.0,
    "filter_bkg": 3.0,
    "drizzle_amount": 1.0,
    "pixel_fraction": 1.0,
}

# Controls how files are staged into the sessions/ folder before processing.
# False (default): symlink files on the same drive, copy files from a different drive.
#   Cross-drive symlinks cause Siril/libraw to read through untrusted mounts and
#   fail with I/O errors (WinError 448), so they are always copied.
# True: always symlink regardless of drive. Use only if your OS / mount allows it
#   and you want to avoid the disk copy (e.g. large raw files on a trusted NAS).
ALWAYS_SYMLINK: bool = False
FRAME_TYPES = ("lights", "darks", "flats", "biases")

# Accepted input frame extensions for reading metadata / matching masters.
FITS_INPUT_EXTENSIONS = (".fit", ".fits", ".fit.fz", ".fits.fz")
XISF_INPUT_EXTENSIONS = (".xisf",)


def _is_fits_file(path) -> bool:
    """Return True if `path` looks like a FITS file by extension."""
    name = str(path).lower()
    return any(name.endswith(ext) for ext in FITS_INPUT_EXTENSIONS)


def _is_xisf_file(path) -> bool:
    """Return True if `path` looks like an XISF file by extension."""
    name = str(path).lower()
    return any(name.endswith(ext) for ext in XISF_INPUT_EXTENSIONS)


def _is_supported_input(path) -> bool:
    """Return True if `path` is an accepted input frame (FITS or XISF)."""
    return _is_fits_file(path) or _is_xisf_file(path)


def _is_rejected_file(path) -> bool:
    """Return True if a dropped file should be skipped when adding to a session.

    Files whose name starts with 'bad_' or that live inside a folder named
    'rejected' (both case-insensitive) are ignored.
    """
    p = Path(path)
    if p.name.lower().startswith("bad_"):
        return True
    return any(part.lower() == "rejected" for part in p.parts)


# ── Exposure / temperature detection for master-dark matching ──────────────
# Master darks are matched to a session by exposure time (must agree) and sensor
# temperature (nearest within a tolerance). FITS values come from headers; XISF
# masters (which astropy cannot open) are parsed from the file name instead.
MASTER_DARK_TEMP_TOLERANCE: float = 3.0
MASTER_DARK_EXP_TOLERANCE: float = 1.0

_EXPTIME_NAME_RE = re.compile(
    r"exp(?:osure|time)?[\s_\-]*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)?",
    re.IGNORECASE,
)
_EXPTIME_NAME_RE_GENERIC = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|seconds)\b",
    re.IGNORECASE,
)
_TEMP_NAME_RE = re.compile(
    r"(?:ccd[\s_\-]*temp|temp)[\s_=:\-]?(-?[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TEMP_NAME_RE_GENERIC = re.compile(
    r"(-?[0-9]+(?:\.[0-9]+)?)\s*(?:deg)?\s*c(?![a-z])",
    re.IGNORECASE,
)


def _read_fits_header_dict(path) -> dict:
    """Read a flattened FITS header (first value wins per key), or {} on error."""
    header: dict = {}
    try:
        with fits.open(str(path)) as hdul:
            for hdu in hdul:
                hdr = getattr(hdu, "header", None)
                if hdr is None:
                    continue
                for key in hdr.keys():
                    if key and key not in header:
                        header[key] = hdr.get(key)
    except Exception:
        return {}
    return header


def _read_fits_value(path, *keys):
    """Return the first present value among `keys` from a FITS header, or None."""
    header = _read_fits_header_dict(path)
    for key in keys:
        val = header.get(key)
        if val not in (None, ""):
            return val
    return None


def _read_fits_object(path):
    """Return the OBJECT header value from a FITS file, or None."""
    if not _is_fits_file(path):
        return None
    val = _read_fits_value(path, "OBJECT")
    return str(val).strip() if val not in (None, "") else None


def _read_fits_filter(path):
    """Return the FILTER header value from a FITS file, or None."""
    if not _is_fits_file(path):
        return None
    val = _read_fits_value(path, "FILTER")
    return str(val).strip() if val not in (None, "") else None


def _parse_exptime_from_name(name) -> float | None:
    """Parse an exposure time (seconds) from a file name, or None."""
    stem = Path(name).stem
    match = _EXPTIME_NAME_RE.search(stem) or _EXPTIME_NAME_RE_GENERIC.search(stem)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _parse_temp_from_name(name) -> float | None:
    """Parse a sensor temperature (°C) from a file name, or None."""
    stem = Path(name).stem
    match = _TEMP_NAME_RE.search(stem) or _TEMP_NAME_RE_GENERIC.search(stem)
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _get_exptime_temp(path) -> tuple[float | None, float | None]:
    """Return (exposure_seconds, temperature_C) for a frame.

    FITS frames are read from their headers, falling back to the file name for
    any missing value. XISF frames are parsed from the file name only.
    """
    if _is_fits_file(path):
        exptime = _read_fits_value(path, "EXPTIME", "EXPOSURE")
        temp = _read_fits_value(path, "CCD-TEMP", "CCD_TEMP", "CCDTEMP", "TEMP")
        try:
            exptime = float(exptime) if exptime not in (None, "") else None
        except (TypeError, ValueError):
            exptime = None
        try:
            temp = float(temp) if temp not in (None, "") else None
        except (TypeError, ValueError):
            temp = None
        if exptime is None:
            exptime = _parse_exptime_from_name(path)
        if temp is None:
            temp = _parse_temp_from_name(path)
        return exptime, temp
    return _parse_exptime_from_name(path), _parse_temp_from_name(path)


def _parse_dateobs_night(value) -> str | None:
    """Convert a FITS DATE-OBS value to the acquisition-night date (YYYY-MM-DD).

    DATE-OBS is a UTC timestamp at the start of the exposure. Subtracting 12
    hours before taking the date rolls after-midnight frames back into the night
    the session started (the convention AstroBin expects). Returns None when the
    value cannot be parsed.
    """
    if value in (None, ""):
        return None
    text = str(value).strip().rstrip("Z")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None
    return (dt - timedelta(hours=12)).date().isoformat()


def _read_light_metadata(path) -> dict:
    """Read AstroBin-relevant metadata from a single light frame's FITS header.

    Returns a dict with any of: date, filterName, binning, gain, sensorCooling,
    fNumber, duration. Missing values are omitted.
    """
    meta: dict = {}
    exptime, temp = _get_exptime_temp(path)
    if exptime is not None:
        meta["duration"] = exptime
    if temp is not None:
        meta["sensorCooling"] = temp

    if not _is_fits_file(path):
        return meta

    header = _read_fits_header_dict(path)
    if not header:
        return meta

    def _first(*keys):
        for key in keys:
            val = header.get(key)
            if val not in (None, ""):
                return val
        return None

    night = _parse_dateobs_night(_first("DATE-OBS", "DATE_OBS"))
    if night:
        meta["date"] = night

    filter_name = _first("FILTER")
    if filter_name not in (None, ""):
        meta["filterName"] = str(filter_name).strip()

    binning = _first("XBINNING", "BINNING", "BINX")
    if binning not in (None, ""):
        try:
            meta["binning"] = int(float(binning))
        except (TypeError, ValueError):
            pass

    gain = _first("GAIN", "EGAIN")
    if gain not in (None, ""):
        try:
            meta["gain"] = float(gain)
        except (TypeError, ValueError):
            pass

    if "sensorCooling" not in meta:
        cooling = _first("CCD-TEMP", "CCD_TEMP", "CCDTEMP", "SET-TEMP", "TEMP")
        if cooling not in (None, ""):
            try:
                meta["sensorCooling"] = float(cooling)
            except (TypeError, ValueError):
                pass

    fnumber = _first("FOCRATIO", "FNUMBER", "F-RATIO")
    if fnumber not in (None, ""):
        try:
            meta["fNumber"] = float(fnumber)
        except (TypeError, ValueError):
            pass

    if "duration" not in meta:
        dur = _first("EXPTIME", "EXPOSURE")
        if dur not in (None, ""):
            try:
                meta["duration"] = float(dur)
            except (TypeError, ValueError):
                pass

    return meta


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


class SortableTreeItem(QTreeWidgetItem):
    """Tree item with type-aware sorting.

    The "#" column (column 0) sorts numerically rather than as text. Group-header
    rows (those with ``group_order`` set) always keep their fixed insertion order
    so the Lights/Darks/Flats/Biases sections never shuffle, no matter which
    column the user sorts by.
    """

    group_order = None  # set on group-header rows to pin their position

    def __lt__(self, other):
        if (
            self.group_order is not None
            and getattr(other, "group_order", None) is not None
        ):
            return self.group_order < other.group_order
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        if column == 0:
            return self._as_int(self.text(0)) < self._as_int(other.text(0))
        return self.text(column).lower() < other.text(column).lower()

    @staticmethod
    def _as_int(text):
        try:
            return int(text)
        except (TypeError, ValueError):
            return 0


class FileTypeDialog(QDialog):
    """Prompt the user to choose a frame type for drag-and-dropped files."""

    FRAME_TYPES = ["Lights", "Darks", "Flats", "Biases"]
    MASTER_TYPES = ["Master Dark", "Master Flat", "Master Bias"]

    def __init__(
        self,
        file_count: int,
        current_session_name: str = "Current Session",
        all_session_names: list | None = None,
        current_session_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._current_session_name = current_session_name
        self._all_session_names = all_session_names or [current_session_name]
        self._current_session_index = current_session_index
        self.setWindowTitle("Select Frame Type")
        self.setModal(True)
        self.setMinimumWidth(320)
        self.chosen_type: str | None = None
        self.chosen_scope: str = "current"  # "current", "selected", or "all"
        self.chosen_session_indices: list = [current_session_index]

        layout = QVBoxLayout(self)
        plural = "s" if file_count != 1 else ""
        layout.addWidget(
            QLabel(
                f"You dropped {file_count} file{plural}.\nWhat type of frames are these?"
            )
        )

        for frame_type in self.FRAME_TYPES:
            btn = QPushButton(frame_type)
            btn.setMinimumHeight(50)
            btn.setMinimumWidth(100)
            if frame_type.lower() == "lights":
                btn.clicked.connect(lambda checked, t=frame_type: self._select(t))
            else:
                btn.clicked.connect(
                    lambda checked, t=frame_type: self._select_master(t)
                )
            layout.addWidget(btn)

        sep = QLabel("— Master calibration frames —")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px; margin-top: 4px;"
        )
        layout.addWidget(sep)

        for master_type in self.MASTER_TYPES:
            btn = QPushButton(master_type)
            btn.setMinimumHeight(50)
            btn.setMinimumWidth(100)
            btn.setStyleSheet(
                "QPushButton { background-color: #e8f4e8; color: #1d4e2d; }"
                " QPushButton:hover { background-color: #c3e6cb; }"
            )
            btn.clicked.connect(lambda checked, t=master_type: self._select_master(t))
            layout.addWidget(btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def _select(self, frame_type: str):
        self.chosen_type = frame_type
        self.chosen_scope = "current"
        self.accept()

    def _select_master(self, frame_type: str):
        if len(self._all_session_names) <= 1:
            self.chosen_type = frame_type
            self.chosen_scope = "current"
            self.accept()
            return

        sub = QDialog(self)
        sub.setWindowTitle("Apply to which sessions?")
        sub.setModal(True)
        sub_layout = QVBoxLayout(sub)

        sub_layout.addWidget(QLabel(f"Add <b>{frame_type}</b> to:"))

        checkboxes: list[QCheckBox] = []
        for i, name in enumerate(self._all_session_names):
            cb = QCheckBox(name)
            cb.setChecked(i == self._current_session_index)
            checkboxes.append(cb)
            sub_layout.addWidget(cb)

        btn_row = QHBoxLayout()
        selected_btn = QPushButton("Selected Sessions")
        all_btn = QPushButton("All Sessions")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(selected_btn)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(cancel_btn)
        sub_layout.addLayout(btn_row)

        result = {"action": None}

        def on_selected():
            result["action"] = "selected"
            sub.accept()

        def on_all():
            result["action"] = "all"
            sub.accept()

        selected_btn.clicked.connect(on_selected)
        all_btn.clicked.connect(on_all)
        cancel_btn.clicked.connect(sub.reject)

        if sub.exec() != QDialog.DialogCode.Accepted:
            return  # stay in outer dialog

        if result["action"] == "all":
            self.chosen_type = frame_type
            self.chosen_scope = "all"
            self.chosen_session_indices = list(range(len(self._all_session_names)))
            self.accept()
        elif result["action"] == "selected":
            indices = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
            if not indices:
                return  # nothing checked — stay in dialog
            self.chosen_type = frame_type
            self.chosen_scope = "selected"
            self.chosen_session_indices = indices
            self.accept()
        # else: cancel — stay in outer dialog


class DragDropTreeWidget(QTreeWidget):
    """A QTreeWidget (multi-column) that accepts file drag-and-drop and emits the dropped paths."""

    _NORMAL_STYLE = ""
    _HOVER_STYLE = (
        "QTreeWidget { border: 2px dashed #2563eb;"
        " background-color: rgba(37, 99, 235, 0.07); }"
    )

    def __init__(self, on_drop_callback, on_delete_callback=None, parent=None):
        super().__init__(parent)
        self._on_drop = on_drop_callback
        self._on_delete = on_delete_callback
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    def keyPressEvent(self, event):
        if (
            self._on_delete is not None
            and self.selectedItems()
            and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        ):
            self._on_delete()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.topLevelItemCount() == 0:
            painter = QPainter(self.viewport())
            painter.save()
            pen_color = self.palette().color(self.palette().ColorRole.PlaceholderText)
            painter.setPen(pen_color)
            font = painter.font()
            font.setPointSize(9)
            font.setItalic(True)
            painter.setFont(font)
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drop files or folders here\nor use the Add buttons above",
            )
            painter.restore()

    def dragEnterEvent(self, event: QDragEnterEvent | None):
        if event is None:
            return
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            self.setStyleSheet(self._HOVER_STYLE)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event is None:
            return
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self._NORMAL_STYLE)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent | None):
        self.setStyleSheet(self._NORMAL_STYLE)
        if event is None:
            return
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            event.ignore()
            return
        paths = [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self._on_drop(paths)
        else:
            event.ignore()


class PreprocessingInterface(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        self.initialization_successful = False

        self.siril = s.SirilInterface()

        # if drizzle is off, images will be debayered on convert
        self.drizzle_status = False
        self.drizzle_factor = 0

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
        self.astrometry_net_available = self.local_astrometry_net_available()
        # home directory is unchanged
        self.home_directory = self.siril.get_siril_wd()
        self.current_working_directory = self.siril.get_siril_wd()
        self.cwd_label = self.current_working_directory

        # Assigns collected_lights directory to store all pp_lights files
        self.collected_lights_dir = os.path.join(
            self.current_working_directory, "collected_lights"
        )

        # Sessions
        self.sessions = self.create_sessions(1)  # Start with one session
        self.chosen_session = self.sessions[0]

        # Reusable master calibration config (set via the Master Config dialog).
        # master_darks_dir: folder of master darks matched to sessions by
        # exposure + temperature. master_bias_path: single master bias applied
        # to every session. Persisted to a shared JSON in the Siril config dir.
        self.master_darks_dir: str | None = None
        self.master_bias_path: str | None = None
        self._last_bortle: str = ""
        self._load_master_config()

        self.session_dropdown = QComboBox()
        # self.update_dropdown()  # Fill it with sessions
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

        self.chosen_session = self.get_session_by_index(index)
        self.current_session = f"Session {index+1}"
        if hasattr(self, "file_listbox"):
            self.refresh_file_list()

    def add_dropdown_session(self):
        self.add_session(Session())
        self.update_dropdown()
        new_index = len(self.sessions) - 1
        self.session_dropdown.setCurrentIndex(new_index)  # selects new session
        self.chosen_session = self.sessions[new_index]
        self.current_session = f"Session {new_index+1}"
        self.refresh_file_list()
        self.update_process_separately_checkbox()  # Update checkbox state

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
        self.update_process_separately_checkbox()  # Update checkbox state

    def update_dropdown(self):
        session_names = [f"Session {i+1}" for i in range(len(self.sessions))]
        self.session_dropdown.clear()  # remove old items
        self.session_dropdown.addItems(session_names)  # add new items

    def _scan_folder_for_sessions(self, folder: Path):
        """Recursively scan `folder` for recognised frame-type subdirectories.

        Recognised directories (lights/darks/flats/biases etc.) that contain
        actual files are treated as leaf nodes — their files are collected and
        the scanner does NOT descend into them further.

        Recognised directories that contain NO direct files are treated as
        passthrough containers and recursed into.  This handles structures like:

            parent/lights/night_1/LIGHT/<files>
            parent/lights/night_2/LIGHT/<files>

        where `lights/` exists at an intermediate level.

        Unrecognised directories are treated as potential session boundaries.

        Returns:
            ("single", {frame_type: [files]})           – one group found
            ("multi",  {Path: {frame_type: [files]}})   – multiple groups found
            ("none",   {})                              – nothing recognised found

        Frame types prefixed with "_stacked_" come from *_stacked folder names.
        """
        _FT_MAP = {
            "lights": "lights",
            "light": "lights",
            "darks": "darks",
            "dark": "darks",
            "flats": "flats",
            "flat": "flats",
            "biases": "biases",
            "bias": "biases",
            "dark flats": "biases",
            "dark_flats": "biases",
            "dark flat": "biases",
            "dark_flat": "biases",
            "darkflat": "biases",
        }
        _ST_MAP = {
            "darks_stacked": "darks",
            "flats_stacked": "flats",
            "biases_stacked": "biases",
        }

        def recurse(d: Path, group_key: Path | None) -> dict:
            """Return {group_key_path: {frame_type: [files]}} for all discovered groups.

            For each child directory of d:
            - Recognised + has files  → leaf: collect files, stop descending into it
            - Recognised + no files   → passthrough: recurse into it keeping group_key
            - Unrecognised            → session boundary: recurse with child as new group_key
            """
            collected: dict[str, list[Path]] = {}  # frame-type matches at this level
            sub_sessions: dict[Path, dict] = {}  # session groups from deeper recursion

            for child in sorted(d.iterdir()):
                if not child.is_dir():
                    continue
                if child.name.lower() == "rejected":
                    continue
                child_key = child.name.lower()

                if child_key in _FT_MAP or child_key in _ST_MAP:
                    files = sorted(
                        f
                        for f in child.iterdir()
                        if f.is_file() and not _is_rejected_file(f)
                    )
                    if files:
                        # Leaf: recognised dir with direct files
                        if child_key in _FT_MAP:
                            collected.setdefault(_FT_MAP[child_key], []).extend(files)
                        else:
                            collected.setdefault(
                                f"_stacked_{_ST_MAP[child_key]}", []
                            ).extend(files)
                    else:
                        # Passthrough: recognised dir with no direct files — recurse keeping group_key
                        for k, v in recurse(child, group_key).items():
                            for ft, flist in v.items():
                                sub_sessions.setdefault(k, {}).setdefault(
                                    ft, []
                                ).extend(flist)
                else:
                    # Unrecognised dir — potential session boundary
                    new_group = child if group_key is None else group_key
                    for k, v in recurse(child, new_group).items():
                        for ft, flist in v.items():
                            sub_sessions.setdefault(k, {}).setdefault(ft, []).extend(
                                flist
                            )

            if collected:
                gkey = group_key if group_key is not None else d
                result = {gkey: collected}
                for k, v in sub_sessions.items():
                    result.setdefault(k, {}).update(
                        {ft: flist for ft, flist in v.items()}
                    )
                return result

            return sub_sessions

        all_sessions = recurse(folder, None)
        if not all_sessions:
            return "none", {}
        if len(all_sessions) == 1:
            return "single", list(all_sessions.values())[0]
        return "multi", all_sessions

    def _handle_multi_session_drop(self, root_folder: Path | None, session_data: dict):
        """Show a confirmation dialog then create new sessions from a multi-group folder drop.

        root_folder is used for display only; pass None when data comes from
        multiple dropped parent directories.
        """
        groups = sorted(session_data.keys(), key=lambda p: p.name)

        # Determine if any two groups share the same final folder name.
        # When they do, prepend the nearest non-recognised ancestor for disambiguation.
        _RECOGNISED_NAMES = {
            "lights",
            "light",
            "darks",
            "dark",
            "flats",
            "flat",
            "biases",
            "bias",
            "dark flats",
            "dark_flats",
            "dark flat",
            "dark_flat",
            "darkflat",
        }

        def _ancestor_label(p: Path) -> str:
            """Walk up past recognised intermediate dirs to find a meaningful ancestor name."""
            candidate = p.parent
            while candidate != candidate.parent:
                if candidate.name.lower() not in _RECOGNISED_NAMES:
                    return candidate.name
                candidate = candidate.parent
            return ""

        group_names = [g.name for g in groups]
        has_duplicates = len(group_names) != len(set(group_names))

        def _display_label(g: Path) -> str:
            if not has_duplicates:
                return g.name
            ancestor = _ancestor_label(g)
            return f"{ancestor} / {g.name}" if ancestor else g.name

        summary_parts = []
        for g in groups:
            parts = []
            for ft, files in sorted(session_data[g].items()):
                display = (
                    (ft[len("_stacked_") :] + " (stacked)")
                    if ft.startswith("_stacked_")
                    else ft
                )
                parts.append(f"{display}: {len(files)}")
            summary_parts.append(
                f"<li><b>{_display_label(g)}</b> &mdash; {', '.join(parts)}</li>"
            )

        if root_folder is not None:
            header = (
                f"Found <b>{len(groups)}</b> session group(s) in "
                f"<b>{root_folder.name}/</b>:<br>"
            )
        else:
            header = f"Found <b>{len(groups)}</b> session group(s) across dropped folders:<br>"

        msg = QMessageBox(self)
        msg.setWindowTitle("Multi-Session Folder Detected")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(
            f"{header}"
            f"<ul>{''.join(summary_parts)}</ul>"
            f"Create <b>{len(groups)}</b> new session(s)?"
        )
        btn_create = msg.addButton(
            f"Create {len(groups)} Session(s)", QMessageBox.ButtonRole.AcceptRole
        )
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        if msg.clickedButton() is not btn_create:
            return

        # If the current session is completely empty, reuse it for the first group
        # instead of leaving it as a blank Session 1.
        first_group, *rest_groups = groups
        current_is_empty = (
            not self.chosen_session.lights
            and not self.chosen_session.darks
            and not self.chosen_session.flats
            and not self.chosen_session.biases
        )

        def _populate(session: Session, group_dir: Path):
            for ft, ft_files in session_data[group_dir].items():
                real_ft = ft[len("_stacked_") :] if ft.startswith("_stacked_") else ft
                match real_ft:
                    case "lights":
                        session.lights.extend(ft_files)
                    case "darks":
                        session.darks.extend(ft_files)
                    case "flats":
                        session.flats.extend(ft_files)
                    case "biases":
                        session.biases.extend(ft_files)

        if current_is_empty:
            _populate(self.chosen_session, first_group)
            self.siril.log(
                f"> Loaded '{first_group.name}' into {self.session_dropdown.currentText()}",
                LogColor.BLUE,
            )
            remaining = rest_groups
        else:
            remaining = groups

        for group_dir in remaining:
            new_session = Session()
            self.sessions.append(new_session)
            _populate(new_session, group_dir)
            self.siril.log(
                f"> Created Session {len(self.sessions)} from '{group_dir.name}'",
                LogColor.BLUE,
            )

        self.update_dropdown()
        new_idx = len(self.sessions) - 1
        self.session_dropdown.setCurrentIndex(new_idx)
        self.chosen_session = self.sessions[new_idx]
        self.current_session = f"Session {new_idx + 1}"
        self.update_process_separately_checkbox()

    def _handle_dropped_files(self, paths: list):
        """Called by DragDropTreeWidget when files or folders are dropped onto the list.

        Folders named lights/darks/flats/biases (or dark flats → biases) are
        recognised and their contents are added to the *current session* directly,
        bypassing the frame-type dialog.

        A parent folder is scanned recursively (stopping at recognised directories)
        to find frame-type subfolders.  If one group is found it is added to the
        current session; if multiple groups are found the user is offered the option
        to create a new session for each group.

        Individual files whose names contain _darks_stacked, _flats_stacked, or
        _biases_stacked are auto-categorised into their respective frame types.
        When more than one session exists the user is asked whether to apply them
        to the current session only or to all sessions.

        All other plain files still trigger the frame-type dialog.
        """
        _FOLDER_TYPE_MAP = {
            "lights": "lights",
            "light": "lights",
            "darks": "darks",
            "dark": "darks",
            "flats": "flats",
            "flat": "flats",
            "biases": "biases",
            "bias": "biases",
            "dark flats": "biases",
            "dark_flats": "biases",
            "dark flat": "biases",
            "dark_flat": "biases",
            "darkflat": "biases",
        }
        _STACKED_FOLDER_MAP = {
            "darks_stacked": "darks",
            "flats_stacked": "flats",
            "biases_stacked": "biases",
        }

        # Drop 'bad_' files and anything under a 'rejected' folder before staging.
        paths = [p for p in paths if not _is_rejected_file(p)]
        folders = [p for p in paths if Path(p).is_dir()]
        files = [p for p in paths if Path(p).is_file()]

        # ── Process folders ──────────────────────────────────────────────────
        # Regular named folders → always current session
        folder_additions: dict[str, list[Path]] = {}
        # Stacked calibration folders → ask current vs all when >1 session
        stacked_additions: dict[str, list[Path]] = {}
        # Accumulated multi-session groups from all scanned folders — one dialog at end
        pending_multi_sessions: dict[Path, dict] = {}
        # "single" scan results keyed by the dropped folder — resolved after the loop
        pending_single_sessions: dict[Path, dict] = {}

        for folder in folders:
            folder = Path(folder)
            folder_key = folder.name.lower()
            if folder_key in _FOLDER_TYPE_MAP:
                frame_type = _FOLDER_TYPE_MAP[folder_key]
                folder_files = sorted(
                    f
                    for f in folder.iterdir()
                    if f.is_file() and not _is_rejected_file(f)
                )
                if folder_files:
                    folder_additions.setdefault(frame_type, []).extend(folder_files)
                else:
                    # No direct files — the recognised folder may contain session subdirs
                    mode, scan_data = self._scan_folder_for_sessions(folder)
                    if mode == "single":
                        pending_single_sessions[folder] = scan_data
                    elif mode == "multi":
                        pending_multi_sessions.update(scan_data)
                    else:
                        self.siril.log(
                            f"> Folder '{folder.name}' is empty and contains no recognised subfolders. Skipping.",
                            LogColor.SALMON,
                        )
            elif folder_key in _STACKED_FOLDER_MAP:
                frame_type = _STACKED_FOLDER_MAP[folder_key]
                folder_files = sorted(
                    f
                    for f in folder.iterdir()
                    if f.is_file() and not _is_rejected_file(f)
                )
                if folder_files:
                    stacked_additions.setdefault(frame_type, []).extend(folder_files)
            else:
                # Scan recursively for recognised frame-type subdirectories
                mode, scan_data = self._scan_folder_for_sessions(folder)
                if mode == "none":
                    self.siril.log(
                        f"> Folder '{folder.name}' does not contain recognised "
                        f"subfolders (lights, darks, flats, biases, dark_flats, "
                        f"darks_stacked, flats_stacked, biases_stacked). Skipping.",
                        LogColor.SALMON,
                    )
                elif mode == "single":
                    # Track per-folder — resolved after the loop
                    pending_single_sessions[folder] = scan_data
                elif mode == "multi":
                    # Accumulate — will show one combined dialog after the loop
                    pending_multi_sessions.update(scan_data)

        # Resolve pending "single" scan results:
        # - Exactly one, no multi pending → add directly to the current session
        # - More than one, or mixed with multi → treat each folder as its own session
        if len(pending_single_sessions) == 1 and not pending_multi_sessions:
            for ft_map in pending_single_sessions.values():
                for ft, ft_files in ft_map.items():
                    if ft.startswith("_stacked_"):
                        stacked_additions.setdefault(ft[len("_stacked_") :], []).extend(
                            ft_files
                        )
                    else:
                        folder_additions.setdefault(ft, []).extend(ft_files)
        elif pending_single_sessions:
            # Multiple single-session folders dropped — promote each to its own session group
            for src_folder, ft_map in pending_single_sessions.items():
                pending_multi_sessions[src_folder] = ft_map

        # Show a single combined dialog for all accumulated multi-session groups
        if pending_multi_sessions:
            root = None if len(folders) > 1 else Path(folders[0])
            self._handle_multi_session_drop(root, pending_multi_sessions)

        # Apply regular folders → current session only
        for frame_type, folder_files in folder_additions.items():
            match frame_type:
                case "lights":
                    self.chosen_session.lights.extend(folder_files)
                case "darks":
                    self.chosen_session.darks.extend(folder_files)
                case "flats":
                    self.chosen_session.flats.extend(folder_files)
                case "biases":
                    self.chosen_session.biases.extend(folder_files)
            self.siril.log(
                f"> Added {len(folder_files)} {frame_type} files to "
                f"{self.session_dropdown.currentText()} (folder drop)",
                LogColor.BLUE,
            )

        # Apply stacked calibration folders → ask session scope if >1 session
        if stacked_additions:
            if len(self.sessions) > 1:
                stacked_names = ", ".join(f"{ft}_stacked" for ft in stacked_additions)
                msg = QMessageBox(self)
                msg.setWindowTitle("Apply Stacked Calibration Frames")
                msg.setText(
                    f"You dropped stacked calibration folder(s):\n<b>{stacked_names}</b>\n\n"
                    f"Apply to which sessions?"
                )
                msg.setTextFormat(Qt.TextFormat.RichText)
                btn_current = msg.addButton(
                    f"Current Session ({self.session_dropdown.currentText()})",
                    QMessageBox.ButtonRole.AcceptRole,
                )
                btn_all = msg.addButton(
                    "All Sessions", QMessageBox.ButtonRole.ActionRole
                )
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked is btn_all:
                    stacked_target_sessions = self.sessions
                    stacked_session_label = "all sessions"
                elif clicked is btn_current:
                    stacked_target_sessions = [self.chosen_session]
                    stacked_session_label = self.session_dropdown.currentText()
                else:
                    stacked_target_sessions = []
                    stacked_session_label = ""
            else:
                stacked_target_sessions = [self.chosen_session]
                stacked_session_label = self.session_dropdown.currentText()

            for frame_type, stacked_files in stacked_additions.items():
                for session in stacked_target_sessions:
                    match frame_type:
                        case "darks":
                            session.darks.extend(stacked_files)
                        case "flats":
                            session.flats.extend(stacked_files)
                        case "biases":
                            session.biases.extend(stacked_files)
                if stacked_target_sessions:
                    self.siril.log(
                        f"> Added {len(stacked_files)} {frame_type} (stacked) files to "
                        f"{stacked_session_label} (folder drop)",
                        LogColor.BLUE,
                    )

        # ── Process plain files ──────────────────────────────────────────────
        # First, pull out any stacked calibration files identified by filename pattern.
        # e.g. session1_240s_2024-12-03_darks_stacked.fit → darks
        _STACKED_FILE_SUFFIXES = {
            "_darks_stacked": "darks",
            "_flats_stacked": "flats",
            "_biases_stacked": "biases",
        }

        stacked_files: dict[str, list[Path]] = {}
        regular_files: list[Path] = []
        for f in files:
            stem = Path(f).stem.lower()
            matched = False
            for suffix, frame_type in _STACKED_FILE_SUFFIXES.items():
                if suffix in stem:
                    stacked_files.setdefault(frame_type, []).append(Path(f))
                    matched = True
                    break
            if not matched:
                regular_files.append(Path(f))

        # Apply stacked calibration files → ask session scope if >1 session
        if stacked_files:
            if len(self.sessions) > 1:
                stacked_names = ", ".join(f"{ft}_stacked" for ft in stacked_files)
                msg = QMessageBox(self)
                msg.setWindowTitle("Apply Stacked Calibration Frames")
                msg.setText(
                    f"You dropped stacked/master calibration file(s):\n<b>{stacked_names}</b>\n\n"
                    f"Apply to which sessions?"
                )
                msg.setTextFormat(Qt.TextFormat.RichText)
                btn_current = msg.addButton(
                    f"Current Session ({self.session_dropdown.currentText()})",
                    QMessageBox.ButtonRole.AcceptRole,
                )
                btn_all = msg.addButton(
                    "All Sessions", QMessageBox.ButtonRole.ActionRole
                )
                msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked is btn_all:
                    stacked_file_sessions = self.sessions
                    stacked_file_label = "all sessions"
                elif clicked is btn_current:
                    stacked_file_sessions = [self.chosen_session]
                    stacked_file_label = self.session_dropdown.currentText()
                else:
                    stacked_file_sessions = []
                    stacked_file_label = ""
            else:
                stacked_file_sessions = [self.chosen_session]
                stacked_file_label = self.session_dropdown.currentText()

            for frame_type, sf_list in stacked_files.items():
                for session in stacked_file_sessions:
                    match frame_type:
                        case "darks":
                            session.darks.extend(sf_list)
                        case "flats":
                            session.flats.extend(sf_list)
                        case "biases":
                            session.biases.extend(sf_list)
                if stacked_file_sessions:
                    self.siril.log(
                        f"> Added {len(sf_list)} {frame_type} (stacked) file(s) to "
                        f"{stacked_file_label} (drag & drop)",
                        LogColor.BLUE,
                    )

        # Remaining non-stacked files → show frame-type dialog as before
        if regular_files:
            dlg = FileTypeDialog(
                file_count=len(regular_files),
                current_session_name=self.session_dropdown.currentText(),
                all_session_names=[f"Session {i+1}" for i in range(len(self.sessions))],
                current_session_index=self.session_dropdown.currentIndex(),
                parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted or dlg.chosen_type is None:
                if folder_additions or stacked_additions or stacked_files:
                    self.refresh_file_list()
                return
            filetype = dlg.chosen_type
            scope = dlg.chosen_scope

            # Determine which sessions to apply to
            if scope == "all":
                target_sessions = self.sessions
            elif scope == "selected":
                target_sessions = [self.sessions[i] for i in dlg.chosen_session_indices]
            else:
                target_sessions = [self.chosen_session]

            # Map master types to their underlying calibration list
            _master_map = {
                "master dark": "darks",
                "master flat": "flats",
                "master bias": "biases",
            }
            resolved_type = _master_map.get(filetype.lower(), filetype.lower())

            for session in target_sessions:
                match resolved_type:
                    case "lights":
                        session.lights.extend(regular_files)
                    case "darks":
                        session.darks.extend(regular_files)
                    case "flats":
                        session.flats.extend(regular_files)
                    case "biases":
                        session.biases.extend(regular_files)

            if scope == "all":
                session_label = "all sessions"
            elif scope == "selected":
                session_label = ", ".join(
                    f"Session {i+1}" for i in dlg.chosen_session_indices
                )
            else:
                session_label = self.session_dropdown.currentText()

            self.siril.log(
                f"> Added {len(regular_files)} {filetype} files to {session_label} (drag & drop)",
                LogColor.BLUE,
            )

        self.refresh_file_list()

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
                    src = Path(file)

                    # If source and destination are on different drives, always copy.
                    # Symlinking across drives (e.g. M:\ → H:\) causes Siril/libraw to
                    # resolve the symlink back to the original path and read from the
                    # untrusted mount, producing "Error in libraw Input/output error".
                    src_drive = src.resolve(strict=False).drive.lower()
                    dst_drive = dest_path.resolve(strict=False).drive.lower()
                    same_drive = src_drive == dst_drive

                    if ALWAYS_SYMLINK or same_drive:
                        try:
                            os.symlink(
                                src.resolve(strict=False),
                                dest_path.resolve(strict=False),
                            )
                            self.siril.log(
                                f"Symlinked {file} to {dest_path}", LogColor.BLUE
                            )
                            continue
                        except (OSError, NotImplementedError):
                            pass  # fall through to copy

                    shutil.copy(str(src), str(dest_path))
                    self.siril.log(f"Copied {file} to {dest_path}", LogColor.BLUE)
            else:
                self.siril.log(
                    f"Skipping {image_type}: no files found", LogColor.SALMON
                )

    # Background/foreground colors per frame type
    _FRAME_COLORS = {
        "lights": ("#c3e6cb", "#1d4e2d"),
        "darks": ("#b8daff", "#003680"),
        "flats": ("#ffeaa7", "#7d5a00"),
        "biases": ("#d1c4e9", "#311b5e"),
    }

    def _file_filter_label(self, file_type, path):
        """Filter label shown in the file list's Filter column.

        Darks and biases carry no filter. Lights/flats use a cached per-file
        FILTER-header read so repeated refreshes don't re-read over the network.
        """
        if file_type in ("darks", "biases"):
            return "\u2014"  # em dash
        cache = getattr(self, "_file_filter_cache", None)
        if cache is None:
            cache = self._file_filter_cache = {}
        key = str(path)
        if key not in cache:
            detected = _read_fits_filter(path)
            cache[key] = detected if detected else "\u2014"
        return cache[key]

    def _file_object_label(self, path):
        """Target name shown in the file list's Object column (cached header read)."""
        cache = getattr(self, "_file_object_cache", None)
        if cache is None:
            cache = self._file_object_cache = {}
        key = str(path)
        if key not in cache:
            obj = _read_fits_object(path)
            cache[key] = obj if obj else "\u2014"
        return cache[key]

    def refresh_file_list(self):
        self.file_listbox.setSortingEnabled(False)
        self.file_listbox.clear()
        self.siril.log(
            f"Switched to {self.session_dropdown.currentText()}", LogColor.BLUE
        )

        counts = {ft: 0 for ft in FRAME_TYPES}
        if self.chosen_session:
            for order, file_type in enumerate(FRAME_TYPES):
                files = self.chosen_session.get_files_by_type(file_type)
                counts[file_type] = len(files)
                if not files:
                    continue
                bg_hex, fg_hex = self._FRAME_COLORS.get(
                    file_type, ("#ffffff", "#000000")
                )
                row_bg = QBrush(QColor(bg_hex))
                row_fg = QBrush(QColor(fg_hex))

                # Group header — a strong colored bar spanning the whole row.
                header_item = SortableTreeItem(
                    [f"{file_type.capitalize()}  ({len(files)})"]
                )
                header_item.group_order = order
                header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                hfont = header_item.font(0)
                hfont.setBold(True)
                header_item.setFont(0, hfont)
                header_item.setBackground(0, QBrush(QColor(fg_hex)))
                header_item.setForeground(0, QBrush(QColor(bg_hex)))
                self.file_listbox.addTopLevelItem(header_item)
                header_item.setFirstColumnSpanned(True)

                for n, file in enumerate(files, start=1):
                    filt = self._file_filter_label(file_type, file)
                    obj = self._file_object_label(file)
                    child = SortableTreeItem([str(n), filt, obj, file.name])
                    for col in range(4):
                        child.setBackground(col, row_bg)
                        child.setForeground(col, row_fg)
                    child.setTextAlignment(
                        0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                    child.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
                    child.setToolTip(3, str(file.resolve()))
                    child.setData(0, Qt.ItemDataRole.UserRole, (file_type, file))
                    header_item.addChild(child)

        self.file_listbox.expandAll()
        self.file_listbox.setSortingEnabled(True)
        self.file_listbox.sortByColumn(0, Qt.SortOrder.AscendingOrder)

        # Group-box title with per-type counts, e.g.
        # "Files in Session 1 — 28 L · 10 D · 10 F · 10 B  (total 58)"
        if hasattr(self, "session_content_group"):
            idx = self.session_dropdown.currentIndex() + 1
            abbr = {"lights": "L", "darks": "D", "flats": "F", "biases": "B"}
            parts = [f"{counts[ft]} {abbr[ft]}" for ft in FRAME_TYPES if counts[ft]]
            total = sum(counts.values())
            title = f"Files in Session {idx}"
            if parts:
                title += f" \u2014 {' \u00b7 '.join(parts)}  (total {total})"
            self.session_content_group.setTitle(title)

        self.update_frame_buttons()

    def update_frame_buttons(self):
        """Show a green circle on each Add button when that frame type has files loaded."""
        if not hasattr(self, "lights_btn"):
            return
        btn_map = {
            "lights": (self.lights_btn, "Add Lights"),
            "darks": (self.darks_btn, "Add Darks"),
            "flats": (self.flats_btn, "Add Flats"),
            "biases": (self.biases_btn, "Add Biases"),
        }
        for file_type, (btn, label) in btn_map.items():
            files = self.chosen_session.get_files_by_type(file_type)
            btn.setText(f"\U0001f7e2 {label}" if files else label)

    # Debug code
    def show_all_sessions(self):
        for session in self.sessions:
            for file_type in FRAME_TYPES:
                files = session.get_files_by_type(file_type)
                if files:
                    self.siril.log(f"--- {file_type.upper()} ---", LogColor.BLUE)
                    for index, file in enumerate(files):
                        self.siril.log(
                            f"{index + 1:>4}. {file_type.capitalize():^20}  {str(file.resolve())}",
                            LogColor.BLUE,
                        )

    def remove_selected_files(self):
        # Only file rows carry (frame_type, path) data; group headers are ignored.
        selected_entries = []
        for item in self.file_listbox.selectedItems():
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data is not None:
                selected_entries.append(data)
        if not selected_entries:
            return

        msg = f"Are you sure you want to delete {len(selected_entries)} files? (Note: This will only remove them from the session, not delete them from disk.)"
        reply = QMessageBox.question(
            self,
            "Delete Selected Files?",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            for filetype, path in selected_entries:
                file_list = getattr(self.chosen_session, filetype)
                if path in file_list:
                    file_list.remove(path)
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
            self.update_process_separately_checkbox()  # Update checkbox state

    def _session_label(self, index, session):
        """Build a display label like 'Session 1' with per-type counts."""
        counts = session.get_file_count()
        parts = []
        for ft, abbr in (
            ("lights", "L"),
            ("darks", "D"),
            ("flats", "F"),
            ("biases", "B"),
        ):
            if counts[ft]:
                parts.append(f"{counts[ft]}{abbr}")
        label = f"Session {index + 1}"
        if parts:
            label += f" ({', '.join(parts)})"
        return label

    # ── Master calibration config / import / reset ────────────────────────
    # Top-level key for this script's settings within the shared config file.
    _CONFIG_SECTION = "osc"

    def _master_config_path(self) -> Path | None:
        """Return the path to the shared Naztronomy scripts config JSON, or None.

        The file lives in the Siril user config directory and is shared across
        the Naztronomy scripts; this script reads/writes its own ``osc`` section
        so the master darks folder and master bias file are remembered between
        runs.
        """
        try:
            config_dir = self.siril.get_siril_configdir()
        except Exception:
            return None
        if not config_dir:
            return None
        return Path(config_dir) / "naztronomy_scripts_config.json"

    def _load_master_config(self):
        """Load the persisted master darks folder, master bias file and the last
        Bortle choice from the shared config file."""
        path = self._master_config_path()
        if not path or not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        section = data.get(self._CONFIG_SECTION)
        if not isinstance(section, dict):
            return
        self.master_darks_dir = section.get("master_darks_dir") or None
        self.master_bias_path = section.get("master_bias_path") or None
        self._last_bortle = section.get("bortle", "") or ""

    def _write_config_section(self, updates: dict) -> bool:
        """Merge ``updates`` into this script's section of the shared config
        file, preserving other scripts' sections and this section's other keys.
        """
        path = self._master_config_path()
        if not path:
            return False
        data = {}
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError):
                data = {}
        section = data.get(self._CONFIG_SECTION)
        if not isinstance(section, dict):
            section = {}
        section.update(updates)
        data[self._CONFIG_SECTION] = section
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return True
        except OSError as exc:
            self.siril.log(f"> Could not save config: {exc}", LogColor.SALMON)
            return False

    def _save_master_config(self):
        """Persist the current master darks folder and master bias file."""
        if self._write_config_section(
            {
                "master_darks_dir": self.master_darks_dir or "",
                "master_bias_path": self.master_bias_path or "",
            }
        ):
            self.siril.log(
                f"> Saved config to {self._master_config_path()}", LogColor.BLUE
            )

    def open_master_config(self):
        """Popup to configure the master darks folder and the master bias file."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Master Calibration Config")
        dialog.setModal(True)
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)

        intro = QLabel(
            "Master darks are matched to each session by exposure time and\n"
            "temperature. The master bias is applied to every session. Click\n"
            "'Import Masters' afterwards to apply them. These locations are\n"
            "remembered between runs."
        )
        intro.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(intro)

        layout.addWidget(QLabel("<b>Master darks folder</b>"))
        darks_row = QHBoxLayout()
        darks_edit = QLineEdit(self.master_darks_dir or "")
        darks_edit.setPlaceholderText(
            "Folder containing master dark frames (FITS/XISF)"
        )
        darks_browse = QPushButton("Browse\u2026")
        darks_row.addWidget(darks_edit)
        darks_row.addWidget(darks_browse)
        layout.addLayout(darks_row)

        layout.addWidget(QLabel("<b>Master bias file</b>"))
        bias_row = QHBoxLayout()
        bias_edit = QLineEdit(self.master_bias_path or "")
        bias_edit.setPlaceholderText("Single master bias frame (FITS/XISF)")
        bias_browse = QPushButton("Browse\u2026")
        bias_row.addWidget(bias_edit)
        bias_row.addWidget(bias_browse)
        layout.addLayout(bias_row)

        def browse_darks():
            start = darks_edit.text().strip() or self.siril.get_siril_wd()
            chosen = QFileDialog.getExistingDirectory(
                dialog, "Select Master Darks Folder", start
            )
            if chosen:
                darks_edit.setText(chosen)

        def browse_bias():
            start = bias_edit.text().strip() or self.siril.get_siril_wd()
            chosen, _ = QFileDialog.getOpenFileName(
                dialog,
                "Select Master Bias File",
                start,
                "Calibration frames (*.fit *.fits *.fit.fz *.fits.fz *.xisf)",
            )
            if chosen:
                bias_edit.setText(chosen)

        darks_browse.clicked.connect(browse_darks)
        bias_browse.clicked.connect(browse_bias)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip(
            "Clear the saved master darks folder and master bias file."
        )
        clear_btn.clicked.connect(lambda: (darks_edit.clear(), bias_edit.clear()))
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)
        save_btn.clicked.connect(dialog.accept)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.master_darks_dir = darks_edit.text().strip() or None
            self.master_bias_path = bias_edit.text().strip() or None
            self._save_master_config()
            self.siril.log(
                "Master config saved: darks="
                f"{self.master_darks_dir or 'none'}, "
                f"bias={self.master_bias_path or 'none'}",
                LogColor.BLUE,
            )

    def _match_master_dark(self, dark_library, light_exp, light_temp):
        """Pick the best master dark for a light's exposure/temperature.

        Exposure must agree within MASTER_DARK_EXP_TOLERANCE seconds. Among the
        exposure matches the one with the nearest temperature (within
        MASTER_DARK_TEMP_TOLERANCE) is chosen. When neither the light nor any
        candidate carries a temperature, the first exposure match is used.
        Returns the chosen Path, or None when nothing matches.
        """
        exp_matches = [
            (path, temp)
            for (path, exptime, temp) in dark_library
            if abs(exptime - light_exp) <= MASTER_DARK_EXP_TOLERANCE
        ]
        if not exp_matches:
            return None

        if light_temp is None or all(temp is None for _, temp in exp_matches):
            return exp_matches[0][0]

        best_path = None
        best_delta = None
        for path, temp in exp_matches:
            if temp is None:
                continue
            delta = abs(temp - light_temp)
            if delta <= MASTER_DARK_TEMP_TOLERANCE and (
                best_delta is None or delta < best_delta
            ):
                best_delta = delta
                best_path = path
        return best_path

    def import_masters(self):
        """Match configured master darks to each session by exposure+temperature
        and apply the configured master bias to every session."""
        if not self.master_darks_dir and not self.master_bias_path:
            QMessageBox.information(
                self,
                "Import Masters",
                "No master darks folder or master bias file configured.\n\n"
                "Click 'Master Config…' first to set them.",
            )
            return

        # Build the master-dark library: (path, exptime, temp) per dark file.
        dark_library: list[tuple[Path, float, float | None]] = []
        if self.master_darks_dir:
            folder = Path(self.master_darks_dir)
            if folder.is_dir():
                for f in sorted(folder.rglob("*")):
                    if not f.is_file() or not _is_supported_input(f):
                        continue
                    exptime, temp = _get_exptime_temp(f)
                    if exptime is None:
                        self.siril.log(
                            f"> Skipping master dark '{f.name}': no exposure time "
                            "found in header or file name",
                            LogColor.SALMON,
                        )
                        continue
                    dark_library.append((f, exptime, temp))
            else:
                self.siril.log(
                    f"> Master darks folder not found: {self.master_darks_dir}",
                    LogColor.SALMON,
                )

        bias_file: Path | None = None
        if self.master_bias_path:
            candidate = Path(self.master_bias_path)
            if candidate.is_file():
                bias_file = candidate
            else:
                self.siril.log(
                    f"> Master bias file not found: {self.master_bias_path}",
                    LogColor.SALMON,
                )

        darks_applied = 0
        bias_applied = 0
        for index, session in enumerate(self.sessions):
            label = self._session_label(index, session)
            if not session.lights:
                continue

            if dark_library:
                light_exp, light_temp = _get_exptime_temp(session.lights[0])
                if light_exp is None:
                    self.siril.log(
                        f"> {label}: could not read light exposure time; "
                        "skipping master-dark match",
                        LogColor.SALMON,
                    )
                else:
                    best = self._match_master_dark(dark_library, light_exp, light_temp)
                    if best is None:
                        temp_part = (
                            f" / temp {light_temp:g}\u00b0C"
                            if light_temp is not None
                            else ""
                        )
                        self.siril.log(
                            f"> {label}: no master dark matches exposure "
                            f"{light_exp:g}s{temp_part} \u2014 skipped",
                            LogColor.SALMON,
                        )
                    elif best not in session.darks:
                        session.darks.append(best)
                        darks_applied += 1
                        self.siril.log(
                            f"> {label}: applied master dark '{best.name}'",
                            LogColor.GREEN,
                        )

            if bias_file is not None and bias_file not in session.biases:
                session.biases.append(bias_file)
                bias_applied += 1

        if bias_file is not None and bias_applied:
            self.siril.log(
                f"> Applied master bias '{bias_file.name}' to {bias_applied} "
                "session(s)",
                LogColor.GREEN,
            )

        self.siril.log(
            f"Import Masters complete: {darks_applied} master dark(s) and "
            f"{bias_applied} master bias assignment(s) added.",
            LogColor.BLUE,
        )
        self.update_dropdown()
        self.refresh_file_list()

    def reset_calibration(self):
        """Remove all calibration frames (darks, flats, biases) from chosen
        sessions. Lights are left untouched."""
        if not self.sessions:
            return

        if len(self.sessions) == 1:
            reply = QMessageBox.question(
                self,
                "Reset Calibration",
                "Are you sure you want to remove the calibration frames "
                "(darks, flats, biases) from this session?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._clear_session_calibration([0])
            return

        # Multiple sessions: let the user pick which ones to clear.
        indices, scope = self._prompt_reset_calibration_sessions()
        if indices is None:
            return  # cancelled

        scope_word = "all" if scope == "all" else "selected"
        reply = QMessageBox.question(
            self,
            "Reset Calibration",
            f"Are you sure you want to remove the calibration frames "
            f"(darks, flats, biases) from the {scope_word} sessions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._clear_session_calibration(indices)

    def _clear_session_calibration(self, indices: list):
        """Clear darks/flats/biases for the sessions at the given indices."""
        cleared = 0
        for i in indices:
            if 0 <= i < len(self.sessions):
                session = self.sessions[i]
                if session.darks or session.flats or session.biases:
                    cleared += 1
                session.darks.clear()
                session.flats.clear()
                session.biases.clear()
        self.siril.log(
            f"Reset calibration: cleared calibration frames from {cleared} "
            "session(s).",
            LogColor.BLUE,
        )
        self.update_dropdown()
        self.refresh_file_list()

    def _prompt_reset_calibration_sessions(self):
        """Show a checkbox list of sessions with Selected/All/Cancel options.

        Returns a tuple ``(indices, scope)`` where ``indices`` is the list of
        chosen session indices and ``scope`` is ``"all"`` or ``"selected"``.
        Returns ``(None, None)`` if the user cancels.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Reset calibration for which sessions?")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)
        dlg_layout = QVBoxLayout(dialog)

        dlg_layout.addWidget(
            QLabel("Remove <b>calibration frames</b> (darks, flats, biases) from:")
        )

        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(360)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)

        current_index = self.session_dropdown.currentIndex()
        checkboxes = []
        for i, session in enumerate(self.sessions):
            cb = QCheckBox(self._session_label(i, session))
            cb.setChecked(i == current_index)
            checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        dlg_layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        selected_btn = QPushButton("Selected Sessions")
        all_btn = QPushButton("All Sessions")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(selected_btn)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(cancel_btn)
        dlg_layout.addLayout(btn_row)

        result = {"action": None}

        def on_selected():
            result["action"] = "selected"
            dialog.accept()

        def on_all():
            result["action"] = "all"
            dialog.accept()

        selected_btn.clicked.connect(on_selected)
        all_btn.clicked.connect(on_all)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None, None

        if result["action"] == "all":
            return list(range(len(self.sessions))), "all"
        if result["action"] == "selected":
            indices = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
            if not indices:
                return None, None  # nothing checked — treat as cancel
            return indices, "selected"
        return None, None

    # ── AstroBin acquisition CSV export ───────────────────────────────────
    # Columns in order. Only "number" and "duration" are mandatory; the rest are
    # emitted (header always present) when available and left blank otherwise.
    _CSV_COLUMNS = (
        "date",
        "filter",
        "filterName",
        "number",
        "duration",
        "binning",
        "gain",
        "sensorCooling",
        "fNumber",
        "darks",
        "flats",
        "flatDarks",
        "bias",
        "bortle",
    )

    @staticmethod
    def _csv_num(value):
        """Format a number for CSV: drop a trailing .0 from whole numbers."""
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _session_csv_row(self, session):
        """Build one AstroBin CSV row from a session, or None when it has no lights."""
        if not session.lights:
            return None

        meta = _read_light_metadata(session.lights[0])
        cooling = meta.get("sensorCooling")
        row = {
            "date": meta.get("date", ""),
            "filter": "",  # AstroBin filter ID — not derivable from headers
            "filterName": meta.get("filterName", ""),
            "number": str(len(session.lights)),
            "duration": self._csv_num(meta.get("duration")),
            "binning": self._csv_num(meta.get("binning")),
            "gain": self._csv_num(meta.get("gain")),
            "sensorCooling": (
                self._csv_num(round(cooling)) if cooling is not None else ""
            ),
            "fNumber": f"{meta['fNumber']:.2f}" if "fNumber" in meta else "",
            "darks": str(len(session.darks)) if session.darks else "",
            "flats": str(len(session.flats)) if session.flats else "",
            "flatDarks": "",  # flat-darks are not tracked per session
            "bias": str(len(session.biases)) if session.biases else "",
            "bortle": "",  # sky quality — not derivable from headers
        }
        return row

    def _prompt_bortle(self):
        """Ask for the Bortle scale to apply to every row. Returns the chosen
        value as a string ("" when 'Leave blank'), or None if cancelled."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Bortle Sky Quality")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Select the Bortle scale to apply to every session:"))

        combo = QComboBox()
        combo.addItem("", "")
        bortle_descriptions = {
            1: "Excellent dark-sky site",
            2: "Typical truly dark site",
            3: "Rural sky",
            4: "Rural/suburban transition",
            5: "Suburban sky",
            6: "Bright suburban sky",
            7: "Suburban/urban transition",
            8: "City sky",
            9: "Inner-city sky",
        }
        for value, desc in bortle_descriptions.items():
            combo.addItem(f"{value} \u2014 {desc}", str(value))
        saved_idx = combo.findData(self._last_bortle or "")
        if saved_idx >= 0:
            combo.setCurrentIndex(saved_idx)
        layout.addWidget(combo)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return combo.currentData()

    def export_sessions_csv(self):
        """Export all sessions with lights to an AstroBin acquisition CSV."""
        rows = []
        for session in self.sessions:
            row = self._session_csv_row(session)
            if row is not None:
                rows.append(row)

        if not rows:
            QMessageBox.information(
                self,
                "Export Sessions to CSV",
                "No sessions with light frames to export.",
            )
            return

        missing_duration = sum(1 for r in rows if not r["duration"])
        if missing_duration:
            self.siril.log(
                f"> {missing_duration} session(s) have no detectable exposure "
                "time; their 'duration' column will be blank.",
                LogColor.SALMON,
            )

        bortle = self._prompt_bortle()
        if bortle is None:
            return  # user cancelled
        self._last_bortle = bortle
        self._write_config_section({"bortle": bortle})
        for row in rows:
            row["bortle"] = bortle

        try:
            start_dir = self.siril.get_siril_wd()
        except Exception:
            start_dir = ""
        default_path = os.path.join(start_dir, "astrobin_acquisitions.csv")

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sessions to CSV",
            default_path,
            "CSV Files (*.csv)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        try:
            with open(save_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self._CSV_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            self.siril.log(f"Failed to export CSV: {exc}", LogColor.RED)
            QMessageBox.warning(
                self,
                "Export Sessions to CSV",
                f"Could not write the CSV file:\n{exc}",
            )
            return

        self.siril.log(
            f"Exported {len(rows)} session(s) to {save_path}", LogColor.GREEN
        )

    # end session methods

    # Start Siril processing methods
    # image_type: lights, darks, biases, flats
    def convert_files(self, image_type):
        directory = os.path.join(self.current_working_directory, image_type)
        self.siril.log(f'Converting files in "{directory}"', LogColor.BLUE)
        if os.path.isdir(directory):
            print(f"Found directory for {image_type}: {directory}")
            self.siril.cmd("cd", f'"{directory}"')

            # Ignore hidden files and dirs.
            # Also accept symlinks: os.path.isfile() silently returns False for
            # symlinks whose targets live on an untrusted mount (WinError 448).
            def _is_file_or_link(p):
                return os.path.islink(p) or (os.path.isfile(p) and not os.path.isdir(p))

            file_count = len(
                [
                    name
                    for name in os.listdir(directory)
                    if _is_file_or_link(os.path.join(directory, name))
                    and not name.startswith(".")
                ]
            )
            if file_count == 0:
                self.siril.log(
                    f"No files found in {image_type} directory. Skipping conversion.",
                    LogColor.SALMON,
                )
                return
            elif file_count == 1:
                self.siril.log(
                    f"Only one file found in {image_type} directory. Treating it like a master {image_type} frame.",
                    LogColor.BLUE,
                )
                first_file = next(
                    name
                    for name in os.listdir(directory)
                    if _is_file_or_link(os.path.join(directory, name))
                    and not name.startswith(".")
                )
                src = os.path.join(directory, first_file)

                dst = os.path.join(
                    self.current_working_directory,
                    "process",
                    f"{image_type}_stacked{self.fits_extension}",
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                self.siril.log(
                    f"Copied master {image_type} to process as {image_type}_stacked.",
                    LogColor.BLUE,
                )
                self.siril.cmd("cd", "..")
                # return false because there's no conversion
                return False
            else:
                try:
                    # Check file extension to determine whether to use link or convert. Fixes bug for other raw files
                    first_file = next(
                        name
                        for name in os.listdir(directory)
                        if _is_file_or_link(os.path.join(directory, name))
                        and not name.startswith(".")
                    )
                    file_ext = os.path.splitext(first_file)[1].lower()
                    fits_extensions = [".fit", ".fits", ".fit.fz", ".fits.fz"]

                    if file_ext in fits_extensions:
                        args = ["link", image_type, "-out=../process"]
                    else:
                        args = ["convert", image_type, "-out=../process"]

                    # if "lights" in image_type.lower():
                    #     if not self.drizzle_status:
                    #         args.append("-debayer")
                    # else:
                    #     if not self.drizzle_status:
                    #         # flats, darks, bias: only debayer if drizzle is not set
                    #         args.append("-debayer")

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
                return True
        else:
            self.siril.error_messagebox(f"Directory {directory} does not exist", True)
            raise NoImageError(
                (
                    f'No directory named "{image_type}" at this location. Make sure the working directory is correct.'
                )
            )

    def local_astrometry_net_available(self):
        """Return True when Siril can find a local solve-field installation."""
        configured_dir = ""
        try:
            configured_dir = self.siril.get_siril_config("core", "asnet_dir")
        except (s.DataError, s.CommandError, s.SirilError):
            pass

        if configured_dir and configured_dir != "(not set)":
            candidates = [
                os.path.join(configured_dir, "solve-field"),
                os.path.join(configured_dir, "solve-field.exe"),
                os.path.join(configured_dir, "bin", "solve-field"),
                os.path.join(configured_dir, "bin", "solve-field.exe"),
            ]
            if any(os.path.isfile(candidate) for candidate in candidates):
                return True

        if shutil.which("solve-field"):
            return True

        if sys.platform.startswith("win"):
            ansvr_root = os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "cygwin_ansvr"
            )
            return any(
                os.path.isfile(os.path.join(ansvr_root, "bin", executable))
                for executable in ("solve-field", "solve-field.exe")
            )

        return False

    # Plate solve on sequence runs when file count < 2048
    def seq_plate_solve(self, seq_name, use_astrometry_net=False):
        """Runs the siril command 'seqplatesolve' to plate solve the converted files."""
        # self.siril.cmd("cd", "process")
        args = ["seqplatesolve", seq_name]

        args.extend(["-nocache", "-force", "-disto=ps_distortion"])

        try:
            self.siril.cmd(*args)
            self.siril.log(f"Platesolved {seq_name}", LogColor.GREEN)
            return True
        except (s.DataError, s.CommandError, s.SirilError) as e:
            if use_astrometry_net and self.astrometry_net_available:
                self.siril.log(
                    f"Siril seqplatesolve reported a failure; retrying unsolved "
                    f"frames with local astrometry.net (slower): {e}",
                    LogColor.SALMON,
                )
                fallback_args = [arg for arg in args if arg != "-force"]
                fallback_args.extend(["-localasnet", "-blindpos", "-blindres"])
                try:
                    self.siril.cmd(*fallback_args)
                    self.siril.log(
                        f"Astrometry.net fallback completed for {seq_name}",
                        LogColor.GREEN,
                    )
                    return True
                except (s.DataError, s.CommandError, s.SirilError) as fallback_error:
                    e = fallback_error
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
        self,
        seq_name,
        drizzle_amount,
        pixel_fraction,
        filter_wfwhm=3,
        filter_round=3,
        filter_stars=3,
        filter_bkg=3,
        use_filter_round=False,
        use_filter_wfwhm=False,
        use_filter_stars=False,
        use_filter_bkg=False,
    ):
        """Apply Existing Registration to the sequence."""
        cmd_args = [
            "seqapplyreg",
            seq_name,
            "-kernel=square",
        ]

        # Sigma, Percentage, or absolute value for each filter type (if enabled).
        # "abs" appends the raw value with no suffix (Siril absolute threshold).
        def _filter_value(mode: str, value: float) -> str:
            if mode == "σ":
                return f"{value}k"
            if mode == "%":
                return f"{int(value)}%"
            return f"{value}"  # abs

        if use_filter_round:
            cmd_args.append(
                f"-filter-round={_filter_value(self.roundness_mode_combo.currentText(), filter_round)}"
            )
        if use_filter_wfwhm:
            cmd_args.append(
                f"-filter-wfwhm={_filter_value(self.fwhm_mode_combo.currentText(), filter_wfwhm)}"
            )
        if use_filter_stars:
            cmd_args.append(
                f"-filter-nbstars={_filter_value(self.stars_mode_combo.currentText(), filter_stars)}"
            )
        if use_filter_bkg:
            cmd_args.append(
                f"-filter-bkg={_filter_value(self.bkg_mode_combo.currentText(), filter_bkg)}"
            )

        # If not doing a paneled mosaic, use max framing, otherwise crop down to reference frame so edges don't have ugly noise
        if not self.paneled_mosaic_radio.isChecked():
            cmd_args.append("-framing=max")

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
                # return
            else:
                self.siril.cmd(
                    "stack",
                    f"{seq_name} rej 3 3",
                    "-norm=mul",
                    f"-out={seq_name}_stacked",
                )

                # return
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
                self.siril.cmd("cd", "..")
            except (s.DataError, s.CommandError, s.SirilError) as e:
                self.siril.log(f"Command execution failed: {e}", LogColor.RED)
                self.close_dialog()

        self.siril.log(f"Completed stacking {seq_name}!", LogColor.GREEN)
        # Copy the stacked calibration files to ../masters directory
        # Store original working directory path by going up 3 levels from process dir
        original_wd = self.home_directory
        masters_dir = os.path.join(original_wd, "masters")
        os.makedirs(masters_dir, exist_ok=True)

        src = os.path.join(
            self.current_working_directory,
            f"process/{seq_name}_stacked{self.fits_extension}",
        )

        # Get current session name from working directory path
        session_name = Path(self.current_working_directory).name  # e.g. session1

        # Read FITS headers if file exists
        filename_parts = [session_name, seq_name, "stacked"]

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
        # TODO: prefix for each session
        cmd_args = [
            "calibrate",
            f"{seq_name}",
        ]
        if not self.drizzle_status and not self.mono_check.isChecked():
            cmd_args.append("-debayer")

        if os.path.exists(
            os.path.join(
                self.current_working_directory,
                f"process/darks_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-dark=darks_stacked")
            # Cosmetic Correction with sigma clipping 3 low and 3 high
            cmd_args.append("-cc=dark 3 3")
        if os.path.exists(
            os.path.join(
                self.current_working_directory,
                f"process/flats_stacked{self.fits_extension}",
            )
        ):
            cmd_args.append("-flat=flats_stacked")

        # apply dark flats to lights if using that instead of bias frames
        if (
            os.path.exists(
                os.path.join(
                    self.current_working_directory,
                    f"process/biases_stacked{self.fits_extension}",
                )
            )
            and self.dark_flats_check.isChecked()
            # and not self.bg_extract_check.isChecked()
        ):
            cmd_args.append("-bias=biases_stacked")
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
        self,
        seq_name,
        feather,
        feather_amount,
        rejection=False,
        output_name=None,
        overlap_norm=False,
        output_norm=True,
        stack_weighted=False,
        weighting_method="Weighted FWHM",
    ):
        """Stack it all, and feather if it's provided"""
        out = "result" if output_name is None else output_name

        cmd_args = [
            "stack",
            f"{seq_name}",
            " rej 3 3" if rejection else " rej none",
            "-norm=addscale",
            "-output_norm" if output_norm else "",
            "-overlap_norm" if overlap_norm else "",
            "-rgb_equal",
            "-maximize",
            "-filter-included",
            "-32b",
            f"-out={out}",
        ]
        if stack_weighted:
            weighting_map = {
                "Number of Stars": "nbstars",
                "Weighted FWHM": "wfwhm",
                "Noise": "noise",
            }
            weight_option = weighting_map.get(weighting_method, "wfwhm")
            cmd_args.append(f"-weight={weight_option}")
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
        file_name = f"result_drizzle-{drizzle_str}x_{current_datetime}{suffix}"

        # Get header info from loaded image for filename
        current_fits_headers = self.siril.get_image_fits_header(return_as="dict")

        object_name = current_fits_headers.get("OBJECT", "Unknown").replace(" ", "_")
        exptime = int(current_fits_headers.get("EXPTIME", 0))
        livetime = int(current_fits_headers.get("LIVETIME", 0))
        stack_count = int(current_fits_headers.get("STACKCNT", 0))

        file_name = f"{object_name}_{stack_count:03d}x{exptime}sec_{livetime}s_"  # {date_obs_str}"
        if self.drizzle_status:
            file_name += f"drizzle-{drizzle_str}x_"

        file_name += f"{current_datetime}{suffix}"
        # Add filter information if available
        filter_name = current_fits_headers.get("FILTER", "").strip().replace(" ", "_")
        if filter_name:
            file_name += f"_{filter_name}"

        try:
            self.siril.cmd(
                "save",
                f"{file_name}",
            )
            self.siril.log(f"Saved file: {file_name}", LogColor.GREEN)
            return file_name
        except (s.DataError, s.CommandError, s.SirilError) as e:
            self.siril.log(f"Save command execution failed: {e}", LogColor.RED)
            self.close_dialog()

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
        if not os.path.isdir(process_dir):
            return
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
                    # Retry loop for safe deletion
                    for i in range(3):
                        try:
                            os.remove(file_path)
                            break
                        except OSError:
                            time.sleep(0.5)
                    else:
                        # If loop completes without break, deletion failed
                        self.siril.log(f"Failed to delete {file_path}", LogColor.SALMON)
        self.siril.log(f"Cleaned up {prefix}", LogColor.BLUE)

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{APP_NAME} — Help")
        dialog.setMinimumSize(620, 580)
        dialog.resize(660, 640)

        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        # Header
        header_label = QLabel(f"<b>{APP_NAME}</b>")
        header_font = QFont()
        header_font.setPointSize(11)
        header_label.setFont(header_font)
        outer.addWidget(header_label)

        author_label = QLabel(f'<a href="{WEBSITE}">{AUTHOR} (Naztronomy)</a>')
        author_label.setOpenExternalLinks(True)
        outer.addWidget(author_label)

        # Scrollable help text
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        browser.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        browser.setHtml("""
            <style>
            body  { font-family: sans-serif; font-size: 13px; margin: 4px; }
            h3    { margin-bottom: 4px; margin-top: 14px; color: #2c7bb6; }
            h3:first-child { margin-top: 0; }
            ul    { margin-top: 2px; padding-left: 18px; }
            li    { margin-bottom: 3px; }
            .note { color: #888; font-style: italic; }
            </style>

            <h3>General</h3>
            <ul>
            <li>Use a <b>blank working directory</b> for the cleanest setup.</li>
            <li>Calibration frames (darks, flats, biases) are <b>optional</b>.</li>
            <li>Single calibration files are treated as <b>masters</b> automatically.</li>
            <li>Master frames are saved to a <b>masters/</b> directory with descriptive names.</li>
            <li>Filter settings can be used to <b>exclude poor quality frames</b> before stacking.</li>
            <li>Maximum of <b>2048 total light frames</b> across all sessions on Windows. Experimental UCRT64 version increases the limit to 8192 files.</li>
            <li>Always <b>include logs</b> when asking for help. Click the download arrow at the bottom of the console to export your logs.</li>
            </ul>

            <h3>Sessions</h3>
            <ul>
            <li>Add multiple sessions to process <b>data from different nights or targets</b>.</li>
            <li>Each session has its own lights, darks, flats, and biases.</li>
            <li>Calibration cannot be shared between sessions (yet).</li>
            <li>Preprocessed lights are saved to a <b>collected_lights/</b> directory when
                <i>Save Calibrated Lights</i> is enabled.</li>
            </ul>

            <h3>Master Calibration &amp; Export</h3>
            <ul>
            <li><b>Master Config…</b>: set a reusable <i>master darks folder</i> and a single
                <i>master bias file</i>. These locations are remembered between runs.</li>
            <li><b>Import Masters</b>: applies the configured masters to every session. Master
                darks are matched to each session by <b>exposure time and temperature</b>; the
                master bias is applied to all sessions.</li>
            <li><b>Reset Calibration</b>: removes darks, flats, and biases from chosen sessions
                (pick which sessions when more than one exists). Light frames are left untouched.</li>
            <li><b>Export Sessions to CSV</b>: writes an <b>AstroBin acquisition CSV</b> with the
                date, filter, frame counts, exposure, gain, binning, cooling, and f-number read
                from each session's first light frame. You are prompted for a Bortle value that is
                applied to every row.</li>
            <li>The file list is a <b>sortable, grouped tree</b> — click a column header to sort;
                rows stay grouped by frame type and the group title shows per-type counts.</li>
            </ul>

            <h3>Drag &amp; Drop</h3>
            <ul>
            <li>You can drag and drop <b>files or folders</b> directly onto the file list.</li>
            <li><b>Named folders</b> (<code>lights</code>, <code>darks</code>, <code>flats</code>,
                <code>biases</code>, <code>dark_flats</code>) are recognised automatically — their
                contents are added to the <b>current session</b> with no dialog.</li>
            <li>Drop a <b>parent folder</b> that contains those named subfolders and each subfolder
                is scanned and categorised the same way.</li>
            <li><b>Stacked calibration files</b> whose filenames contain <code>_darks_stacked</code>,
                <code>_flats_stacked</code>, or <code>_biases_stacked</code> are auto-categorised
                into their respective frame type. When more than one session exists you will be
                asked whether to apply them to the <i>current session</i> or <i>all sessions</i>.</li>
            <li>Any other dropped files trigger the normal <b>frame-type dialog</b> where you
                choose Lights, Darks, Flats, Biases, or a Master calibration type and the target
                session scope.</li>
            <li>You can mix files and folders in a single drop - each group is handled
                independently.</li>
            </ul>

            <h3>Presets</h3>
            <ul>
            <li>Presets auto-save/load from the <b>presets/</b> directory in your working folder.</li>
            <li>Use <b>Save As…</b> / <b>Load From…</b> (dropdown arrow on the buttons) to choose
                a custom file location.</li>
            </ul>

            <h3>Drizzle</h3>
            <ul>
            <li>Drizzle can improve resolution but <b>increases processing time</b> and file size significantly. It may also product black frames which are then automatically purged by this script.</li>
            <li>Recommended to use drizzle 1x.</li>
            <li>Lower pixel-fraction values reduce artifacts but may increase noise.</li>
            </ul>

            <h3>Target Modes</h3>
            <ul>
            <li><b>Single Target</b>: All sessions are combined into one final stacked image.
                Best for imaging the same object across multiple nights.</li>
            <li><b>Multi Target (Do Not Combine)</b>: Each session is processed and stacked
                <i>separately</i>. Use when sessions contain different objects or filters that
                should not be merged.</li>
            <li><b>Create Paneled Mosaic</b>: Sessions are registered and stacked together
                with overlap-normalisation to produce a seamless mosaic. Each session should
                cover a different panel of the same field.</li>
            <li><b>Mono (Experimental)</b>: For monochrome cameras. Frames are calibrated and
                stacked individually per session and are <b>not combined</b>. Debayering is
                skipped. A final mono_stacks folder is created where the stacked and registered mono sessions are saved.</li>
            </ul>

            <h3>Create Final Stack</h3>
            <ul>
            <li>When <i>Single Target</i> is selected and <i>Save Calibrated Lights</i> is
                <b>off</b>, each session is stacked individually first, then the per-session
                stacks are combined into a final result.</li>
            <li>When <i>Save Calibrated Lights</i> is <b>on</b>, all calibrated lights are
                collected and stacked together in one pass.</li>
            </ul>
            """)
        outer.addWidget(browser)

        # Social / community buttons
        links_label = QLabel("<b>Community &amp; Support</b>")
        outer.addWidget(links_label)

        links_row = QHBoxLayout()
        links_row.setSpacing(8)

        social_buttons = [
            (
                "YouTube",
                "#FF0000",
                f"{YOUTUBE}",
            ),
            ("Discord", "#5865F2", f"{DISCORD}"),
            ("Patreon", "#FF424D", f"{PATREON}"),
            ("Buy Me a Coffee", "#FFDD00", f"{BUY_ME_A_COFFEE}"),
            # ("Website", "#2c7bb6", f"{WEBSITE}"),
        ]

        for label, color, url in social_buttons:
            btn = QPushButton(label)
            text_color = "#000" if color == "#FFDD00" else "#fff"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: {text_color};"
                f" border: none; border-radius: 4px; padding: 6px 10px; font-weight: bold; }}"
                f" QPushButton:hover {{ opacity: 0.85; border: 1px solid rgba(0,0,0,0.3); }}"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(url)
            btn.clicked.connect(
                lambda checked, u=url: QDesktopServices.openUrl(QUrl(u))
            )
            links_row.addWidget(btn)

        outer.addLayout(links_row)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(dialog.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        dialog.exec()

    def create_widgets(self):
        """Creates the UI widgets using PyQt6."""

        # Main layout
        main_widget = QWidget()
        self.setMinimumSize(750, 700)
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
        files_layout.setSpacing(8)

        # ── Top container: Session Management ──────────────────────────────
        session_mgmt_group = QGroupBox("Session Management")
        session_mgmt_layout = QVBoxLayout(session_mgmt_group)
        session_mgmt_layout.setContentsMargins(10, 8, 10, 8)

        session_row = QGridLayout()
        session_row.setHorizontalSpacing(6)
        session_row.setVerticalSpacing(6)
        session_label = QLabel("Session:")
        self.update_dropdown()
        self.session_dropdown.setCurrentIndex(0)

        add_session_btn = QPushButton("+ Add Session")
        add_session_btn.clicked.connect(self.add_dropdown_session)
        remove_session_btn = QPushButton("\u2013 Remove Session")
        remove_session_btn.clicked.connect(self.remove_session)

        self.master_config_btn = QPushButton("Master Config\u2026")
        self.master_config_btn.setToolTip(
            "Configure a reusable master darks folder and a single master bias\n"
            "file. Master darks are matched to each session by exposure time and\n"
            "temperature; the master bias is applied to every session."
        )
        self.master_config_btn.clicked.connect(self.open_master_config)

        self.import_masters_btn = QPushButton("Import Masters")
        self.import_masters_btn.setToolTip(
            "Apply the configured master darks and master bias to all open\n"
            "sessions. Darks are matched by exposure time and temperature."
        )
        self.import_masters_btn.clicked.connect(self.import_masters)

        self.reset_calibration_btn = QPushButton("Reset Calibration")
        self.reset_calibration_btn.setToolTip(
            "Remove all calibration frames (darks, flats, biases) from chosen\n"
            "sessions. With multiple sessions you can pick which sessions to\n"
            "clear; light frames are left untouched."
        )
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)

        # Row 0: Session label + dropdown, then the two session buttons.
        session_row.addWidget(session_label, 0, 0)
        session_row.addWidget(self.session_dropdown, 0, 1)
        session_row.addWidget(add_session_btn, 0, 2)
        session_row.addWidget(remove_session_btn, 0, 3)
        # Row 1: master buttons lined up under the session buttons (cols 1-3).
        session_row.addWidget(self.master_config_btn, 1, 1)
        session_row.addWidget(self.import_masters_btn, 1, 2)
        session_row.addWidget(self.reset_calibration_btn, 1, 3)
        # Let the dropdown column absorb extra width so buttons keep their size.
        session_row.setColumnStretch(1, 1)
        session_mgmt_layout.addLayout(session_row)

        files_layout.addWidget(session_mgmt_group)

        # ── Bottom container: Session Content ──────────────────────────────
        self.session_content_group = QGroupBox("Files in Session 1")
        session_content_layout = QVBoxLayout(self.session_content_group)
        session_content_layout.setContentsMargins(10, 8, 10, 8)
        session_content_layout.setSpacing(6)

        # Frame buttons
        frame_buttons = QHBoxLayout()
        self.lights_btn = QPushButton("Add Lights")
        self.lights_btn.clicked.connect(lambda: self.load_files("Lights"))
        self.darks_btn = QPushButton("Add Darks")
        self.darks_btn.clicked.connect(lambda: self.load_files("Darks"))
        self.flats_btn = QPushButton("Add Flats")
        self.flats_btn.clicked.connect(lambda: self.load_files("Flats"))
        self.biases_btn = QPushButton("Add Biases")
        self.biases_btn.clicked.connect(lambda: self.load_files("Biases"))
        self.biases_btn.setToolTip("Bias frames or Dark Flats can be used.")

        for btn in [self.lights_btn, self.darks_btn, self.flats_btn, self.biases_btn]:
            frame_buttons.addWidget(btn)
        session_content_layout.addLayout(frame_buttons)

        drop_hint_label = QLabel(
            "\u2193  or drag & drop files directly onto the list below"
        )
        drop_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_hint_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        session_content_layout.addWidget(drop_hint_label)

        # Files list
        self.file_listbox = DragDropTreeWidget(
            on_drop_callback=self._handle_dropped_files,
            on_delete_callback=self.remove_selected_files,
        )
        self.file_listbox.setColumnCount(4)
        self.file_listbox.setHeaderLabels(["#", "Filter", "Object", "File name"])
        self.file_listbox.setRootIsDecorated(True)
        self.file_listbox.setIndentation(14)
        self.file_listbox.setUniformRowHeights(True)
        self.file_listbox.setAlternatingRowColors(False)
        self.file_listbox.setSortingEnabled(True)
        self.file_listbox.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_listbox.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self.file_listbox.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(True)
        self.file_listbox.setToolTip(
            "Drag and drop files or folders here to add them to the current session.\n"
            "Named folders (lights, darks, flats, biases) are auto-categorised.\n"
            "Drop a parent folder containing those subfolders to load an entire session,\n"
            "or a folder with session sub-folders to auto-create multiple sessions.\n"
            "Click a column header to sort; rows are grouped by frame type."
        )
        self.file_listbox.viewport().setMouseTracking(True)
        session_content_layout.addWidget(self.file_listbox)

        # Export sessions to an AstroBin acquisition CSV.
        export_csv_btn = QPushButton("Export Sessions to CSV")
        export_csv_btn.setToolTip(
            "Export each session (date, filter, frame counts, exposure, gain,\n"
            "binning, cooling, f-number) to an AstroBin acquisition CSV file.\n"
            "'number' and 'duration' are always filled; other columns are blank\n"
            "when the data is unavailable."
        )
        export_csv_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #6f42c1;"
            "    color: white;"
            "    border: none;"
            "    padding: 6px 12px;"
            "    border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #8250df; }"
            "QPushButton:pressed { background-color: #5a32a3; }"
        )
        export_csv_btn.clicked.connect(self.export_sessions_csv)
        session_content_layout.addWidget(export_csv_btn)

        file_buttons = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected File(s)")
        remove_btn.clicked.connect(self.remove_selected_files)
        remove_btn.setToolTip(
            "Remove the selected file(s) from this session.\n"
            "Tip: You can also press Delete or Backspace while a file(s) is highlighted."
        )
        reset_btn = QPushButton("Reset Everything")
        reset_btn.clicked.connect(self.reset_everything)
        reset_btn.setToolTip("Warning: This will remove all sessions and files!")

        file_buttons.addWidget(remove_btn)
        file_buttons.addWidget(reset_btn)
        session_content_layout.addLayout(file_buttons)

        files_layout.addWidget(self.session_content_group)

        # Processing tab
        processing_tab = QWidget()
        processing_tab_outer = QVBoxLayout(processing_tab)
        processing_tab_outer.setContentsMargins(0, 0, 0, 0)
        processing_tab_outer.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        scroll_content = QWidget()
        processing_layout = QVBoxLayout(scroll_content)

        # Drizzle settings
        preprocessing_group = QGroupBox("Optional Preprocessing Steps")
        preprocessing_layout = QVBoxLayout()

        dark_flats_tooltip = "If your bias frames are dark flats instead, check this box. It'll be properly applied to the light frames during calibration."
        self.dark_flats_check = QCheckBox("Using Dark Flats?")
        self.dark_flats_check.setToolTip(dark_flats_tooltip)
        preprocessing_layout.addWidget(self.dark_flats_check)

        cleanup_tooltip = "Enable this option to delete all intermediary files after they are done processing. This saves space on your hard drive.\nNote: If your session is batched, this option is automatically enabled even if it's unchecked!"
        self.cleanup_check = QCheckBox("Clean up intermediate files")
        self.cleanup_check.setToolTip(cleanup_tooltip)
        preprocessing_layout.addWidget(self.cleanup_check)

        bg_extract_tooltip = "Removes background gradients from your images before stacking. Uses Polynomial value 1 and 10 samples."

        self.bg_extract_check = QCheckBox("Background Extraction")
        self.bg_extract_check.setToolTip(bg_extract_tooltip)
        preprocessing_layout.addWidget(self.bg_extract_check)

        drizzle_tooltip = "Drizzle integration can improve resolution but increases processing time and file size. Use values above 1.0 with caution."
        self.drizzle_checkbox = QCheckBox("Enable Drizzle")
        self.drizzle_checkbox.setToolTip(drizzle_tooltip)
        preprocessing_layout.addWidget(self.drizzle_checkbox)

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
        preprocessing_layout.addLayout(drizzle_amount_layout)

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
        preprocessing_layout.addLayout(pixel_fraction_layout)

        self.drizzle_checkbox.toggled.connect(self.pixel_fraction_spinbox.setEnabled)

        self.astrometry_net_check = QCheckBox(
            "Fall back to local astrometry.net for unsolved frames (slower)"
        )
        self.astrometry_net_check.setToolTip(
            "After Siril's built-in sequence solve reports a failure, retry only "
            "frames without an existing astrometric solution using local "
            "astrometry.net. Requires suitable astrometry.net index files."
        )
        if self.astrometry_net_available:
            preprocessing_layout.addWidget(self.astrometry_net_check)

        preprocessing_group.setLayout(preprocessing_layout)
        processing_layout.addWidget(preprocessing_group)

        # Registration settings
        filter_group = QGroupBox("Optional Filter Settings")
        filter_layout = QVBoxLayout()

        # Roundness filter
        roundness_label_tooltip = "Filters images by star roundness, calculated using the second moments of detected stars. \nA lower roundness value applies a stricter filter, keeping only frames with well-defined, circular stars. Higher roundness values allow more variation in star shapes."
        roundness_layout = QHBoxLayout()
        self.roundness_check = QCheckBox("Filter Roundness:")
        self.roundness_check.setToolTip(roundness_label_tooltip)
        self.roundness_spinbox = QDoubleSpinBox()
        self.roundness_spinbox.setRange(1, 4)
        self.roundness_spinbox.setSingleStep(0.1)
        self.roundness_spinbox.setValue(UI_DEFAULTS["filter_round"])
        self.roundness_spinbox.setDecimals(1)
        self.roundness_spinbox.setMinimumWidth(80)
        self.roundness_spinbox.setSuffix(" σ")
        self.roundness_spinbox.setEnabled(False)
        self.roundness_spinbox.setToolTip(roundness_label_tooltip)
        self.roundness_mode_combo = QComboBox()
        self.roundness_mode_combo.addItems(["σ", "%", "abs"])
        self.roundness_mode_combo.setFixedWidth(65)
        self.roundness_mode_combo.setEnabled(False)
        self.roundness_check.toggled.connect(self.roundness_spinbox.setEnabled)
        self.roundness_check.toggled.connect(self.roundness_mode_combo.setEnabled)
        self.roundness_mode_combo.currentTextChanged.connect(
            lambda _: self._on_filter_mode_changed(
                self.roundness_mode_combo, self.roundness_spinbox
            )
        )
        roundness_layout.addWidget(self.roundness_check)
        roundness_layout.addWidget(self.roundness_spinbox)
        roundness_layout.addWidget(self.roundness_mode_combo)
        filter_layout.addLayout(roundness_layout)

        # FWHM filter
        fwhm_label_tooltip = "Filters images by weighted Full Width at Half Maximum (FWHM), calculated using star sharpness. \nA lower sigma value applies a stricter filter, keeping only frames close to the median FWHM. Higher sigma allows more variation."
        fwhm_layout = QHBoxLayout()
        self.fwhm_check = QCheckBox("Filter FWHM:")
        self.fwhm_check.setToolTip(fwhm_label_tooltip)
        self.fwhm_spinbox = QDoubleSpinBox()
        self.fwhm_spinbox.setRange(1, 4)
        self.fwhm_spinbox.setSingleStep(0.1)
        self.fwhm_spinbox.setValue(UI_DEFAULTS["filter_wfwhm"])
        self.fwhm_spinbox.setDecimals(1)
        self.fwhm_spinbox.setMinimumWidth(80)
        self.fwhm_spinbox.setSuffix(" σ")
        self.fwhm_spinbox.setEnabled(False)
        self.fwhm_spinbox.setToolTip(fwhm_label_tooltip)
        self.fwhm_mode_combo = QComboBox()
        self.fwhm_mode_combo.addItems(["σ", "%", "abs"])
        self.fwhm_mode_combo.setFixedWidth(65)
        self.fwhm_mode_combo.setEnabled(False)
        self.fwhm_check.toggled.connect(self.fwhm_spinbox.setEnabled)
        self.fwhm_check.toggled.connect(self.fwhm_mode_combo.setEnabled)
        self.fwhm_mode_combo.currentTextChanged.connect(
            lambda _: self._on_filter_mode_changed(
                self.fwhm_mode_combo, self.fwhm_spinbox
            )
        )
        fwhm_layout.addWidget(self.fwhm_check)
        fwhm_layout.addWidget(self.fwhm_spinbox)
        fwhm_layout.addWidget(self.fwhm_mode_combo)
        filter_layout.addLayout(fwhm_layout)

        # Star count filter
        stars_label_tooltip = "Filters images by star count. Frames with significantly fewer stars than the median are excluded. \nA lower sigma value applies a stricter filter. Higher sigma allows more variation."
        stars_layout = QHBoxLayout()
        self.stars_check = QCheckBox("Filter Star Count:")
        self.stars_check.setToolTip(stars_label_tooltip)
        self.stars_spinbox = QDoubleSpinBox()
        self.stars_spinbox.setRange(1, 4)
        self.stars_spinbox.setSingleStep(0.1)
        self.stars_spinbox.setValue(UI_DEFAULTS["filter_stars"])
        self.stars_spinbox.setDecimals(1)
        self.stars_spinbox.setMinimumWidth(80)
        self.stars_spinbox.setSuffix(" σ")
        self.stars_spinbox.setEnabled(False)
        self.stars_spinbox.setToolTip(stars_label_tooltip)
        self.stars_mode_combo = QComboBox()
        self.stars_mode_combo.addItems(["σ", "%", "abs"])
        self.stars_mode_combo.setFixedWidth(65)
        self.stars_mode_combo.setEnabled(False)
        self.stars_check.toggled.connect(self.stars_spinbox.setEnabled)
        self.stars_check.toggled.connect(self.stars_mode_combo.setEnabled)
        self.stars_mode_combo.currentTextChanged.connect(
            lambda _: self._on_filter_mode_changed(
                self.stars_mode_combo, self.stars_spinbox
            )
        )
        stars_layout.addWidget(self.stars_check)
        stars_layout.addWidget(self.stars_spinbox)
        stars_layout.addWidget(self.stars_mode_combo)
        filter_layout.addLayout(stars_layout)

        # Background filter
        bkg_label_tooltip = "Filters images by background level. Frames with a significantly higher background than the median are excluded. \nA lower sigma value applies a stricter filter. Higher sigma allows more variation."
        bkg_layout = QHBoxLayout()
        self.bkg_check = QCheckBox("Filter Background:")
        self.bkg_check.setToolTip(bkg_label_tooltip)
        self.bkg_spinbox = QDoubleSpinBox()
        self.bkg_spinbox.setRange(1, 4)
        self.bkg_spinbox.setSingleStep(0.1)
        self.bkg_spinbox.setValue(UI_DEFAULTS["filter_bkg"])
        self.bkg_spinbox.setDecimals(1)
        self.bkg_spinbox.setMinimumWidth(80)
        self.bkg_spinbox.setSuffix(" σ")
        self.bkg_spinbox.setEnabled(False)
        self.bkg_spinbox.setToolTip(bkg_label_tooltip)
        self.bkg_mode_combo = QComboBox()
        self.bkg_mode_combo.addItems(["σ", "%", "abs"])
        self.bkg_mode_combo.setFixedWidth(65)
        self.bkg_mode_combo.setEnabled(False)
        self.bkg_check.toggled.connect(self.bkg_spinbox.setEnabled)
        self.bkg_check.toggled.connect(self.bkg_mode_combo.setEnabled)
        self.bkg_mode_combo.currentTextChanged.connect(
            lambda _: self._on_filter_mode_changed(
                self.bkg_mode_combo, self.bkg_spinbox
            )
        )
        bkg_layout.addWidget(self.bkg_check)
        bkg_layout.addWidget(self.bkg_spinbox)
        bkg_layout.addWidget(self.bkg_mode_combo)
        filter_layout.addLayout(bkg_layout)

        filter_group.setLayout(filter_layout)
        processing_layout.addWidget(filter_group)

        # Stacking settings
        stacking_group = QGroupBox("Stacking Settings")
        stacking_layout = QVBoxLayout()

        feather_tooltip = "Blends the edges of stacked frames to reduce edge artifacts in the final image."
        feather_amount_tooltip = "Size of the feathering blend in pixels. Larger values create smoother transitions but may affect more of the image edge."
        feather_layout = QHBoxLayout()
        self.feather_checkbox = QCheckBox("Enable Feather:")
        self.feather_checkbox.setToolTip(feather_tooltip)
        self.feather_amount_spinbox = QSpinBox()
        self.feather_amount_spinbox.setRange(5, 2000)
        self.feather_amount_spinbox.setSingleStep(5)
        self.feather_amount_spinbox.setValue(UI_DEFAULTS["feather_amount"])
        self.feather_amount_spinbox.setMinimumWidth(80)
        self.feather_amount_spinbox.setSuffix(" px")
        self.feather_amount_spinbox.setEnabled(False)
        self.feather_amount_spinbox.setToolTip(feather_amount_tooltip)
        feather_layout.addWidget(self.feather_checkbox)
        feather_layout.addWidget(self.feather_amount_spinbox)
        stacking_layout.addLayout(feather_layout)

        self.feather_checkbox.toggled.connect(self.feather_amount_spinbox.setEnabled)

        weight_tooltip = "Weight frames during stacking to bias the result toward higher-quality frames."
        weight_method_tooltip = "Weighting method: Number of Stars (more stars = more weight), Weighted FWHM (sharper = more weight), Noise (less noise = more weight)."
        weight_layout = QHBoxLayout()
        self.weight_stack_check = QCheckBox("Weight stacking:")
        self.weight_stack_check.setToolTip(weight_tooltip)
        self.weight_method_combo = QComboBox()
        self.weight_method_combo.addItems(["Number of Stars", "Weighted FWHM", "Noise"])
        self.weight_method_combo.setCurrentText("Weighted FWHM")
        self.weight_method_combo.setEnabled(False)
        self.weight_method_combo.setToolTip(weight_method_tooltip)
        self.weight_stack_check.toggled.connect(self.weight_method_combo.setEnabled)
        weight_layout.addWidget(self.weight_stack_check)
        weight_layout.addWidget(self.weight_method_combo)
        stacking_layout.addLayout(weight_layout)

        save_calibrated_lights_tooltip = "Save calibrated light frames after processing. Allows you to collect everything even if you don't create stacks immediately."
        self.save_calibrated_lights_check = QCheckBox("Save calibrated lights")
        self.save_calibrated_lights_check.setToolTip(save_calibrated_lights_tooltip)
        stacking_layout.addWidget(self.save_calibrated_lights_check)

        output_norm_tooltip = "Normalize the output stack so pixel values are scaled relatively. Recommended for most use cases. Turn it OFF for photometry or if you see strange artifacts in where your background is clipping to zero."
        self.output_norm_check = QCheckBox("Output normalization")
        self.output_norm_check.setToolTip(output_norm_tooltip)
        self.output_norm_check.setChecked(True)
        stacking_layout.addWidget(self.output_norm_check)

        # Target mode radio buttons
        target_mode_box = QGroupBox("Target Mode")
        target_mode_layout = QVBoxLayout()

        self.single_target_radio = QRadioButton("Single target")
        self.single_target_radio.setToolTip(
            "All sessions are of the same target and will be combined into a single final stack."
        )
        self.single_target_radio.setChecked(True)

        self.multi_target_radio = QRadioButton(
            "Multi target (Do not combine into final stack)"
        )
        self.multi_target_radio.setToolTip(
            "Each session is a different target. Sessions are stacked individually — no combined stack is produced."
        )
        self.multi_target_radio.setEnabled(len(self.sessions) > 1)

        self.paneled_mosaic_radio = QRadioButton("Paneled mosaic")
        self.paneled_mosaic_radio.setToolTip(
            "Sessions are mosaic panels of the same target. Each session is stacked individually, then stitched into a mosaic. No drizzle or filters applied during stitching. Applies Overlap Normalization so your panels MUST have overlaps."
        )
        self.paneled_mosaic_radio.setEnabled(len(self.sessions) > 1)

        self.mono_radio = QRadioButton("Mono (Experimental)")
        self.mono_radio.setToolTip(
            "Experimental: Process images as monochrome (no debayering). Use only for monochrome cameras or special processing needs. Sessions are processed individually — no combined stack is produced."
        )
        # Alias so all existing self.mono_check.isChecked() references keep working
        self.mono_check = self.mono_radio

        self.target_mode_button_group = QButtonGroup()
        self.target_mode_button_group.addButton(self.single_target_radio)
        self.target_mode_button_group.addButton(self.multi_target_radio)
        self.target_mode_button_group.addButton(self.paneled_mosaic_radio)
        self.target_mode_button_group.addButton(self.mono_radio)

        target_mode_layout.addWidget(self.single_target_radio)
        target_mode_layout.addWidget(self.multi_target_radio)
        target_mode_layout.addWidget(self.paneled_mosaic_radio)
        target_mode_layout.addWidget(self.mono_radio)
        target_mode_box.setLayout(target_mode_layout)
        stacking_layout.addWidget(target_mode_box)

        self.target_mode_button_group.buttonToggled.connect(self.on_target_mode_changed)

        create_final_stack_tooltip = "Create a final stack by combining all preprocessed lights. Automatically enabled and locked for paneled mosaic."

        self.create_final_stack_check = QCheckBox("Create final stack")
        self.create_final_stack_check.setToolTip(create_final_stack_tooltip)
        self.create_final_stack_check.setChecked(True)
        stacking_layout.addWidget(self.create_final_stack_check)

        stacking_group.setLayout(stacking_layout)
        processing_layout.addWidget(stacking_group)

        scroll_area.setWidget(scroll_content)
        processing_tab_outer.addWidget(scroll_area)

        # Process button (always visible, outside scroll area)
        process_btn = QPushButton("Start Preprocessing")
        process_btn.setMinimumHeight(38)
        process_btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: #ffffff;"
            " border: none; border-radius: 4px; font-weight: bold; font-size: 13px; }"
            " QPushButton:hover { background-color: #1d4ed8; }"
            " QPushButton:pressed { background-color: #1e40af; }"
        )
        process_btn.clicked.connect(
            lambda: self.run_script(
                bg_extract=self.bg_extract_check.isChecked(),
                use_astrometry_net=(
                    self.astrometry_net_available
                    and self.astrometry_net_check.isChecked()
                ),
                drizzle=self.drizzle_checkbox.isChecked(),
                drizzle_amount=round(self.drizzle_amount_spinbox.value(), 1),
                pixel_fraction=round(self.pixel_fraction_spinbox.value(), 2),
                feather=self.feather_checkbox.isChecked(),
                feather_amount=round(self.feather_amount_spinbox.value(), 0),
                filter_round=round(self.roundness_spinbox.value(), 1),
                filter_wfwhm=round(self.fwhm_spinbox.value(), 1),
                filter_stars=round(self.stars_spinbox.value(), 1),
                filter_bkg=round(self.bkg_spinbox.value(), 1),
                use_filter_round=self.roundness_check.isChecked(),
                use_filter_wfwhm=self.fwhm_check.isChecked(),
                use_filter_stars=self.stars_check.isChecked(),
                use_filter_bkg=self.bkg_check.isChecked(),
                clean_up_files=self.cleanup_check.isChecked(),
                process_separately=self.multi_target_radio.isChecked()
                or self.paneled_mosaic_radio.isChecked(),
                save_calibrated_lights=self.save_calibrated_lights_check.isChecked(),
                paneled_mosaic=self.paneled_mosaic_radio.isChecked(),
                stack_weighted=self.weight_stack_check.isChecked(),
                weighting_method=self.weight_method_combo.currentText(),
                output_norm=self.output_norm_check.isChecked(),
            )
        )
        processing_tab_outer.addWidget(process_btn)

        # Add tabs
        tab_widget.addTab(files_tab, "1. Files")
        tab_widget.addTab(processing_tab, "2. Processing")
        self.tab_widget = tab_widget
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

        save_presets_button = QToolButton()
        save_presets_button.setText("Save Presets")
        save_presets_button.setMinimumWidth(100)
        save_presets_button.setMinimumHeight(35)
        save_presets_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        save_presets_button.clicked.connect(self.save_presets)
        save_menu = QMenu(save_presets_button)
        save_as_action = QAction("Save As...", self)
        save_as_action.triggered.connect(self.save_presets_as)
        save_menu.addAction(save_as_action)
        save_presets_button.setMenu(save_menu)
        button_layout.addWidget(save_presets_button)

        load_presets_button = QToolButton()
        load_presets_button.setText("Load Presets")
        load_presets_button.setMinimumWidth(100)
        load_presets_button.setMinimumHeight(35)
        load_presets_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        load_presets_button.clicked.connect(self.load_presets)
        load_menu = QMenu(load_presets_button)
        load_from_action = QAction("Load From...", self)
        load_from_action.triggered.connect(self.load_presets_from)
        load_menu.addAction(load_from_action)
        load_presets_button.setMenu(load_menu)
        button_layout.addWidget(load_presets_button)

        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.setMinimumHeight(35)
        close_button.clicked.connect(self.close_dialog)
        button_layout.addWidget(close_button)

        _NAV_STYLE = (
            "QPushButton {"
            "  background-color: #ffffff;"
            "  color: #000000;"
            "  border: 2px solid #1e3a8a;"
            "  border-radius: 4px;"
            "  font-weight: 800;"
            "  font-size: 13px;"
            "  padding: 0 14px;"
            "}"
            "QPushButton:hover { background-color: #eff6ff; }"
            "QPushButton:pressed { background-color: #dbeafe; }"
        )

        self.nav_button = QPushButton("Next \u2192")
        self.nav_button.setMinimumWidth(100)
        self.nav_button.setMinimumHeight(35)
        self.nav_button.setStyleSheet(_NAV_STYLE)
        self.nav_button.clicked.connect(self._tab_nav)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        button_layout.addWidget(self.nav_button)

    def _tab_nav(self):
        idx = self.tab_widget.currentIndex()
        last = self.tab_widget.count() - 1
        if idx < last:
            self.tab_widget.setCurrentIndex(idx + 1)
        else:
            self.tab_widget.setCurrentIndex(idx - 1)

    def _on_tab_changed(self, idx: int):
        last = self.tab_widget.count() - 1
        if idx < last:
            self.nav_button.setText("Next \u2192")
        else:
            self.nav_button.setText("\u2190 Back")

    def close_dialog(self):
        try:
            self.siril.disconnect()
        except Exception:
            pass  # Ignore disconnect errors
        self.close()

    def print_footer(self):
        self.siril.log(
            f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            LogColor.GREEN,
        )
        self.siril.log(
            f"""
        Thank you for using the {APP_NAME}!! 
        The author of this script is Nazmus Nasir (Naztronomy).
        Website: https://www.Naztronomy.com 
        YouTube: https://www.YouTube.com/Naztronomy 
        Discord: https://discord.gg/yXKqrawpjr
        Patreon: https://www.patreon.com/c/naztronomy
        Buy me a Coffee: https://www.buymeacoffee.com/naztronomy
        """,
            LogColor.BLUE,
        )

    def update_process_separately_checkbox(self):
        """Update the enabled state of target mode radios based on session count."""
        multi_session = len(self.sessions) > 1
        self.multi_target_radio.setEnabled(multi_session)
        self.paneled_mosaic_radio.setEnabled(multi_session)
        if not multi_session:
            self.single_target_radio.setChecked(True)

    def on_target_mode_changed(self, button, checked):
        """Update create_final_stack state when target mode radio selection changes."""
        if not checked:
            return
        if self.paneled_mosaic_radio.isChecked():
            self.create_final_stack_check.setChecked(True)
            self.create_final_stack_check.setEnabled(False)
        elif self.multi_target_radio.isChecked() or self.mono_radio.isChecked():
            self.create_final_stack_check.setChecked(False)
            self.create_final_stack_check.setEnabled(False)
        else:  # single target
            self.create_final_stack_check.setChecked(True)
            self.create_final_stack_check.setEnabled(True)

    def _on_filter_mode_changed(self, combo, spinbox):
        """Update spinbox properties when filter mode changes between σ, % and abs."""
        mode = combo.currentText()
        if mode == "σ":
            spinbox.setRange(1, 4)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(1)
            spinbox.setValue(3.0)
            spinbox.setSuffix(" σ")
        elif mode == "%":
            spinbox.setRange(1, 100)
            spinbox.setSingleStep(1)
            spinbox.setDecimals(0)
            spinbox.setValue(100)
            spinbox.setSuffix(" %")
        else:  # abs
            spinbox.setRange(0, 1000)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)
            spinbox.setValue(0)
            spinbox.setSuffix("")

    def save_presets(self, filepath=None):
        """Save current UI settings and session data to a preset file.
        If filepath is None, saves to the default location."""
        # Collect settings
        presets = {
            "dark_flats": self.dark_flats_check.isChecked(),
            "bg_extract": self.bg_extract_check.isChecked(),
            "use_astrometry_net": (
                self.astrometry_net_available and self.astrometry_net_check.isChecked()
            ),
            "drizzle": self.drizzle_checkbox.isChecked(),
            "drizzle_amount": round(self.drizzle_amount_spinbox.value(), 1),
            "pixel_fraction": round(self.pixel_fraction_spinbox.value(), 2),
            "feather": self.feather_checkbox.isChecked(),
            "feather_amount": round(self.feather_amount_spinbox.value(), 0),
            "filter_round": round(self.roundness_spinbox.value(), 1),
            "filter_wfwhm": round(self.fwhm_spinbox.value(), 1),
            "filter_stars": round(self.stars_spinbox.value(), 1),
            "filter_bkg": round(self.bkg_spinbox.value(), 1),
            "use_filter_round": self.roundness_check.isChecked(),
            "use_filter_wfwhm": self.fwhm_check.isChecked(),
            "use_filter_stars": self.stars_check.isChecked(),
            "use_filter_bkg": self.bkg_check.isChecked(),
            "filter_round_mode": self.roundness_mode_combo.currentText(),
            "filter_wfwhm_mode": self.fwhm_mode_combo.currentText(),
            "filter_stars_mode": self.stars_mode_combo.currentText(),
            "filter_bkg_mode": self.bkg_mode_combo.currentText(),
            "cleanup": self.cleanup_check.isChecked(),
            "target_mode": (
                "paneled"
                if self.paneled_mosaic_radio.isChecked()
                else (
                    "multi"
                    if self.multi_target_radio.isChecked()
                    else "mono" if self.mono_radio.isChecked() else "single"
                )
            ),
            "create_final_stack": self.create_final_stack_check.isChecked(),
            "save_calibrated_lights": self.save_calibrated_lights_check.isChecked(),
            "output_norm": self.output_norm_check.isChecked(),
            "stack_weighted": self.weight_stack_check.isChecked(),
            "weighting_method": self.weight_method_combo.currentText(),
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

        if not filepath:
            cwd = self.current_working_directory
            if not cwd:
                self.siril.log(
                    "No working directory set - use 'Save As...' to choose a location.",
                    LogColor.SALMON,
                )
                self.save_presets_as()
                return
            presets_dir = os.path.join(cwd, "presets")
            os.makedirs(presets_dir, exist_ok=True)
            filepath = os.path.join(presets_dir, "naztronomy_osc_pp_presets.json")

        try:
            with open(filepath, "w") as f:
                json.dump(presets, f, indent=4)
            self.siril.log(
                f"Saved presets and session data to {filepath}", LogColor.GREEN
            )
        except Exception as e:
            self.siril.log(f"Failed to save presets: {e}", LogColor.RED)

    def save_presets_as(self):
        """Save presets to a user-chosen file location."""
        cwd = self.current_working_directory or ""
        presets_dir = os.path.join(cwd, "presets") if cwd else ""
        if presets_dir:
            os.makedirs(presets_dir, exist_ok=True)
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Presets As",
            presets_dir or cwd or "",
            "JSON Files (*.json);;All Files (*.*)",
        )
        if filepath:
            self.save_presets(filepath=filepath)

    def load_presets(self, filepath=None):
        """Load settings and session data from a preset file.
        If filepath is None, loads from the default location (or shows dialog if not found).
        """
        try:
            if not filepath:
                cwd = self.current_working_directory
                default_presets_file = (
                    os.path.join(cwd, "presets", "naztronomy_osc_pp_presets.json")
                    if cwd
                    else None
                )
                # Only use the default file if it exists and has content
                if (
                    default_presets_file
                    and os.path.exists(default_presets_file)
                    and os.path.getsize(default_presets_file) > 0
                ):
                    filepath = default_presets_file
                else:
                    if default_presets_file and os.path.exists(default_presets_file):
                        self.siril.log(
                            "Default presets file is empty — please choose a file.",
                            LogColor.SALMON,
                        )
                    start_dir = os.path.join(cwd, "presets") if cwd else ""
                    filepath, _ = QFileDialog.getOpenFileName(
                        self,
                        "Load Presets",
                        start_dir,
                        "JSON Files (*.json);;All Files (*.*)",
                    )
                    if not filepath:  # User canceled
                        self.siril.log("Preset loading canceled", LogColor.BLUE)
                        return

            with open(filepath) as f:
                presets = json.load(f)

                # Load UI settings
                self.dark_flats_check.setChecked(presets.get("dark_flats", False))
                self.bg_extract_check.setChecked(presets.get("bg_extract", False))
                self.astrometry_net_check.setChecked(
                    self.astrometry_net_available
                    and presets.get("use_astrometry_net", False)
                )
                self.drizzle_checkbox.setChecked(presets.get("drizzle", False))
                self.drizzle_amount_spinbox.setValue(presets.get("drizzle_amount", 1.0))
                self.pixel_fraction_spinbox.setValue(presets.get("pixel_fraction", 1.0))
                self.feather_checkbox.setChecked(presets.get("feather", False))
                self.feather_amount_spinbox.setValue(presets.get("feather_amount", 20))
                self.roundness_mode_combo.setCurrentText(
                    presets.get("filter_round_mode", "σ")
                )
                self.fwhm_mode_combo.setCurrentText(
                    presets.get("filter_wfwhm_mode", "σ")
                )
                self.stars_mode_combo.setCurrentText(
                    presets.get("filter_stars_mode", "σ")
                )
                self.bkg_mode_combo.setCurrentText(presets.get("filter_bkg_mode", "σ"))
                self.roundness_spinbox.setValue(presets.get("filter_round", 3.0))
                self.fwhm_spinbox.setValue(presets.get("filter_wfwhm", 3.0))
                self.stars_spinbox.setValue(presets.get("filter_stars", 3.0))
                self.bkg_spinbox.setValue(presets.get("filter_bkg", 3.0))
                self.roundness_check.setChecked(presets.get("use_filter_round", False))
                self.fwhm_check.setChecked(presets.get("use_filter_wfwhm", False))
                self.stars_check.setChecked(presets.get("use_filter_stars", False))
                self.bkg_check.setChecked(presets.get("use_filter_bkg", False))
                self.cleanup_check.setChecked(presets.get("cleanup", False))
                target_mode = presets.get("target_mode", "single")
                # Support legacy presets that used boolean fields
                if target_mode == "single" and presets.get("paneled_mosaic", False):
                    target_mode = "paneled"
                elif target_mode == "single" and presets.get(
                    "process_separately", False
                ):
                    target_mode = "multi"
                elif target_mode == "single" and presets.get("mono", False):
                    target_mode = "mono"
                if target_mode == "paneled" and len(self.sessions) > 1:
                    self.paneled_mosaic_radio.setChecked(True)
                elif target_mode == "multi" and len(self.sessions) > 1:
                    self.multi_target_radio.setChecked(True)
                elif target_mode == "mono":
                    self.mono_radio.setChecked(True)
                else:
                    self.single_target_radio.setChecked(True)
                self.create_final_stack_check.setChecked(
                    presets.get("create_final_stack", True)
                )
                self.save_calibrated_lights_check.setChecked(
                    presets.get("save_calibrated_lights", False)
                )
                self.output_norm_check.setChecked(presets.get("output_norm", True))
                self.weight_stack_check.setChecked(presets.get("stack_weighted", False))
                self.weight_method_combo.setCurrentText(
                    presets.get("weighting_method", "Weighted FWHM")
                )

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
                    self.update_process_separately_checkbox()

                self.siril.log(
                    f"Loaded presets and {len(sessions_data)} sessions from {filepath}",
                    LogColor.GREEN,
                )
        except Exception as e:
            self.siril.log(f"Error loading presets: {str(e)}", LogColor.RED)

    def load_presets_from(self):
        """Load presets from a user-chosen file."""
        cwd = self.current_working_directory or ""
        presets_dir = os.path.join(cwd, "presets") if cwd else ""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Presets From",
            (presets_dir if presets_dir and os.path.exists(presets_dir) else cwd),
            "JSON Files (*.json);;All Files (*.*)",
        )
        if filepath:
            self.load_presets(filepath=filepath)

    # TODO: Replace paneled_mosaic boolean with target_mode enum for clarity and scalability
    def run_script(
        self,
        bg_extract: bool = False,
        use_astrometry_net: bool = False,
        drizzle: bool = False,
        drizzle_amount: float = UI_DEFAULTS["drizzle_amount"],
        pixel_fraction: float = UI_DEFAULTS["pixel_fraction"],
        feather: bool = False,
        feather_amount: float = UI_DEFAULTS["feather_amount"],
        filter_round: float = UI_DEFAULTS["filter_round"],
        filter_wfwhm: float = UI_DEFAULTS["filter_wfwhm"],
        filter_stars: float = UI_DEFAULTS["filter_stars"],
        filter_bkg: float = UI_DEFAULTS["filter_bkg"],
        use_filter_round: bool = False,
        use_filter_wfwhm: bool = False,
        use_filter_stars: bool = False,
        use_filter_bkg: bool = False,
        clean_up_files: bool = False,
        process_separately: bool = False,
        save_calibrated_lights: bool = False,
        paneled_mosaic: bool = False,
        stack_weighted: bool = False,
        weighting_method: str = "Weighted FWHM",
        output_norm: bool = True,
    ):
        # ── Pre-flight: check every session has at least 2 light frames ──────
        problem_sessions = []
        for i, session in enumerate(self.sessions):
            if len(session.lights) == 0:
                problem_sessions.append((i + 1, len(session.lights), "no lights"))
            elif len(session.lights) == 1:
                problem_sessions.append((i + 1, len(session.lights), "only 1 light"))

        if problem_sessions:
            lines = "".join(
                f"• Session {n}: {reason} - at least 2 are required\n"
                for n, _, reason in problem_sessions
            )
            QMessageBox.warning(
                self,
                "Not Enough Light Frames",
                f"The following session(s) do not have enough light frames to stack:\n\n"
                f"{lines}\n"
                f"Each session needs at least 2 light frames.",
            )
            return

        self.siril.log(
            f"Running script version {VERSION} with arguments:\n"
            f"dark_flats={self.dark_flats_check.isChecked()}\n"
            f"bg_extract={bg_extract}\n"
            f"use_astrometry_net={use_astrometry_net}\n"
            f"drizzle={drizzle}\n"
            f"drizzle_amount={drizzle_amount}\n"
            f"pixel_fraction={pixel_fraction}\n"
            f"feather={feather}\n"
            f"feather_amount={feather_amount}\n"
            f"filter_round={filter_round} (enabled={use_filter_round})\n"
            f"filter_wfwhm={filter_wfwhm} (enabled={use_filter_wfwhm})\n"
            f"filter_stars={filter_stars} (enabled={use_filter_stars})\n"
            f"filter_bkg={filter_bkg} (enabled={use_filter_bkg})\n"
            f"clean_up_files={clean_up_files}\n"
            f"process_separately={process_separately}\n"
            f"save_calibrated_lights={save_calibrated_lights}\n"
            f"paneled_mosaic={paneled_mosaic}\n"
            f"create_final_stack={self.create_final_stack_check.isChecked()}\n"
            f"save_calibrated_lights={self.save_calibrated_lights_check.isChecked()}\n"
            f"target_mode={'paneled mosaic' if self.paneled_mosaic_radio.isChecked() else 'multi target' if self.multi_target_radio.isChecked() else 'mono' if self.mono_radio.isChecked() else 'single target'}\n"
            f"stack_weighted={stack_weighted} method={weighting_method}\n"
            f"output_norm={output_norm}\n"
            f"build={VERSION}-{BUILD}",
            LogColor.BLUE,
        )
        self.siril.cmd("close")

        # Check if old processing directories exist
        if (
            os.path.exists("sessions")
            or os.path.exists("process")
            or os.path.exists("collected_lights")
            or os.path.exists("mono_stacks")
            or os.path.exists("individual_stacks")
            or os.path.exists("paneled_mosaic_process")
            or os.path.exists("final_stack_process")
        ):
            msg = """One or more old processing directories found (sessions, process, collected_lights, mono_stacks, individual_stacks, paneled_mosaic_process). 
                \nDo you want to delete them and start fresh?
                \nNote: There is no way to recover this data if you choose 'Yes'."""
            answer = QMessageBox.question(
                self,
                "Old Processing Files Found",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                if os.path.exists("sessions"):
                    shutil.rmtree("sessions", ignore_errors=True)
                    self.siril.log("Cleaned up old sessions directories", LogColor.BLUE)
                if os.path.exists("process"):
                    shutil.rmtree("process", ignore_errors=True)
                    self.siril.log("Cleaned up old process directory", LogColor.BLUE)
                if os.path.exists("collected_lights"):
                    shutil.rmtree("collected_lights", ignore_errors=True)
                    self.siril.log(
                        "Cleaned up old collected_lights directory", LogColor.BLUE
                    )
                if os.path.exists("mono_stacks"):
                    shutil.rmtree("mono_stacks", ignore_errors=True)
                    self.siril.log(
                        "Cleaned up old mono_stacks directory", LogColor.BLUE
                    )
                if os.path.exists("individual_stacks"):
                    shutil.rmtree("individual_stacks", ignore_errors=True)
                    self.siril.log(
                        "Cleaned up old individual_stacks directory", LogColor.BLUE
                    )
                if os.path.exists("paneled_mosaic_process"):
                    shutil.rmtree("paneled_mosaic_process", ignore_errors=True)
                    self.siril.log(
                        "Cleaned up old paneled_mosaic_process directory", LogColor.BLUE
                    )
                if os.path.exists("final_stack_process"):
                    shutil.rmtree("final_stack_process", ignore_errors=True)
                    self.siril.log(
                        "Cleaned up old final_stack_process directory", LogColor.BLUE
                    )
            else:
                self.siril.log(
                    "User chose to preserve old processing files. Stopping script.",
                    LogColor.BLUE,
                )
                return
        # Check files - if more than 2048, batch them:
        self.drizzle_status = drizzle
        self.drizzle_factor = drizzle_amount

        # Check files in working directory/lights.
        # create sub folders with more than 2048 divided by equal amounts

        # Get all sessions
        session_to_process = self.get_all_sessions()

        # True when user wants a final stack but hasn't opted to save calibrated lights.
        # Each session is stacked individually first, then combined at the end.
        needs_per_session_stack = (
            self.single_target_radio.isChecked()
            and self.create_final_stack_check.isChecked()
            and not save_calibrated_lights
            and not self.mono_check.isChecked()
        )

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
                    converted = self.convert_files(image_type=image_type)
                    if converted:
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

            # Current directory where files are located
            current_dir = os.path.join(self.current_working_directory, "process")

            # Only save calibrated lights if requested
            if save_calibrated_lights and os.path.isdir(current_dir):
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
                        dest_path = os.path.join(
                            self.collected_lights_dir, new_file_name
                        )

                        shutil.copy2(src_path, dest_path)
                        self.siril.log(
                            f"Moved {file_name} to {self.collected_lights_dir} as {new_file_name}",
                            LogColor.BLUE,
                        )
            else:
                self.siril.log(
                    "Skipping save of calibrated lights (save_calibrated_lights is unchecked)",
                    LogColor.BLUE,
                )

            # Process separately if requested or mono is selected
            # IF paneled mosaic, create the individual stacks dir and images in there for later processing
            if (
                process_separately
                or self.mono_check.isChecked()
                or needs_per_session_stack
            ):
                # Create individual_stacks directory
                dirname = (
                    "mono_stacks"
                    if self.mono_check.isChecked()
                    else "individual_stacks"
                )
                individual_stacks_dir = os.path.join(self.home_directory, dirname)
                os.makedirs(individual_stacks_dir, exist_ok=True)

                # Process this session individually
                individual_seq_name = "pp_lights_"
                # self.siril.create_new_seq(individual_seq_name)

                if bg_extract:
                    self.seq_bg_extract(seq_name=individual_seq_name)
                    individual_seq_name = "bkg_" + individual_seq_name

                individual_plate_solve_status = self.seq_plate_solve(
                    seq_name=individual_seq_name,
                    use_astrometry_net=use_astrometry_net,
                )

                if individual_plate_solve_status:
                    self.seq_apply_reg(
                        seq_name=individual_seq_name,
                        drizzle_amount=drizzle_amount,
                        pixel_fraction=pixel_fraction,
                        filter_wfwhm=filter_wfwhm,
                        filter_round=filter_round,
                        filter_stars=filter_stars,
                        filter_bkg=filter_bkg,
                        use_filter_round=use_filter_round,
                        use_filter_wfwhm=use_filter_wfwhm,
                        use_filter_stars=use_filter_stars,
                        use_filter_bkg=use_filter_bkg,
                    )
                else:
                    # If Siril can't plate solve, we apply regular registration with 2pass and then apply registration with max framing
                    self.regular_register_seq(
                        seq_name=individual_seq_name,
                        drizzle_amount=drizzle_amount,
                        pixel_fraction=pixel_fraction,
                    )
                    self.seq_apply_reg(
                        seq_name=individual_seq_name,
                        drizzle_amount=drizzle_amount,
                        pixel_fraction=pixel_fraction,
                        filter_wfwhm=filter_wfwhm,
                        filter_round=filter_round,
                        filter_stars=filter_stars,
                        filter_bkg=filter_bkg,
                        use_filter_round=use_filter_round,
                        use_filter_wfwhm=use_filter_wfwhm,
                        use_filter_stars=use_filter_stars,
                        use_filter_bkg=use_filter_bkg,
                    )

                individual_seq_name = f"r_{individual_seq_name}"

                # Scans for black frames due to existing Siril bug.
                # if drizzle:
                #     self.scan_black_frames(seq_name=individual_seq_name, folder="process")

                # Stack this individual session
                individual_stack_name = f"{session_name}_stacked"
                self.seq_stack(
                    seq_name=individual_seq_name,
                    feather=feather,
                    feather_amount=feather_amount,
                    rejection=True,
                    output_name=individual_stack_name,
                    overlap_norm=False,
                    output_norm=output_norm,
                    stack_weighted=stack_weighted,
                    weighting_method=weighting_method,
                )

                # Save individual stack
                self.load_image(image_name=individual_stack_name)
                individual_file_name = self.save_image(f"_{session_name}")
                # Remove any quotes from the filename
                individual_file_name = individual_file_name.strip("'\"")
                self.siril.log(
                    f"Saved individual stack as {individual_file_name}", LogColor.GREEN
                )
                # Move individual stack to individual_stacks directory
                src_individual = os.path.join(
                    self.current_working_directory,
                    "process",
                    f"{individual_file_name}{self.fits_extension}",
                )
                new_dst_filename = (
                    "mono_" + individual_file_name
                    if self.mono_check.isChecked()
                    else individual_file_name
                )
                dst_individual = os.path.join(
                    individual_stacks_dir, f"{new_dst_filename}{self.fits_extension}"
                )
                if os.path.exists(src_individual):
                    shutil.move(src_individual, dst_individual)
                    self.siril.log(
                        f"Moved {new_dst_filename} to individual_stacks directory",
                        LogColor.BLUE,
                    )
                else:
                    self.siril.log(
                        f"Source file not found: {src_individual}", LogColor.RED
                    )

                self.siril.cmd("close")
                time.sleep(3)  # Small delay to ensure Siril processes the command

            # Go back to the previous directory
            self.siril.cmd("cd", "../../..")
            self.current_working_directory = self.siril.get_siril_wd()
            # If clean up is selected, delete the session# directories one after another.
            if clean_up_files:
                shutil.rmtree(
                    os.path.join(
                        self.current_working_directory, "sessions", session_name
                    ),
                    ignore_errors=True,
                )

            self.siril.cmd("close")
            time.sleep(3)  # Small delay to ensure Siril processes the command

        if self.mono_check.isChecked():
            # TODO: If mono, go into the mono_stacks directory and combine all session stacks into one sequence and register them but not stack
            self.siril.log(
                "Mono checked: " + str(self.mono_check.isChecked()), LogColor.BLUE
            )
            mono_dir = "mono_stacks"
            fits_files = [
                fname
                for fname in os.listdir(mono_dir)
                if fname.startswith("mono_")
                and fname.endswith(self.fits_extension)
                and not fname.startswith(".")
            ]
            self.siril.log(
                f"Found {len(fits_files)} mono_*.fits files in {mono_dir}",
                LogColor.BLUE,
            )
            if len(fits_files) > 1:
                self.siril.cmd("cd", f'"{mono_dir}"')
                cwd = self.siril.get_siril_wd()
                # Move all mono_*.fits files into a "lights" folder
                mono_lights_dir = os.path.join(mono_dir, "lights")
                os.makedirs(mono_lights_dir, exist_ok=True)
                for fname in fits_files:
                    src = os.path.join(mono_dir, fname)
                    dst = os.path.join(mono_lights_dir, fname)
                    shutil.copy2(src, dst)

                # Call the convert command on the lights folder
                args = ["convert", "lights", "-out=../mono_process"]
                self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                self.siril.cmd(*args)

                # Go into the process directory
                self.siril.cmd("cd", "../mono_process")

                # Register and apply registration to the lights_ sequence
                seq_name = "lights_"
                cmd_args = ["register", seq_name, "-2pass"]
                try:
                    self.siril.cmd(*cmd_args)
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(f"Data error occurred: {e}", LogColor.RED)

                cmd_args = ["seqapplyreg", seq_name]

                self.siril.log(
                    "Command arguments: " + " ".join(cmd_args), LogColor.BLUE
                )

                try:
                    self.siril.cmd(*cmd_args)
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(f"Data error occurred: {e}", LogColor.RED)

                self.siril.log(
                    f"Applied existing registration to seq {seq_name}", LogColor.GREEN
                )

                # Read the lights_conversion.txt file
                conversion_file = os.path.join(
                    os.getcwd(), "mono_process", "lights_conversion.txt"
                )
                self.siril.log(
                    f"Looking for lights_conversion.txt in: {os.getcwd()}, {conversion_file}",
                    LogColor.BLUE,
                )
                if os.path.exists(conversion_file):
                    with open(conversion_file, "r") as f:
                        print(f"Opened conversion file: {conversion_file}")
                        for line in f:
                            if "->" in line:
                                src_path, dest_path = line.strip().split(" -> ")
                                src_path = src_path.strip("'")
                                dest_path = dest_path.strip("'")

                                # Get the original filename from the source path
                                original_name = os.path.basename(src_path)

                                # Create new filename with 'r_' prefix
                                new_name = "r_" + original_name
                                # Get the destination file (lights_xxxxx.fits)
                                dest_file = os.path.basename(dest_path)
                                # Full path to the registered file (r_lights_xxxxx.fits)
                                registered_file = os.path.join(
                                    os.getcwd(), "mono_process", "r_" + dest_file
                                )
                                # New destination in mono_stacks
                                final_dest = os.path.join(mono_dir, new_name)
                                # Move the file if it exists
                                if os.path.exists(registered_file):
                                    shutil.move(registered_file, final_dest)
                                    self.siril.log(
                                        f"Moved {registered_file} to {final_dest}",
                                        LogColor.BLUE,
                                    )
                else:
                    self.siril.log("lights_conversion.txt not found", LogColor.SALMON)
                self.siril.cmd("cd", "../")

        if (
            not self.mono_check.isChecked()
            and self.create_final_stack_check.isChecked()
            and save_calibrated_lights
        ):
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
                try:
                    if session_files:
                        self.siril.cmd("merge", *session_files, seq_name)
                        self.siril.log(
                            f"Merged session files: {', '.join(session_files)}",
                            LogColor.GREEN,
                        )
                    else:
                        self.siril.log(
                            "No session files found to merge", LogColor.SALMON
                        )
                except (s.DataError, s.CommandError, s.SirilError) as e:
                    self.siril.log(
                        f"Could not merge sequences. Please see error and stack the individual sessions manually:\n {e}",
                        LogColor.RED,
                    )
            else:
                seq_name = "session1_pp_lights_"

            if bg_extract:
                self.seq_bg_extract(seq_name=seq_name)
                seq_name = "bkg_" + seq_name

            plate_solve_status = self.seq_plate_solve(
                seq_name=seq_name,
                use_astrometry_net=use_astrometry_net,
            )

            if plate_solve_status:
                self.seq_apply_reg(
                    seq_name=seq_name,
                    drizzle_amount=drizzle_amount,
                    pixel_fraction=pixel_fraction,
                    filter_wfwhm=filter_wfwhm,
                    filter_round=filter_round,
                    filter_stars=filter_stars,
                    filter_bkg=filter_bkg,
                    use_filter_round=use_filter_round,
                    use_filter_wfwhm=use_filter_wfwhm,
                    use_filter_stars=use_filter_stars,
                    use_filter_bkg=use_filter_bkg,
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
                    filter_wfwhm=filter_wfwhm,
                    filter_round=filter_round,
                    filter_stars=filter_stars,
                    filter_bkg=filter_bkg,
                    use_filter_round=use_filter_round,
                    use_filter_wfwhm=use_filter_wfwhm,
                    use_filter_stars=use_filter_stars,
                    use_filter_bkg=use_filter_bkg,
                )

            seq_name = f"r_{seq_name}"

            # Scans for black frames due to existing Siril bug.
            try:
                if drizzle:
                    self.scan_black_frames(
                        seq_name=seq_name, folder=self.collected_lights_dir
                    )
            except (s.DataError, s.CommandError, s.SirilError) as e:
                self.siril.log(
                    f"Data error occurred during black frame scan: {e}", LogColor.RED
                )

            # Stacks the sequence with rejection
            stack_name = (
                "merge_stacked" if len(session_to_process) > 1 else "final_stacked"
            )
            self.seq_stack(
                seq_name=seq_name,
                feather=feather,
                feather_amount=feather_amount,
                rejection=True,
                output_name=stack_name,
                overlap_norm=False,
                output_norm=output_norm,
                stack_weighted=stack_weighted,
                weighting_method=weighting_method,
            )

            self.load_image(image_name=stack_name)
            self.siril.cmd("cd", "../")
            self.current_working_directory = self.siril.get_siril_wd()
            file_name = self.save_image("_og")
            self.load_image(image_name=file_name)
        elif (
            not self.mono_check.isChecked()
            and self.create_final_stack_check.isChecked()
            and self.multi_target_radio.isChecked()
        ):
            self.siril.log(
                "Final stack creation skipped due to multi-target mode",
                LogColor.BLUE,
            )
        elif needs_per_session_stack:
            # Single target, create final stack, no save calibrated lights:
            # combine the per-session individual stacks produced above.
            self.siril.log(
                "Building final stack from individual session stacks...",
                LogColor.BLUE,
            )
            individual_stacks_dir = os.path.join(
                self.home_directory, "individual_stacks"
            )
            final_stack_process_dir = os.path.join(
                self.current_working_directory, "final_stack_process"
            )
            os.makedirs(final_stack_process_dir, exist_ok=True)

            fits_files = [
                fname
                for fname in os.listdir(individual_stacks_dir)
                if fname.endswith(self.fits_extension) and not fname.startswith(".")
            ]
            self.siril.log(
                f"Found {len(fits_files)} individual stack(s) to combine: {fits_files}",
                LogColor.BLUE,
            )

            if len(fits_files) == 0:
                self.siril.log(
                    "No individual stacks found, skipping final stack.",
                    LogColor.SALMON,
                )
            elif len(fits_files) == 1:
                # Single session — just load, cd home, save
                self.siril.cmd("cd", f'"{individual_stacks_dir}"')
                self.load_image(image_name=os.path.splitext(fits_files[0])[0])
                self.siril.cmd("cd", f'"{self.home_directory}"')
                self.current_working_directory = self.siril.get_siril_wd()
                file_name = self.save_image("_og")
                self.load_image(image_name=file_name)
                self.siril.log(f"Final stack saved as {file_name}", LogColor.GREEN)
            else:
                # Multiple sessions — copy to final_stack_process, link, plate solve, register, stack
                for fname in fits_files:
                    src = os.path.join(individual_stacks_dir, fname)
                    dst = os.path.join(final_stack_process_dir, fname)
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        self.siril.log(f"Failed to copy {fname}: {e}", LogColor.RED)

                lights_dir = os.path.join(final_stack_process_dir, "lights")
                os.makedirs(lights_dir, exist_ok=True)
                for fname in fits_files:
                    src = os.path.join(final_stack_process_dir, fname)
                    dst = os.path.join(lights_dir, fname)
                    if os.path.exists(src):
                        shutil.move(src, dst)

                self.siril.cmd("cd", f'"{lights_dir}"')
                self.siril.cmd("link", "lights")
                seq_name = "lights_"

                plate_solve_ok = self.seq_plate_solve(
                    seq_name=seq_name,
                    use_astrometry_net=use_astrometry_net,
                )
                if plate_solve_ok:
                    try:
                        self.siril.cmd(
                            "seqapplyreg", seq_name, "-kernel=square", "-framing=max"
                        )
                        seq_name = f"r_{seq_name}"
                    except (s.DataError, s.CommandError, s.SirilError) as e:
                        self.siril.log(
                            f"Could not apply registration: {e}", LogColor.RED
                        )
                else:
                    try:
                        self.siril.cmd("register", seq_name, "-2pass")
                        self.siril.cmd("seqapplyreg", seq_name)
                        seq_name = f"r_{seq_name}"
                    except (s.DataError, s.CommandError, s.SirilError) as e:
                        self.siril.log(
                            f"Could not apply registration: {e}", LogColor.RED
                        )

                self.seq_stack(
                    seq_name=seq_name,
                    feather=feather,
                    feather_amount=feather_amount,
                    rejection=True,
                    output_name="final_stacked",
                    overlap_norm=False,
                    output_norm=output_norm,
                    stack_weighted=stack_weighted,
                    weighting_method=weighting_method,
                )
                self.load_image(image_name="final_stacked")
                self.siril.cmd("cd", f'"{self.home_directory}"')
                self.current_working_directory = self.siril.get_siril_wd()
                file_name = self.save_image("_og")
                self.load_image(image_name=file_name)
                self.siril.log(f"Final stack saved as {file_name}", LogColor.GREEN)
        else:
            self.siril.log("Final stack creation skipped", LogColor.BLUE)

        # Paneled mosaic workflow
        if paneled_mosaic:
            self.siril.log("Starting paneled mosaic creation...", LogColor.BLUE)
            individual_stacks_dir = os.path.join(
                self.home_directory, "individual_stacks"
            )
            paneled_mosaic_process_dir = os.path.join(
                self.current_working_directory, "paneled_mosaic_process"
            )
            os.makedirs(paneled_mosaic_process_dir, exist_ok=True)

            # Check if individual_stacks directory exists with stacks
            if os.path.exists(individual_stacks_dir):
                self.siril.log(
                    f"Looking for individual stacks in: {individual_stacks_dir}",
                    LogColor.BLUE,
                )
                # Look for individual stack files (not prefixed with "mono_" unless mono is checked)
                search_prefix = "mono_" if self.mono_check.isChecked() else ""
                fits_files = [
                    fname
                    for fname in os.listdir(individual_stacks_dir)
                    if fname.endswith(self.fits_extension)
                    and fname.startswith(search_prefix)
                    and not fname.startswith(".")
                ]

                self.siril.log(
                    f"Found {len(fits_files)} individual stack files: {fits_files}",
                    LogColor.BLUE,
                )

                if len(fits_files) > 1:
                    # Copy individual stacks to the paneled_mosaic_process directory
                    self.siril.log(
                        f"Copying {len(fits_files)} stacks to paneled_mosaic_process...",
                        LogColor.BLUE,
                    )
                    for fname in fits_files:
                        src = os.path.join(individual_stacks_dir, fname)
                        dst = os.path.join(paneled_mosaic_process_dir, fname)
                        try:
                            shutil.copy2(src, dst)
                            self.siril.log(f"Copied {fname}", LogColor.BLUE)
                        except Exception as e:
                            self.siril.log(f"Failed to copy {fname}: {e}", LogColor.RED)

                    # Change directory to paneled_mosaic_process
                    self.siril.cmd("cd", f'"{paneled_mosaic_process_dir}"')
                    self.current_working_directory = self.siril.get_siril_wd()

                    # Create a lights folder for conversion
                    lights_dir = os.path.join(paneled_mosaic_process_dir, "lights")
                    os.makedirs(lights_dir, exist_ok=True)
                    self.siril.log(f"Moving files to {lights_dir}...", LogColor.BLUE)
                    for fname in fits_files:
                        src = os.path.join(paneled_mosaic_process_dir, fname)
                        dst = os.path.join(lights_dir, fname)
                        try:
                            if os.path.exists(src):
                                shutil.move(src, dst)
                                self.siril.log(
                                    f"Moved {fname} to lights/", LogColor.BLUE
                                )
                            else:
                                self.siril.log(
                                    f"Source file not found: {src}", LogColor.SALMON
                                )
                        except Exception as e:
                            self.siril.log(f"Failed to move {fname}: {e}", LogColor.RED)

                    # Verify files exist in lights directory
                    lights_files = (
                        os.listdir(lights_dir) if os.path.exists(lights_dir) else []
                    )
                    self.siril.log(
                        f"Lights directory now contains: {lights_files}",
                        LogColor.BLUE,
                    )

                    # Link FITS files to create sequence (cd into lights directory first)
                    self.siril.cmd("cd", f'"{lights_dir}"')
                    args = ["link", "lights"]
                    self.siril.log(" ".join(str(arg) for arg in args), LogColor.GREEN)
                    self.siril.cmd(*args)

                    # All sequence processing happens in the lights directory
                    seq_name = "lights_"

                    # Plate solve the sequence (without filters)
                    self.siril.log(
                        f"Plate solving paneled mosaic sequence {seq_name}...",
                        LogColor.BLUE,
                    )
                    try:
                        self.seq_plate_solve(
                            seq_name=seq_name,
                            use_astrometry_net=use_astrometry_net,
                        )
                        # self.siril.cmd(
                        #     "seqplatesolve",
                        #     seq_name,
                        # )
                        self.siril.log(f"Plate solved {seq_name}", LogColor.GREEN)
                        plate_solve_ok = True
                    except (s.DataError, s.CommandError, s.SirilError) as e:
                        self.siril.log(
                            f"Plate solve failed for paneled mosaic: {e}",
                            LogColor.RED,
                        )
                        plate_solve_ok = False

                    if plate_solve_ok:
                        # Apply registration (no drizzle, no filters)
                        try:
                            self.siril.cmd(
                                "seqapplyreg",
                                seq_name,
                                "-kernel=square",
                                "-framing=max",
                            )
                            self.siril.log(
                                f"Applied registration to {seq_name}",
                                LogColor.GREEN,
                            )
                            seq_name = f"r_{seq_name}"
                        except (s.DataError, s.CommandError, s.SirilError) as e:
                            self.siril.log(
                                f"Could not apply registration: {e}", LogColor.RED
                            )
                    else:
                        # TODO: Test this path
                        # Regular registration if plate solve failed
                        try:
                            self.siril.cmd("register", seq_name, "-2pass")
                            self.siril.cmd("seqapplyreg", seq_name)
                            seq_name = f"r_{seq_name}"
                        except (s.DataError, s.CommandError, s.SirilError) as e:
                            self.siril.log(
                                f"Could not apply regular registration: {e}",
                                LogColor.RED,
                            )

                    # Stack with overlap_norm=True (happens in lights directory)
                    # stack r_lights_  rej none -norm=addscale -output_norm -overlap_norm -rgb_equal -maximize -filter-included -weight=wfwhm -out=paneled_mosaic_stacked -feather=70
                    # Don't reject and don't weigh final panel stack
                    self.seq_stack(
                        seq_name=seq_name,
                        feather=feather,
                        feather_amount=feather_amount,
                        rejection=False,
                        output_name="paneled_mosaic_stacked",
                        overlap_norm=True,
                        output_norm=output_norm,
                        stack_weighted=False,
                        weighting_method=weighting_method,
                    )

                    # self.siril.cmd(
                    #     "stack",
                    #     f"{seq_name}",
                    #     " rej none",
                    #     "-norm=addscale",
                    #     "-output_norm",
                    #     "-overlap_norm",
                    #     "-rgb_equal",
                    #     "-maximize",
                    #     "-filter-included",
                    #     f"-feather={feather_amount}",
                    #     "-out=paneled_mosaic_stacked",
                    # )

                    # # Return to paneled_mosaic_process directory
                    # self.siril.cmd("cd", "..")

                    # Move final stack to home directory
                    try:
                        # Stacked file is in the lights subdirectory
                        paneled_mosaic_file = os.path.join(
                            lights_dir,
                            f"paneled_mosaic_stacked{self.fits_extension}",
                        )
                        if os.path.exists(paneled_mosaic_file):
                            final_location = os.path.join(
                                self.home_directory,
                                f"paneled_mosaic_final{self.fits_extension}",
                            )
                            self.load_image(image_name="paneled_mosaic_stacked")
                            print(rf"Home directory: {self.home_directory}")
                            # home_dir = f"{self.home_directory}"
                            self.siril.cmd("cd", f'"{self.home_directory}"')
                            self.save_image("_paneled_mosaic_final")
                            self.siril.log(
                                f"Paneled mosaic final stack saved to {final_location}",
                                LogColor.GREEN,
                            )
                        else:
                            self.siril.log(
                                f"Could not find paneled_mosaic_stacked{self.fits_extension}",
                                LogColor.RED,
                            )
                    except Exception as e:
                        self.siril.log(
                            f"Error saving paneled mosaic final stack: {e}",
                            LogColor.RED,
                        )

                    # self.siril.cmd("cd", self.current_working_directory)

                else:
                    self.siril.log(
                        "Not enough individual stacks for paneled mosaic (need at least 2)",
                        LogColor.SALMON,
                    )
            else:
                self.siril.log(
                    f"Individual stacks directory not found, skipping paneled mosaic",
                    LogColor.SALMON,
                )

        # Delete the blank sessions dir
        if clean_up_files:
            try:
                shutil.rmtree(
                    os.path.join(self.current_working_directory, "sessions"),
                    ignore_errors=True,
                )
            except Exception as e:
                self.siril.log(
                    f"Error cleaning up sessions directory {os.path.join(self.current_working_directory, 'sessions')}: {e}",
                    LogColor.SALMON,
                )
            extension = self.fits_extension.lstrip(".")
            collected_lights_dir = os.path.join(
                self.current_working_directory, "collected_lights"
            )
            try:
                for filename in os.listdir(collected_lights_dir):
                    file_path = os.path.join(collected_lights_dir, filename)

                    if os.path.isfile(file_path) and not (
                        filename.startswith("session") and filename.endswith(extension)
                    ):
                        os.remove(file_path)
                shutil.rmtree(
                    os.path.join(collected_lights_dir, "cache"), ignore_errors=True
                )
                shutil.rmtree(
                    os.path.join(collected_lights_dir, "drizztmp"), ignore_errors=True
                )

                self.siril.log("Cleaned up collected_lights directory", LogColor.BLUE)
            except Exception as e:
                self.siril.log(
                    f"Collected Lights Dir not found, skipping: {e}", LogColor.SALMON
                )

            if self.mono_check.isChecked():
                shutil.rmtree(
                    os.path.join(self.current_working_directory, "mono_process"),
                    ignore_errors=True,
                )
                shutil.rmtree(
                    os.path.join(
                        self.current_working_directory, "mono_stacks", "lights"
                    ),
                    ignore_errors=True,
                )
                self.siril.log("Cleaned up mono_process directory", LogColor.BLUE)

            # Clean up paneled_mosaic_process but preserve individual_stacks
            paneled_mosaic_process_dir = os.path.join(
                self.current_working_directory, "paneled_mosaic_process"
            )
            if os.path.exists(paneled_mosaic_process_dir):
                try:
                    shutil.rmtree(paneled_mosaic_process_dir, ignore_errors=True)
                    self.siril.log(
                        "Cleaned up paneled_mosaic_process directory", LogColor.BLUE
                    )
                except Exception as e:
                    self.siril.log(
                        f"Could not clean up paneled_mosaic_process: {e}",
                        LogColor.SALMON,
                    )

            try:
                shutil.rmtree(
                    os.path.join(self.current_working_directory, "cache"),
                    ignore_errors=True,
                )
                shutil.rmtree(
                    os.path.join(self.current_working_directory, "drizztmp"),
                    ignore_errors=True,
                )
                self.siril.log(
                    "Cleaned up extra cache and drizztmp directories", LogColor.BLUE
                )
            except Exception as e:
                self.siril.log(
                    f"Cache or drizztmp Dir not found, skipping: {e}", LogColor.SALMON
                )

        # self.clean_up()

        self.print_footer()

        # self.close_dialog()


def main():
    try:
        app = QApplication(sys.argv)
        # qdarktheme.setup_theme()
        qt_themes.set_theme("dracula")
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
