# Naztronomy - Siril Scripts

## Naztronomy-Smart_Telescope_PP.py

A comprehensive Python script that automates the preprocessing workflow for smart telescopes, including file conversion, registration, stacking, and SPCC color calibration.

**Supported Telescopes:**

- ZWO Seestar S30
- ZWO Seestar S50
- Dwarf Mini
- Dwarf 2
- Dwarf 3
- Celestron Origin
- Unistellar eVscope 1 / eQuinox 1
- Unistellar eVscope 2 / eQuinox 2
- Unistellar Odyssey / Odyssey Pro

**Features:**

- Allow batching into smaller subset of images to save space and faster processing. Max batch count on Windows is 2,000 and on Mac/Linux is 25,000
- Optional calibration frame support (darks, flats, biases)
- Automatic master frame creation from calibration files
- Drizzle integration for improved resolution
- Background extraction and filtering options
- Spectrophotometric Color Calibration (SPCC) for supported telescopes
- Save/Load presets functionality

**Demo Video:** [YouTube - Smart Telescope Processing](https://www.youtube.com/watch?v=0RqvX9cR8Eg)

## Naztronomy-OSC_PP.py

An advanced OSC (One Shot Color) image preprocessing script designed for processing images from multiple sessions with full mosaic and alignment capabilities.

**Features:**

- Multi-session support with individual file management
- Automatic plate solving and mosaicking for images with proper headers
- Star alignment fallback for images without coordinates
- Session-based organization and processing
- Individual session stacking option in addition to merged stacks
- Master frame creation from single calibration files
- Preprocessed lights collection for later combination
- Experimental mono camera support (no debayering)
- Comprehensive filter settings for image quality control

**Demo Video:** [YouTube - OSC Image Processing](https://www.youtube.com/watch?v=WjabF-bVBmU)

## Installation

Two ways to install these scripts:

1. Manual: Place the Python `.py` files from this repository in your local Siril scripts directory
2. Automatic: Install directly through Siril by going to **Scripts >> Get Scripts** in Siril and searching for "Naztronomy"

## System Requirements

- Siril 1.3.6 or later
- Python packages: PyQt6, numpy, astropy (automatically installed by the scripts)
- Recommended: Blank working directory for clean setup

## Usage Guidelines

### For Smart Telescope Script:

- Must have a `lights` subdirectory in your working directory
- Calibration frames are optional but recommended
- Supports automatic batching for large datasets
- SPCC requires local Gaia catalog for best results

### For OSC Processing Script:

- Can be run from any directory
- Supports multiple sessions with different frame types
- Images are copied/symlinked for processing (requires disk space)
- Individual sessions can be processed separately or combined
- Experimental mono mode bypasses debayering for monochrome cameras

## Limitations

- **Windows File Limit:** The `.ssf` scripts do not work with file counts > 2048 on Windows
- **OSC Optimization:** Current scripts are optimized for OSC cameras (mono support is experimental)
- **Disk Space:** OSC script copies files before processing, requiring additional storage

## Authors

- **Nazmus Nasir** - [Naztronomy](https://www.naztronomy.com)

### Social Media & Support:

- **YouTube** - [YouTube.com/Naztronomy](https://www.youtube.com/naztronomy)
- **Bluesky** - [Bluesky/Naztronomy.com](https://bsky.app/profile/naztronomy.com)
- **Instagram** - [IG/Naztronomy](https://instagram.com/naztronomy)
- **Discord** - [Join our community](https://discord.gg/yXKqrawpjr)
- **Patreon** - [Support development](https://www.patreon.com/c/naztronomy)
- **Buy Me a Coffee** - [One-time support](https://www.buymeacoffee.com/naztronomy)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Support & Questions

Have questions? You can reach out through several channels:

- Create an issue in the [issues forum](/../../issues)
- Join our [Discord community](https://discord.gg/yXKqrawpjr)
- Comment on the demo videos on YouTube
- Check the help sections within each script

## Contributing

Pull requests are welcome! Please ensure you:

- Test your changes thoroughly
- Follow the existing code style
- Update documentation as needed

Bug fixes and new features will be reviewed before merging.

## Deprecated Scripts (.ssf files)

### Naztronomy-Seestar_Broadband_Mosaic.ssf

A legacy Siril script optimized for broadband (UV/IR block) filters. Automates stacking and mosaic creation for Seestar telescope images.

**⚠️ DEPRECATED:** This script is no longer actively supported. Please use the Python scripts above for new projects.

### Naztronomy-Seestar_Narrowband_Mosaic.ssf

A legacy Siril script tailored for narrowband (LP) filters, handling stacking and mosaic generation for Seestar telescope images.

**⚠️ DEPRECATED:** This script is no longer actively supported. Please use the Python scripts above for new projects.
