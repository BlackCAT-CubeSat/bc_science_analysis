"""Functions used by multiple files in bc_science_analysis."""

from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from science_analysis.bc_image_analyzer import BCImageAnalysis


def grab_caldb(
    caldb: list[str],
) -> tuple[Optional[str], Optional[Path], Optional[Path]]:
    if len(caldb) == 1:
        return caldb[0], None, None

    if len(caldb) == 2:
        coded_mask = Path(caldb[0])
        teldef = Path(caldb[1])
        return None, coded_mask, teldef

    raise ValueError(
        "You must provide a caldb version, or paths to a coded mask file and teldef file."
    )


def initialize_imager(
    *,
    caldb: list[str],
    use_subpixel: bool = False,
    resolution: int = 1,
    global_balance: bool = False,
    show_frame: bool = False,
    overwrite: bool = False,
) -> BCImageAnalysis:
    caldb_version, coded_mask, teldef = grab_caldb(caldb)
    imager = BCImageAnalysis(
        caldb_version=caldb_version,
        coded_mask_file=coded_mask,
        teldef_file=teldef,
        use_subpixel=use_subpixel,
        resolution=resolution,
        balance_per_det=~global_balance,
        hide_frame=~show_frame,
        overwrite=overwrite,
    )
    return imager


def plot_image(image_hdu: fits.PrimaryHDU) -> tuple[Figure, Axes]:
    """Plot the image within an image hdu on RA/DEC axes if WCS is
    provided, or X/Y axes if not.
    """
    wcs = WCS(image_hdu.header)
    if wcs.has_celestial:
        fig, ax = plt.subplots(subplot_kw={"projection": wcs})
        ax.set_xlabel(f"{ax.get_xlabel().split('.')[-1].upper()} (J2000)")
        ax.set_ylabel(f"{ax.get_ylabel().split('.')[-1].upper()} (J2000)")
    else:
        fig, ax = plt.subplots()
        ax.set_xlabel(f"Image X")
        ax.set_ylabel(f"Image Y")

    plotted_image = ax.imshow(image_hdu.data, origin="lower", aspect="equal")
    fig.colorbar(plotted_image, ax=ax, label="Counts")
    ax.grid(True, color="white", ls="dotted")

    return fig, ax
