"""Find all the peaks in a provided BlackCAT image."""

import argparse
from os import PathLike
from pathlib import Path
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from bc_caldb import CURRENT_CALDB_VER
from science_analysis import BCImageAnalysis, initialize_imager, plot_image


def plot_peaks(
    image_hdu: fits.PrimaryHDU,
    peaks: npt.NDArray[np.void],
) -> tuple[Figure, Axes]:
    """Plot circles around identified peaks in an image."""
    fig, ax = plot_image(image_hdu)

    for peak in peaks:
        image_x, image_y = peak["xy"]
        peak_circle = Circle((image_x, image_y), 20, edgecolor="red", fill=False)
        ax.add_patch(peak_circle)

    ax.set_aspect("equal", adjustable="box")

    return fig, ax


def find_peaks(
    *,
    image_path: PathLike | str,
    imager: BCImageAnalysis,
    show: bool = False,
) -> None:
    """Locate peaks and peak quantities in a provided blackcat image.

    Arguments:
        - image_path: Path to the image to find peaks in.
        - imager: Initialized BCImageAnalysis object to use for the
        peak finding.
        - show: Whether to show a plot of the produced image.
    """
    image_path = Path(image_path)

    with fits.open(image_path) as hdul:
        image_hdu = hdul[0]
        peaks = imager.find_peaks(image_hdu.data, image_hdu.header)

        if show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                fig, ax = plot_peaks(image_hdu, peaks)
                ax.set_title(f"{image_path.name} has {len(peaks)} peaks.")
                plt.show()
                plt.close(fig)


def main() -> None:
    # TODO: Provide capability to save peaks as some sort of file.
    # TODO: Provide capability to set peak significance
    # TODO: Provide capability to set neighborhood psf
    # TODO: Provide capability to view peaks as significance, not counts
    """Run using command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find all the peaks in a provided BlackCAT image."
    )
    parser.add_argument(
        "image_path",
        help="Path to image you want to find peaks in.",
        type=str,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the image with identified peaks circled.",
    )
    parser.add_argument(
        "--caldb",
        help="CalDB version to generate from, or paths to teldef and coded mask files. Current version if not specified.",
        type=str,
        nargs="*",
        default=[CURRENT_CALDB_VER],
    )
    parser.add_argument(
        "--use_subpixel",
        action="store_true",
        help="Use subpixels for imaging, instead of full pixels. Memory intensive.",
    )
    parser.add_argument(
        "--resolution",
        help="Resolution of image (det. [sub]pix projected by focal length). 1=Finest, 8 or 32 [if using subpix] = coarsest.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--global_balance",
        action="store_true",
        help="Balance DPH globally, instead of per-detector.",
    )
    parser.add_argument(
        "--show_frame",
        action="store_true",
        help="Don't hide the frame. Will see notable frame shadow patterns in image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any files with existing names instead of raising exception.",
    )
    args = parser.parse_args()

    imager = initialize_imager(
        caldb=args.caldb,
        use_subpixel=args.use_subpixel,
        resolution=args.resolution,
        global_balance=args.global_balance,
        show_frame=args.show_frame,
        overwrite=args.overwrite,
    )
    find_peaks(
        image_path=args.image_path,
        imager=imager,
        show=args.show,
    )


if __name__ == "__main__":
    main()
