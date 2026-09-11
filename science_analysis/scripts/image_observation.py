"""Generate an image of observations' stable events."""

import argparse
from os import PathLike
from pathlib import Path
from typing import Optional
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning
import matplotlib.pyplot as plt
import numpy as np

from bc_caldb import CURRENT_CALDB_VER
from science_analysis import BCImageAnalysis, initialize_imager, plot_image


def extract_paths(l1_dir: PathLike | str, obsid: str) -> tuple[Path, Path, Path]:
    """Extract paths to the stable eventlist, attitude file, and base
    observation directory.

    Arguments:
        - l1_dir: Path to the base directory holding level1 fits files.
        - obsid: Observation id to retrieve the paths for.
    """
    yyyy = obsid[:4]
    mm = obsid[4:6]
    dd = obsid[6:8]

    obsid_path = Path(l1_dir) / f"{yyyy}_{mm}" / f"{yyyy}{mm}{dd}" / obsid
    attitude_path = obsid_path / "auxil" / f"bl{obsid}.att.gz"
    events_path = obsid_path / "event" / f"bl{obsid}phpo_uf.evt.gz"

    if not events_path.exists():
        events_path = obsid_path / "event" / f"bl{obsid}pfpo_uf.evt.gz"

    return events_path, attitude_path, obsid_path


def extract_pointing(
    attitudes: Path, stable_timestamp: float
) -> tuple[float, float, float]:
    """Extracts the median pointing during the stable portion of the
    observation.

    Arguments:
        - attitudes: Path to the orientation file
        - stable_timestamp: Timestamp (in BlackCAT mission time) at
        which the observation first stabilized.
    """
    attitudes = fits.getdata(attitudes)
    stable_attitudes = attitudes[attitudes["TIME"] >= stable_timestamp]
    stable_pointings = stable_attitudes["POINTING"]
    ra = stable_pointings[:, 0]
    dec = stable_pointings[:, 1]
    roll = stable_pointings[:, 2]

    return np.median(ra), np.median(dec), np.median(roll)


def image_eventlist(
    events: Path, attitudes: Path, imager: BCImageAnalysis
) -> fits.PrimaryHDU:
    """Produce an image hdu given a stable eventlist, orientation file,
    and BCImageAnalysis object.

    Arguments:
        - events: Path to the stable eventlist file
        - attitudes: Path to the corresponding orientation file
        - imager: BCImageAnalysis object to do the imaging with
    """
    events, header = fits.getdata(events, 1, header=True)

    ra, dec, roll = extract_pointing(
        attitudes, header.get("TSTABLE", np.min(events["TIME"]))
    )

    imager.set_ra_dec_roll(ra, dec, roll)

    return imager.eventlist_to_image(counts=events, header=header, add_wcs=True)


def image_observation(
    *,
    obsids: list[str],
    l1_dir: PathLike | str,
    imager: BCImageAnalysis,
    show: bool = False,
    outdir: Optional[PathLike | str] = None,
) -> list[fits.PrimaryHDU]:
    """Generate an image of observations' stable events.

    Arguments:
        - obsids: List of observations to generate images for.
        - l1_dir: Path to the base directory holding level1 fits files.
        - imager: Initialized BCImageAnalysis object to use for the
        imaging.
        - show: Whether to show a plot of the produced image.
        - outdir: (Optional) Path to save produced images to. Defaults
        to the observation directory if not provided.
    """
    image_hdus = []
    for obsid in obsids:
        if not isinstance(obsid, str):
            raise TypeError("Observation IDs must be provided as a string.")

        if len(obsid) != 12:
            raise ValueError("Observation IDs must be of the form YYYYmmddHHMM.")

        evt_path, att_path, obs_path = extract_paths(l1_dir, obsid)

        image_hdu = image_eventlist(evt_path, att_path, imager)

        outdir = obs_path if outdir is None else Path(outdir)
        outfile = outdir / f"bl{obsid}.img.gz"
        image_hdu.writeto(outfile, overwrite=imager.overwrite, checksum=True)
        image_hdus.append(image_hdu)

        if show:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                fig, ax = plot_image(image_hdu)
                ax.set_title(f"bl{obsid}")
                plt.show()
                plt.close(fig)

    return image_hdus


def main() -> None:
    """Run using command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an image of an observation's stable events."
    )
    parser.add_argument(
        "obsids",
        help="Observation id(s) to image.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--l1_dir",
        help="Directory to where level1 files are stored. Don't include year/month/day/obsid subpaths.",
        type=str,
        default="/archive/science_operations/level1/",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show each image before moving on to next observation id.",
    )
    parser.add_argument(
        "--outdir",
        help="Directory to store output files. Observation path if not specified.",
        type=str,
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
    image_observation(
        obsids=args.obsids,
        l1_dir=args.l1_dir,
        show=args.show,
        outdir=args.outdir,
        imager=imager,
    )


if __name__ == "__main__":
    main()
