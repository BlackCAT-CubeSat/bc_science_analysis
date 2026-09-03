"""Module for running image-related analysis of BlackCAT eventlists."""

from collections.abc import Collection
from os import PathLike
from pathlib import Path
from typing import Any, Optional, overload
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
import numpy as np
import numpy.typing as npt
from photutils.detection import find_peaks
from photutils.utils.exceptions import NoDetectionsWarning
from scipy.ndimage import uniform_filter

from science_analysis import BCImager


class BCImageAnalysis:
    IMPEAK_DTYPE = np.dtype(
        [
            ("xy", (float, 2)),
            ("radec", (float, 2)),  # Degrees, ICRS
            ("counts", float),
            ("local_rms", float),
            ("global_rms", float),
            ("local_sig", float),
            ("global_sig", float),
        ]
    )

    @overload
    def __init__(
        self,
        *,
        caldb_version: str,
        ra_dec_roll_inst: Optional[Collection[float]] = None,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
        overwrite: bool = False,
    ): ...
    @overload
    def __init__(
        self,
        *,
        coded_mask_file: PathLike | str,
        teldef_file: PathLike | str,
        ra_dec_roll_inst: Optional[Collection[float]] = None,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
        overwrite: bool = False,
    ): ...
    def __init__(
        self,
        *,
        caldb_version: Optional[str] = None,
        coded_mask_file: Optional[PathLike | str] = None,
        teldef_file: Optional[PathLike | str] = None,
        ra_dec_roll_inst: Optional[Collection[float]] = None,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
        overwrite: bool = False,
    ):
        """Class for image-related analysis of BlackCAT eventlists.

        Arguments:
            - caldb_version: (Optional) BlackCAT CalDB version string.
            Mutually exclusive with coded_mask_file and teldef_file.
            - coded_mask_file: (Optional) Path to BlackCAT Aperture
            CalDB file. Mutually exclusive with caldb_version.
            - teldef_file: (Optional) Path to BlackCAT Teldef CalDB
            file. Mutually exclusive with caldb_version.
            - ra_dec_roll_inst: (Optional) Initial ra, dec, and roll
            for eventlist pointing. Defaults to [0, 0, 0] if not
            provided.
            - use_subpixel: Whether or not to use subpixels for
            imaging. Very memory heavy.
            - resolution: Imaging resolution (detector [sub]pixel size
            projected by focal length). 1=Finest (detector [sub]pixel
            size). 8 or 24 = Coarsest (mask cell size. 8 if not
            use_subpixel, 24 if use_subpixel.)
            - balance_per_det: Whether to balance each of the four
            detector DPHs individually. Uses provided array of
            bounding boxes, or generates based on CalDB if 'True'.
            - hide_frame: Sets the support structure opacity to 50% to
            compensate for edge shadow pattern.
            - overwrite: Any files generated will overwrite existing
            files with shared names, instead of raising an exception.
        """
        if ra_dec_roll_inst is None:
            ra_dec_roll_inst = [0, 0, 0]
        if len(ra_dec_roll_inst) != 3:
            raise RuntimeError(
                "ra_dec_roll_inst must have length 3 if provided. Provided "
                f"length is {len(ra_dec_roll_inst)}."
            )
        self._ra_dec_roll = list(ra_dec_roll_inst)

        self._imager = BCImager(
            caldb_version=caldb_version,
            coded_mask_file=coded_mask_file,
            teldef_file=teldef_file,
            use_subpixel=use_subpixel,
            resolution=resolution,
            balance_per_det=balance_per_det,
            hide_frame=hide_frame,
        )

        self._overwrite = overwrite

    @property
    def imager(self) -> BCImager:
        """Imager object in use by current instance of
        BCImageAnalysis.
        """
        return self._imager

    @property
    def overwrite(self) -> bool:
        """Whether current instance of BCImageAnalysis will overwrite
        files.
        """
        return self._overwrite

    @property
    def ra_dec_roll(self) -> list[float]:
        """Current RA, DEC, and roll for eventlist pointing."""
        return self._ra_dec_roll

    @overload
    def eventlist_to_image(
        self,
        *,
        counts: npt.NDArray[np.void],
        header: Optional[fits.Header] = None,
        add_wcs: bool = False,
        outfile: Optional[PathLike | str] = None,
    ) -> fits.PrimaryHDU: ...
    @overload
    def eventlist_to_image(
        self,
        *,
        fitsdata: fits.BinTableHDU | PathLike | str,
        add_wcs: bool = False,
        outfile: Optional[PathLike | str] = None,
    ) -> fits.PrimaryHDU: ...
    def eventlist_to_image(
        self,
        *,
        counts: Optional[npt.NDArray[np.void]] = None,
        header: Optional[fits.Header] = None,
        fitsdata: Optional[fits.BinTableHDU | PathLike | str] = None,
        add_wcs: bool = False,
        outfile: Optional[PathLike | str] = None,
    ) -> fits.PrimaryHDU:
        """Create a sky image from provided event data. Event
        positions should be provided in DET coordinates. Nominally,
        (DETX_axis, DETY_axis) are aligned with (-SATZ_axis,
        -SATY_axis).

        Returns a FITs HDU. Writes that HDU to a fits file if a path
        is provided.

        Arguments:
            - counts: (Optional) Structured array containing data for
            each event. Mututally exclusive with fitsdata.
            - header: (Optional) FITs header for the events provided
            by counts. Unused if fitsdata provided.
            - fitsdata: (Optional) FITs BinTableHDU or path to a fits
            file containing a BinTableHDU of events. Mutually
            exclusive with counts.
            - add_wcs: Whether to use current RA, DEC, and roll to
            add WCS keywords to the image header.
            - outfile: (Optional) Path to write the generate image to.
        """
        counts, header = self._counts_header_from_args(
            counts=counts, header=header, fitsdata=fitsdata
        )

        image_hdu = fits.PrimaryHDU(self.imager.image_counts(counts), header=header)
        image_hdu.header["BUNIT"] = ("COUNTS", "Unit of original pixel value")

        if add_wcs:
            image_hdu = self._add_wcs_keywords(image_hdu)

        if outfile is not None:
            image_hdu.writeto(outfile, overwrite=self.overwrite, checksum=True)

        return image_hdu

    def find_peaks(
        self,
        image: npt.NDArray[np.floating[Any] | np.integer[Any]],
        header: Optional[fits.Header] = None,
        min_sigma: float = 6.0,
        neighborhood_psf: int = 40,
    ) -> Any:
        """Identify peaks within the image with a significance of
        min_sigma times the local RMS or higher.

        Returns peak location (both in image coordinates and RA/DEC if
        able), peak value in counts, local and global RMS, and local
        and global significance.

        Arguments:
            - image: BlackCAT sky image array.
            - header: Image header to grab WCS keywords from if
            provided.
            - min_sigma: The minimum multiplier of the local RMS
            necessary to be identified as a peak.
            - neighborhood_psf: Sidelength of neighborhood in mask cells
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            local_rms = self.local_rms(image, neighborhood_psf)
            peaks = find_peaks(image / local_rms, threshold=min_sigma)

            if peaks is None or len(peaks) == 0:
                return np.zeros(shape=0, dtype=self.IMPEAK_DTYPE)
            else:
                locations = np.round(
                    np.array([peaks["x_peak"], peaks["y_peak"]]).T
                ).astype(int)

            npeaks = len(locations)
            result = np.zeros(shape=npeaks, dtype=self.IMPEAK_DTYPE)

            result["xy"] = locations
            result["radec"] = np.full(locations.shape, np.nan)
            if header is not None:
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(header)
                if wcs.has_celestial:
                    result["radec"] = wcs.all_pix2world(locations, 0, ra_dec_order=True)
                else:
                    warnings.warn(
                        "A header was provided, but no WCS was found. Can't "
                        "convert peak locations to RA and DEC.",
                        RuntimeWarning,
                    )
            result["counts"] = image[locations[:, 1], locations[:, 0]]
            result["local_rms"] = local_rms[locations[:, 1], locations[:, 0]]
            result["global_rms"] = image.std()
            result["local_sig"] = result["impeak"] / result["local_rms"]
            result["global_sig"] = result["impeak"] / result["global_rms"]

        return result

    def get_exposed_area_map(
        self, active_det_ids: npt.ArrayLike[np.integer[Any]]
    ) -> npt.NDArray[np.float64]:
        """Get a map of exposed focal plane area for each sky pixel,
        provided which of the four detectors are active. Values are in
        m^2.

        Arguments:
            active_det_ids: ArrayLike object of up to four unique
            detector IDs indicating that detector is to be considered
            active.
        """
        exposed_area_map = np.zeros(self.imager.image_minshape, dtype=np.float64)

        for det_id in np.unique(active_det_ids):
            exposed_area_map += self.imager.exposed_area_maps[det_id]

        return exposed_area_map

    def local_rms(
        self,
        image: npt.NDArray[np.floating[Any] | np.integer[Any]],
        neighborhood_psf: int = 40,
    ) -> npt.NDArray[np.floating[Any] | np.integer[Any]]:
        """Get the RMS in the vicinity of each point in the image.

        Specifically, in a neighborhood that is a square with each side
        length being equal to neighborhood_psf mask cell sizes.

        Arguments:
            - image: BlackCAT sky image array.
            - neighborhood_psf: Sidelength of neighborhood in mask cells
        """
        neighborhood_psf = (
            2 * (neighborhood_psf / (2 * self.imager.resolution_maskpix)).astype(int)
            + 1
        )

        # Local variance (mean square) may be slightly negative due to precision.
        local_ms = uniform_filter(image**2, size=neighborhood_psf, mode="constant")
        return np.sqrt(np.clip(local_ms, np.finfo(float).eps, None))

    @overload
    def set_ra_dec_roll(self, ra: Collection[float]) -> None: ...
    @overload
    def set_ra_dec_roll(self, ra: float, dec: float, roll: float) -> None: ...
    def set_ra_dec_roll(
        self,
        ra: float | Collection[float],
        dec: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> None:
        """Change the RA, DEC, and roll for eventlist pointing."""
        if dec is None and roll is None:
            if len(ra) == 3:
                ra, dec, roll = ra
            else:
                raise RuntimeError(
                    "RA, DEC, and roll must be specified as a triple or as 3 args."
                )

        self._ra_dec_roll = [ra, dec, roll]

    def _add_wcs_keywords(self, hdu: fits.PrimaryHDU) -> fits.PrimaryHDU:
        # Calculates and adds WCS keywords to the provided HDU

        # WCS keywords refer to the center of the pixel, but 1-idx origin
        # So for a 1x1 image, the center of the (only) pixel is (1., 1.)
        pixd1 = (hdu.data.shape[1] + 1) / 2.0
        pixd2 = (hdu.data.shape[0] + 1) / 2.0

        pixra, pixdec, pixroll = self.ra_dec_roll

        hdu.header["RA_PNT"] = (pixra, "RA of instrument pointing.")
        hdu.header["DEC_PNT"] = (pixdec, "Declination of instrument pointing.")
        hdu.header["PA_PN"] = (pixroll, "Instrument roll, + is CCW as seen from above")

        hdu.header["RADESYS"] = ("FK5", "Equatorial coordinate system")
        hdu.header["EQUINOX"] = (2000, "[yr] Equinox of equatorial coordinates")
        hdu.header["CTYPE1"] = ("RA---TAN", "Pixel coordinate system")
        hdu.header["CTYPE2"] = ("DEC--TAN", "Pixel coordinate system")
        hdu.header["CUNIT1"] = ("deg", "Units used in both CRVAl1 and CDi_j")
        hdu.header["CUNIT2"] = ("deg", "Units used in both CRVAl2 and CDi_j")

        hdu.header["CRPIX1"] = (pixd1, "Reference pixel on the horizonal axis")
        hdu.header["CRPIX2"] = (pixd2, "Reference pixel on the vertical axis")

        hdu.header["CRVAL1"] = (pixra, "WCS RA value at the reference pixel")
        hdu.header["CRVAL2"] = (pixdec, "WCS DEC value at the reference pixel")

        hdu.header["ROWORDER"] = (
            "BOTTOM-UP",
            "(0,0) is lower left; SouthEast if roll=0",
        )

        crota_radians = self.imager.instrument.teldef.rollsign * np.radians(pixroll)
        cos_r = np.cos(crota_radians)
        sin_r = np.sin(crota_radians)

        # Base scales (un-rolled: +X -> -RA, +Y -> +DEC)
        cdelt1 = -self.imager.sky_pixel_size_deg[0]
        cdelt2 = self.imager.sky_pixel_size_deg[1]

        cd1_1 = cdelt1 * cos_r
        cd1_2 = -cdelt2 * sin_r
        cd2_1 = cdelt1 * sin_r
        cd2_2 = cdelt2 * cos_r

        hdu.header["CD1_1"] = (cd1_1, "Scaling and rotation matrix element 1, 1")
        hdu.header["CD1_2"] = (cd1_2, "Scaling and rotation matrix element 1, 2")
        hdu.header["CD2_1"] = (cd2_1, "Scaling and rotation matrix element 2, 1")
        hdu.header["CD2_2"] = (cd2_2, "Scaling and rotation matrix element 2, 2")

        return hdu

    @overload
    def _counts_header_from_args(
        self,
        counts: npt.NDArray[np.void],
        header: Optional[fits.Header] = None,
        fitsdata: None = None,
    ) -> tuple[npt.NDArray[np.void], fits.Header]: ...
    @overload
    def _counts_header_from_args(
        self,
        fitsdata: fits.BinTableHDU | PathLike | str,
        counts: None = None,
        header: None = None,
    ) -> tuple[npt.NDArray[np.void], fits.Header]: ...
    def _counts_header_from_args(
        self,
        counts: Optional[npt.NDArray[np.void]] = None,
        header: Optional[fits.Header] = None,
        fitsdata: Optional[fits.BinTableHDU | PathLike | str] = None,
    ) -> tuple[npt.NDArray[np.void], fits.Header]:
        # Reads the typical arguments for provided eventlists and
        # returns the counts and header.
        if (counts is not None) == (fitsdata is not None):
            raise TypeError(
                "You must specify exactly one of a counts array or FITS data."
            )

        if counts is not None:
            if header is None:
                header = fits.Header()
        else:
            try:
                counts = fitsdata.data
                header = fitsdata.header
            except AttributeError:
                counts, header = fits.getdata(Path(fitsdata), 1, header=True)

        return counts, header
