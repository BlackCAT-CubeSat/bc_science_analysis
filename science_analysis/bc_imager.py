"""Produces sky images when provided appropriate BlackCAT CalDB and
events.
"""

from functools import cached_property
from os import PathLike
from typing import Any, Optional, overload

import numpy as np
import numpy.typing as npt

from science_analysis.bc_instrument import BCInstrument


class BCImager:
    """FFT image algorithm for BlackCAT coded aperture imaging."""

    @overload
    def __init__(
        self,
        *,
        caldb_version: str,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
    ): ...
    @overload
    def __init__(
        self,
        *,
        coded_mask_file: PathLike | str,
        teldef_file: PathLike | str,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
    ): ...
    def __init__(
        self,
        *,
        caldb_version: Optional[str] = None,
        coded_mask_file: Optional[PathLike | str] = None,
        teldef_file: Optional[PathLike | str] = None,
        use_subpixel: bool = False,
        resolution: int = 1,
        balance_per_det: bool | npt.NDArray[np.floating[Any] | np.integer[Any]] = True,
        hide_frame: bool = True,
    ):
        """BlackCAT coded mask image reconstruction using FFT correlation.

        Arguments:
            - caldb_version: (Optional) BlackCAT CalDB version string.
            Mutually exclusive with coded_mask_file and teldef_file.
            - coded_mask_file: (Optional) Path to BlackCAT Aperture
            CalDB file. Mutually exclusive with caldb_version.
            - teldef_file: (Optional) Path to BlackCAT Teldef CalDB
            file. Mutually exclusive with caldb_version.
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
        """
        self._instrument = BCInstrument(
            caldb_version, coded_mask_file, teldef_file, use_subpixel
        )

        self._resolution_detpix = (resolution * np.array([1, 1])).astype(np.int32)
        self._balance_per_det = balance_per_det
        self._hide_frame = hide_frame

    @property
    def instrument(self) -> BCInstrument:
        """BlackCAT instrument object in use by current instance of
        BCImager.
        """
        return self._instrument

    @cached_property
    def balance_boxes(self) -> bool | npt.NDArray[np.float64]:
        """Bounding boxes to be used for balancing each detector plane
        histogram individually.

        Takes the form [[[detxlow0, detylow0], [detxhigh0, detyhigh0]],
        ..., [[detxlown, detylown], [detxhighn, detyhighn]]]
        """
        balance = (
            self._instrument.detector_boxes
            if self._balance_per_det
            else self._balance_per_det
        )
        return balance

    @cached_property
    def dph_minshape(self) -> npt.NDArray[np.int32]:
        """Shape that the full focal plane detector plane histogram
        will take.
        """
        dph_minsize = (
            self._instrument.fpa_pixel_counts / self._resolution_detpix
        ).astype(np.int32)
        # ::-1 since minsize is (x, y) and shape is (y, x)
        return dph_minsize[::-1]

    @cached_property
    def exposed_area_maps(self) -> dict[int, npt.NDArray[np.float64]]:
        """Dictionary of exposed area maps for how much area each sky
        pixel can see on each of the four detectors. Values are in m^2.
        """
        sky_yidc, sky_xidc = np.indices(self.image_minshape, dtype=np.float64)
        # Image is viewed from FPA through the mask, so it is inverted
        # along the x-axis compared to the others
        sky_xoffs_pix = (self.image_minshape[1] - 1) / 2 - sky_xidc
        sky_yoffs_pix = sky_yidc - (self.image_minshape[0] - 1) / 2

        focallen = self._instrument.teldef.focallen
        mask_xoffs_m = sky_xoffs_pix * self.sky_pixel_size_rad[0] * focallen
        mask_yoffs_m = sky_yoffs_pix * self.sky_pixel_size_rad[1] * focallen

        [[maskx_min, masky_min], [maskx_max, masky_max]] = (
            self._instrument.mask_envelope
        )
        min_detxs_hit = maskx_min - mask_xoffs_m
        max_detxs_hit = maskx_max - mask_xoffs_m
        min_detys_hit = masky_min - mask_yoffs_m
        max_detys_hit = masky_max - mask_yoffs_m

        det_exposure_dict = {}
        for det_id in self._instrument.teldef.det_ids:
            [[detx_min, dety_min], [detx_max, dety_max]] = (
                self._instrument.detector_boxes[det_id]
            )

            detx_min_array = min_detxs_hit.copy()
            detx_min_array[detx_min_array <= detx_min] = detx_min
            detx_max_array = max_detxs_hit.copy()
            detx_max_array[detx_max_array >= detx_max] = detx_max
            detx_range_array = detx_max_array - detx_min_array
            detx_range_array[detx_range_array <= 0] = 0

            dety_min_array = min_detys_hit.copy()
            dety_min_array[dety_min_array <= dety_min] = dety_min
            dety_max_array = max_detys_hit.copy()
            dety_max_array[dety_max_array >= dety_max] = dety_max
            dety_range_array = dety_max_array - dety_min_array
            dety_range_array[dety_range_array <= 0] = 0

            det_exposure_dict[det_id] = detx_range_array * dety_range_array

        return det_exposure_dict

    @cached_property
    def image_fftshape(self) -> npt.NDArray[np.int32]:
        """Shape the image fft arrays will take."""
        return 2 ** np.ceil(np.log2(self.image_minshape)).astype(np.int32)

    @cached_property
    def image_minshape(self) -> npt.NDArray[np.int32]:
        """Shape the final sky image will take."""
        return (self.dph_minshape + self.mask_minshape).astype(np.int32)

    @cached_property
    def mask_for_correlate(self) -> npt.NDArray[np.complex64]:
        """Complex-valued array generated from the mask pattern to
        correlate with the detector plane histogram.
        """
        expansion = np.round(1 / self.resolution_maskpix).astype(np.uint64)
        if not np.allclose(1 / expansion, self.resolution_maskpix):
            raise NotImplementedError("Can only scale mask by integer currently.")

        pattern = self._instrument.mask_pattern.astype(np.float64)

        if self._hide_frame:
            frame_block = self._instrument.frame_pattern
            offset = np.mean(pattern[~frame_block])
            pattern -= offset
            pattern[frame_block] = 0.0

        indices = [
            np.arange(0, length * scale, dtype=np.uint64) // scale
            for length, scale in zip(pattern.shape, expansion)
        ]

        pattern = pattern[indices[0], :][:, indices[1]]
        pattern = pattern - pattern.mean()
        pattern = pattern / pattern.std()

        offset = self.dph_minshape
        mask_expanded = np.zeros(self.image_fftshape, dtype=np.float64)
        mask_expanded[
            offset[0] : offset[0] + pattern.shape[0],
            offset[1] : offset[1] + pattern.shape[1],
        ] = pattern

        return self.fft_forward(mask_expanded).conjugate()

    @cached_property
    def mask_minshape(self) -> npt.NDArray[np.int32]:
        """Shape the scaled mask pattern array will take. Depends on
        resolution.
        """
        mask_minsize = (
            self._instrument.mask_cell_count / self.resolution_maskpix
        ).astype(np.int32)
        # ::-1 since minsize is (x, y) and shape is (y, x)
        return mask_minsize[::-1]

    @cached_property
    def resolution_maskpix(self) -> npt.NDArray[np.float64]:
        """Resolution array used for expanding the mask pattern."""
        return (
            self._resolution_detpix
            * self._instrument.fpa_pix_size_array
            / self._instrument.mask_cell_size_array
        )

    @cached_property
    def sky_pixel_size_deg(self) -> npt.NDArray[np.float64]:
        """Tangent plane projection sky pixel size in degress."""
        return np.rad2deg(self.sky_pixel_size_rad)

    @cached_property
    def sky_pixel_size_rad(self) -> npt.NDArray[np.float64]:
        """Tangent plane projection sky pixel size in radians."""
        sky_pix_size_rad = np.arctan(
            self._instrument.fpa_pix_size_array / self._instrument.teldef.focallen
        ).astype(np.float64)
        return sky_pix_size_rad

    def _counts_to_dph(self, counts: npt.NDArray[np.void]) -> npt.NDArray[np.float32]:
        # Convert eventlist counts to a detector plane histogram.

        pixxs, pixys = self.instrument.detxy_to_pixxy(counts["DETX"], counts["DETY"])

        i, j = [
            (d // scale).astype(int)
            for d, scale in zip((pixxs, pixys), self._resolution_detpix)
        ]

        if (
            i.min() < 0
            or i.max() >= self.dph_minshape[1]
            or j.min() < 0
            or j.max() >= self.dph_minshape[0]
        ):
            raise RuntimeError("Detector count outside of detector envelope.")

        dph = (
            np.bincount(
                j * self.dph_minshape[1] + i, minlength=np.prod(self.dph_minshape)
            )
            .reshape(self.dph_minshape)
            .astype(np.float32)
        )

        if self.balance_boxes is not False:
            for [[min_detx, min_dety], [max_detx, max_dety]] in self.balance_boxes:
                [min_pixx, max_pixx], [min_pixy, max_pixy] = (
                    self.instrument.detxy_to_pixxy(
                        np.array([min_detx, max_detx], dtype=np.float64),
                        np.array([min_dety, max_dety], dtype=np.float64),
                    )
                    / self._resolution_detpix[:, np.newaxis]
                )

                [min_pixx, min_pixy] = np.clip(
                    np.ceil([min_pixx, min_pixy]), a_min=0, a_max=None
                ).astype(np.int32)
                [max_pixx, max_pixy] = np.clip(
                    np.floor([max_pixx, max_pixy]), a_min=0, a_max=None
                ).astype(np.int32)

                dph[min_pixx:max_pixx, min_pixy:max_pixy] -= dph[
                    min_pixx:max_pixx, min_pixy:max_pixy
                ].mean()

        return dph

    def _dph_to_image(self, dph: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Convert detector plane histogram to a projected sky image.

        dph_expanded = np.zeros(self.image_fftshape, dtype=np.float32)
        # Need to flip dphx when expanding to match desired alignment
        dph_expanded[: dph.shape[0], : dph.shape[1]] = dph
        dph_fft = self.fft_forward(dph_expanded)
        corr_fft = dph_fft * self.mask_for_correlate
        image = self.fft_inverse(corr_fft)[
            -self.image_minshape[0]:, -self.image_minshape[1]:
        ]
        # TODO: Update once we've got SAT coordinates defined in caldb
        # Flip x-axis to properly rotate around to be looking through the mask
        return image[:, ::-1]

    def image_counts(self, counts: npt.NDArray[np.void]) -> npt.NDArray[np.float32]:
        """Generate sky image from eventlists counts.

        Arguments:
            - counts: Structured array containing data for each event.
        """
        dph = self._counts_to_dph(counts)
        return self._dph_to_image(dph)

    @staticmethod
    def fft_forward(
        reals: npt.NDArray[np.floating[Any] | np.integer[Any]],
    ) -> npt.NDArray[np.complex64]:
        """Encapsulates the numpy fft.rfft2 function."""
        return np.fft.rfft2(reals).astype(np.complex64)

    @staticmethod
    def fft_inverse(complexes: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
        """Encapsulates the numpy fft.irfft2 function."""
        return np.fft.irfft2(complexes).astype(np.float64)
