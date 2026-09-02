"""All necessary state and methods for analysis using the BlackCAT
instrument.
"""

from functools import cached_property
from os import PathLike
from typing import Any, Optional, overload

import numpy as np
import numpy.typing as npt

from bc_caldb import CodedMask, Teldef


class BCInstrument:
    """Object holding the state and methods for analyzing data from
    the BlackCAT instrument.
    """

    @overload
    def __init__(self, caldb_version: str, use_subpixel: bool = False): ...
    @overload
    def __init__(
        self,
        coded_mask_file: PathLike | str,
        teldef_file: PathLike | str,
        use_subpixel: bool = False,
    ): ...
    def __init__(
        self,
        caldb_version: Optional[str] = None,
        coded_mask_file: Optional[PathLike | str] = None,
        teldef_file: Optional[PathLike | str] = None,
        use_subpixel: bool = False,
    ):
        """BlackCAT instrument state and methods for analysis.

        Arguments:
            - caldb_version: (Optional) BlackCAT CalDB version string.
            Mutually exclusive with coded_mask_file and teldef_file.
            - coded_mask_file: (Optional) Path to BlackCAT Aperture
            CalDB file. Mutually exclusive with caldb_version.
            - teldef_file: (Optional) Path to BlackCAT Teldef CalDB
            file. Mutually exclusive with caldb_version.
            - use_subpixel: Whether or not to use subpixels for scaling
            and conversions.
        """
        if (caldb_version is None) == (coded_mask_file is None):
            raise TypeError(
                "You must specify exactly one of caldb version or coded mask caldb file."
            )

        if (caldb_version is None) == (teldef_file is None):
            raise TypeError(
                "You must specify exactly one of caldb version or teldef caldb file."
            )

        if caldb_version is None:
            self._coded_mask = CodedMask.from_caldb_file(coded_mask_file)
            self._teldef = Teldef.from_caldb_file(teldef_file)
        else:
            self._coded_mask = CodedMask.from_caldb_version(caldb_version)
            self._teldef = Teldef.from_caldb_version(caldb_version)

        self._use_subpixel = use_subpixel

    @property
    def frame_pattern(self) -> npt.NDArray[np.bool_]:
        """Pattern showing where the extra support structures are for
        the mask. Important to track since the mask's 50% open ratio
        only holds outside the frame.
        """
        return self._coded_mask.frame_pattern

    @property
    def mask_pattern(self) -> npt.NDArray[np.bool_]:
        """Coded aperture mask pattern."""
        return self._coded_mask.mask_pattern

    @property
    def teldef(self) -> Teldef:
        """BlackCAT Teldef CalDB object in use by current instance of
        BCInstrument.
        """
        return self._teldef

    @cached_property
    def detector_boxes(self) -> npt.NDArray[np.float64]:
        """[[satz_min, saty_min], [satz_max, saty_max]] bounding boxes
        for each of the four detectors.
        """
        detcorners_rawxs = np.array([0, 0, 1650, 1650], dtype=np.float64)
        detcorners_rawys = np.array([0, 1650, 0, 1650], dtype=np.float64)
        detector_boxes = []
        for det_id in self._teldef.det_ids:
            detcorners_detids = np.full(
                detcorners_rawxs.shape, det_id, dtype=np.float64
            )
            detxs, detys = self._teldef.rawxy_to_detxy(
                detcorners_rawxs, detcorners_rawys, detcorners_detids
            )
            _, satys, satzs = self._teldef.detxyz_to_satxyz(detxs, detys)
            detector_boxes.append(
                np.array(
                    [
                        [
                            np.min(satzs) - 0.5 * self._teldef.raw_xscl,
                            np.min(satys) - 0.5 * self._teldef.raw_yscl,
                        ],
                        [
                            np.max(satzs) + 0.5 * self._teldef.raw_xscl,
                            np.max(satys) + 0.5 * self._teldef.raw_xscl,
                        ],
                    ],
                    dtype=np.float64,
                )
            )

        return np.array(detector_boxes, dtype=np.float64)

    @cached_property
    def fpa_envelope(self) -> npt.NDArray[np.float64]:
        """[[satz_min, saty_min], [satz_max, saty_max]] bounding box
        for full focal plane array.
        """
        _, [saty_max, saty_min], [satz_max, satz_min] = self._teldef.detxyz_to_satxyz(
            np.array([self._teldef.detx_min, self._teldef.detx_max]),
            np.array([self._teldef.dety_min, self._teldef.dety_max]),
        )

        fpa_envelope = np.array(
            [
                [satz_min, saty_min],
                [satz_max, saty_max],
            ],
            dtype=np.float64,
        )

        return fpa_envelope

    @cached_property
    def fpa_pixel_counts(self) -> npt.NDArray[np.uint16]:
        """Nearest integer number of focal plane [sub]pixels along the
        SATZ and SATY axes, when accounting for the gaps.
        """
        satz_length, saty_length = np.diff(self.fpa_envelope, axis=0)[0]

        # Round up to make sure we catch all potential hit pixels
        fpa_pixel_counts = np.ceil(
            np.array(
                [satz_length / self.satz_pix_size, saty_length / self.saty_pix_size]
            )
        ).astype(np.uint16)

        return fpa_pixel_counts

    @cached_property
    def fpa_pix_size_array(self) -> npt.NDArray[np.float64]:
        """Focal plane array [sub]pixel sizes along the SATZ and SATY
        axes.
        """
        return np.array([self.satz_pix_size, self.saty_pix_size], dtype=np.float64)

    @cached_property
    def mask_cell_count(self) -> npt.NDArray[np.uint32]:
        """Nearest number of mask cells along the SATZ and SATY axes,
        when accounting for the support structures.
        """
        return np.array(self._coded_mask.mask_pattern.shape[::-1], dtype=np.uint32)

    @cached_property
    def mask_envelope(self) -> npt.NDArray[np.float64]:
        """[[satz_min, saty_min], [satz_max, saty_max]] bounding box
        for the mask.
        """
        mask_mins_arr = np.array(
            [self._coded_mask.crval1, self._coded_mask.crval2], dtype=np.float64
        )
        mask_maxs_arr = mask_mins_arr + self.mask_cell_size_array * self.mask_cell_count
        return np.array([mask_mins_arr, mask_maxs_arr], dtype=np.float64)

    @cached_property
    def mask_cell_size_array(self) -> npt.NDArray[np.float64]:
        """Mask cell sizes along the SATZ and SATY axes."""
        return np.array([self._coded_mask.cdelt1, self._coded_mask.cdelt2], np.float64)

    @cached_property
    def saty_pix_size(self) -> float:
        """Focal plane array [sub]pixel size along the SATY axis."""
        return (
            self._teldef.raw_yscl if self._use_subpixel else self._teldef.raw_yscl * 3
        )

    @cached_property
    def satz_pix_size(self) -> float:
        """Focal plane array [sub]pixel size along the SATZ axis."""
        return (
            self._teldef.raw_xscl if self._use_subpixel else self._teldef.raw_xscl * 3
        )

    def pixxy_to_satzy(
        self,
        pixxs: npt.NDArray[np.floating[Any]],
        pixys: npt.NDArray[np.floating[Any]],
    ) -> tuple[np.float64, np.float64]:
        """Convert from (PIXX, PIXY) to (SATZ, SATY).

        Arguments:
            - pixxs: Numpy array of how many [sub]pixels you are along
            the x axis from the center of the lower-left FPA [sub]pixel.
            - pixys: Numpy array of how many [sub]pixels you are along
            the y axis from the center of the lower-left FPA [sub]pixel.
        """
        satz_low, saty_low = self.fpa_envelope[0]
        satzs = pixxs * self.satz_pix_size + satz_low
        satys = pixys * self.saty_pix_size + saty_low
        return satzs, satys

    def satzy_to_pixxy(
        self,
        satzs: npt.NDArray[np.floating[Any]],
        satys: npt.NDArray[np.floating[Any]],
    ) -> tuple[np.float64, np.float64]:
        """Convert from (SATZ, SATY) to (PIXX, PIXY).

        PIXX and PIXY count how many [sub]pixels you are along the
        respective axes from the center of the lower-left FPA
        [sub]pixel.

        Arguments:
            - satzs: Numpy array of SATZ values.
            - satys: Numpy array of SATY values.
        """
        satz_low, saty_low = self.fpa_envelope[0]
        pixxs = ((satzs - satz_low) * (1 / self.satz_pix_size)).astype(np.float64)
        pixys = ((satys - saty_low) * (1 / self.saty_pix_size)).astype(np.float64)
        return pixxs, pixys
