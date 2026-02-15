"""Primary beam correction for OVRO-LWA FITS images.

Uses the ``extractor_pb_75`` package (installed on the calim cluster) to
apply a beam model correction.  The beam model H5 path is taken from
:mod:`orca.resources.subband_config`.

Ported from the standalone ``pb_correct.py`` script into the orca package.
"""
import os
import sys
import logging
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused across all calls
_beam_model = None


def _get_beam():
    """Lazy-load the beam model singleton."""
    global _beam_model
    if _beam_model is None:
        try:
            from orca.transform.extractor_pb_75 import BeamModel, BEAM_PATH
            _beam_model = BeamModel(BEAM_PATH)
        except Exception as e:
            logger.warning(
                f"Failed to load beam model — PB correction disabled: {e}"
            )
            return None
    return _beam_model


def apply_pb_correction(fits_path: str) -> Optional[str]:
    """Apply primary beam correction and save a new ``.pbcorr.fits`` file.

    Does NOT overwrite the original.  Checks the header ``PBCORR`` keyword
    to prevent double-application.

    Args:
        fits_path: Path to the input FITS image.

    Returns:
        Path to the corrected output file, or *None* on failure / skip.
    """
    if not os.path.exists(fits_path):
        logger.error(f"File not found: {fits_path}")
        return None

    # Skip files that are already corrected (based on filename)
    if fits_path.endswith('.pbcorr.fits'):
        logger.info(f"[PB-Correct] Skipping {fits_path} (already corrected suffix).")
        return None

    try:
        # Check header first without loading data
        header_peek = fits.getheader(fits_path)
        if header_peek.get('PBCORR', False) is True:
            logger.info(f"[PB-Correct] Skipping {fits_path} (PBCORR already set).")
            return None

        beam = _get_beam()
        if beam is None:
            return None

        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

            original_shape = data.shape
            data_sq = data.squeeze()

            wcs = WCS(header).celestial
            obs_date = header.get('DATE-OBS')
            freq_hz = header.get('CRVAL3', header.get('RESTFRQ', 50e6))

            # Grid & Response
            h, w = data_sq.shape
            y_inds, x_inds = np.indices((h, w))
            ra_map, dec_map = wcs.all_pix2world(x_inds, y_inds, 0)

            resp = beam.get_response(ra_map.ravel(), dec_map.ravel(), obs_date, freq_hz)
            resp_map = resp.reshape((h, w))

            # Apply
            valid_mask = resp_map > 0.05
            corrected_data = np.zeros_like(data_sq)
            corrected_data[valid_mask] = data_sq[valid_mask] / resp_map[valid_mask]
            corrected_data[~valid_mask] = np.nan

            # Save to NEW filename
            final_data = corrected_data.reshape(original_shape)
            header['PBCORR'] = (True, 'Applied OVRO-LWA Beam')

            base, ext = os.path.splitext(fits_path)
            out_name = f"{base}.pbcorr{ext}"

            fits.writeto(out_name, final_data, header, overwrite=True)
            logger.info(f"[PB-Correct] Saved corrected image to {os.path.basename(out_name)}")
            return out_name

    except Exception as e:
        logger.error(f"[PB-Correct] Failed on {fits_path}: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m orca.transform.pb_correction <image.fits>")
        sys.exit(1)
    apply_pb_correction(sys.argv[1])
