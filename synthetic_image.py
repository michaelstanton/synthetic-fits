# synthetic_image.py
"""
A Python library for generating synthetic astronomical images, including:
- Star catalog retrieval (Gaia EDR3 via Astroquery)
- Building a simple synthetic WCS
- Converting RA/Dec to pixel coordinates
- Converting magnitudes to flux (photoelectrons)
- Adding point sources to an image with a PSF kernel
- Building a Gaussian PSF kernel
- Converting sky brightness (mag/arcsec^2) to electrons/pixel

Dependencies:
    pip install astroquery astropy numpy
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.io import fits


def get_star_catalog(ra_center, dec_center, radius=0.5, max_mag=16.0):
    """
    Query the Gaia EDR3 (or DR3) catalog around a given RA/Dec,
    returning star positions and magnitudes as an Astropy Table.

    Parameters
    ----------
    ra_center : float
        Right Ascension of the field center in degrees (ICRS).
    dec_center : float
        Declination of the field center in degrees (ICRS).
    radius : float
        Radius of the search region in degrees.
    max_mag : float
        Optional magnitude cut (G-band). Stars brighter than this are returned.

    Returns
    -------
    result_table : Astropy Table or None
        An Astropy Table of star data with columns [RA, Dec, Mag].
        Returns None if no stars are found.
    """

    # Set up a SkyCoord for the query center
    center_coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg, frame='icrs')

    # VizieR configuration
    # ROW_LIMIT = -1 means no row limit, so we get all possible matches (be cautious with large queries).
    Vizier.ROW_LIMIT = -1

    # Gaia EDR3 catalog identifier:
    # "I/350/gaiaedr3" or "I/355/gaiadr3" (depending on the VizieR naming)
    catalog_id = "I/350/gaiaedr3"

    # Query the region
    query_result = Vizier.query_region(
        center_coord,
        radius=radius*u.deg,
        catalog=catalog_id
    )

    if len(query_result) == 0:
        print("No results found in the given region.")
        return None

    # The query may return multiple tables; typically we take the first
    data = query_result[0]

    # Gaia EDR3 typically has columns named RA_ICRS, DE_ICRS, and Gmag (for the G-band)
    # Rename columns for convenience
    if "RA_ICRS" in data.colnames:
        data.rename_column("RA_ICRS", "RA")
    if "DE_ICRS" in data.colnames:
        data.rename_column("DE_ICRS", "Dec")
    if "Gmag" in data.colnames:
        data.rename_column("Gmag", "Mag")

    # If you only want a subset of columns
    desired_cols = ["RA", "Dec", "Mag"]
    for col in desired_cols:
        if col not in data.colnames:
            print(f"Warning: '{col}' column not found in the table.")
            return None

    # Filter by magnitude if desired
    if max_mag is not None:
        data = data[data["Mag"] <= max_mag]

    # Return only the columns we care about
    result_table = data[desired_cols]

    # Optionally, reset the table index
    result_table.index = range(len(result_table))

    return result_table


def build_synthetic_wcs(
    image_width,
    image_height,
    ra_center,
    dec_center,
    pixscale_arcsec,
    rotation_degrees=0.0
):
    """
    Build a simple WCS for a synthetic image where:
      - The image has (image_width x image_height) pixels.
      - The center is at (ra_center, dec_center) [in degrees].
      - The pixel scale is pixscale_arcsec arcseconds per pixel.
      - The image is rotated rotation_degrees degrees clockwise.
    """
    header = fits.Header()

    # Reference pixel at the center of the image
    refpix_x = image_width / 2.0
    refpix_y = image_height / 2.0

    header["NAXIS"] = 2
    header["NAXIS1"] = image_width
    header["NAXIS2"] = image_height

    # Set reference pixel
    header["CRPIX1"] = refpix_x
    header["CRPIX2"] = refpix_y

    # Set reference coordinates
    header["CRVAL1"] = ra_center
    header["CRVAL2"] = dec_center

    # Use TAN projection
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    # Convert arcseconds per pixel to degrees per pixel
    deg_per_pixel = pixscale_arcsec / 3600.0

    # If rotation_degrees > 0, rotate that many degrees clockwise
    # In standard math orientation, that's a negative angle
    import numpy as np
    theta = np.deg2rad(-rotation_degrees)  # negative for clockwise

    # Base scale matrix (no rotation):
    #   [ -deg_per_pixel    0 ]
    #   [      0      +deg_per_pixel ]
    #
    # Multiply by rotation matrix [[cosθ, -sinθ],[sinθ, cosθ]] 
    # to get the final CD matrix:
    cd1_1 = -deg_per_pixel * np.cos(theta)
    cd1_2 = -deg_per_pixel * np.sin(theta)
    cd2_1 = -deg_per_pixel * np.sin(theta)
    cd2_2 = deg_per_pixel * np.cos(theta)

    header["CD1_1"] = cd1_1
    header["CD1_2"] = cd1_2
    header["CD2_1"] = cd2_1
    header["CD2_2"] = cd2_2

    wcs_obj = WCS(header)
    return wcs_obj, header


def world_to_pixel(ra, dec, wcs_obj):
    """
    Convert RA/Dec to pixel coordinates using an Astropy WCS object.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees (ICRS).
    dec : float
        Declination in degrees (ICRS).
    wcs_obj : astropy.wcs.WCS
        A WCS object, typically derived from a FITS header.

    Returns
    -------
    (x, y) : tuple of floats
        Pixel coordinates (0-based indexing).
    """
    x, y = wcs_obj.all_world2pix(ra, dec, 0)  # 0-based origin
    return x, y


def mag_to_flux(mag, exposure_time, aperture_area, quantum_efficiency,
                mag_zero_point=20.0):
    """
    Convert magnitude to flux in photo-electrons per exposure.

    Parameters
    ----------
    mag : float
        Star or satellite magnitude (same photometric band as your zero point).
    exposure_time : float
        Exposure time in seconds.
    aperture_area : float
        Collecting area of your telescope (e.g., in cm^2).
    quantum_efficiency : float
        Effective quantum efficiency [0..1].
    mag_zero_point : float
        A system zero point that converts from magnitude to flux scaling.

    Returns
    -------
    flux_electrons : float
        The total number of electrons detected for the object during the exposure.
    """
    # The factor 10^(-0.4*(mag - mag_zero_point)) is standard for flux from magnitudes
    relative_flux = 10.0 ** (-0.4 * (mag - mag_zero_point))

    # Multiply by telescope + detector parameters
    flux_electrons = exposure_time * aperture_area * quantum_efficiency * relative_flux
    return flux_electrons


def add_psf_to_image(image, x_star, y_star, flux_electrons, psf_kernel):
    """
    Add the flux for a point source to the image, distributing it by psf_kernel.
    The psf_kernel is assumed to be normalized to 1.0 total flux.

    Parameters
    ----------
    image : 2D numpy array
        The synthetic image to be modified in place.
    x_star, y_star : float
        The position in pixel coordinates where the star is centered.
    flux_electrons : float
        Total flux (electrons) for the star/satellite.
    psf_kernel : 2D numpy array
        A normalized PSF kernel (sum(psf_kernel)=1).

    Returns
    -------
    image : 2D numpy array
        The updated image with the star added.
    """
    kernel_size_y, kernel_size_x = psf_kernel.shape

    # Coordinates of the top-left corner in the image if the star is centered
    half_x = kernel_size_x // 2
    half_y = kernel_size_y // 2

    x_min = int(np.floor(x_star) - half_x)
    x_max = x_min + kernel_size_x
    y_min = int(np.floor(y_star) - half_y)
    y_max = y_min + kernel_size_y

    # Check if the kernel goes out of image bounds
    if x_max < 0 or y_max < 0 or x_min >= image.shape[1] or y_min >= image.shape[0]:
        # The star is completely outside the image
        return image

    # Determine the overlap region
    overlap_x_min = max(x_min, 0)
    overlap_y_min = max(y_min, 0)
    overlap_x_max = min(x_max, image.shape[1])
    overlap_y_max = min(y_max, image.shape[0])

    # Image slice
    image_slice = (slice(overlap_y_min, overlap_y_max),
                   slice(overlap_x_min, overlap_x_max))

    # PSF kernel slice
    kernel_x_min = overlap_x_min - x_min
    kernel_y_min = overlap_y_min - y_min
    kernel_x_max = kernel_x_min + (overlap_x_max - overlap_x_min)
    kernel_y_max = kernel_y_min + (overlap_y_max - overlap_y_min)

    kernel_slice = (slice(kernel_y_min, kernel_y_max),
                    slice(kernel_x_min, kernel_x_max))

    # Add flux to the image in the overlapping region
    image[image_slice] += flux_electrons * psf_kernel[kernel_slice]

    return image


def make_gaussian_psf(size=21, fwhm=3.0):
    """
    Create a normalized 2D Gaussian PSF kernel
    of shape (size, size), with the given FWHM in pixels.
    """
    y, x = np.mgrid[0:size, 0:size]
    amp = 1.0
    x0 = y0 = (size - 1) / 2.0

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = Gaussian2D(amp, x0, y0, sigma, sigma)
    psf = gauss(x, y)

    # Normalize so total flux = 1.0
    psf /= np.sum(psf)
    return psf

def make_moffat_psf(size=21, fwhm=3.0, moffat_alpha=2.5):
    """
    Create a normalized 2D Moffat PSF kernel of shape (size, size), using Astropy's Moffat2D.

    In Astropy's Moffat2D:
      - gamma = core width
      - alpha = power index controlling the wing shape

    However, many references define:
      FWHM = 2 * alpha * sqrt(2^(1/beta) - 1),
    with 'beta' as the power index. Astropy calls that 'alpha'.
    
    So if you're used to 'beta' from literature, pass it as 'moffat_alpha'.
    We'll compute gamma from the desired FWHM and the chosen 'moffat_alpha'.
    """
    # Make a 2D coordinate grid
    y, x = np.mgrid[0:size, 0:size]
    x0 = y0 = (size - 1) / 2

    # In Astropy: FWHM is related to gamma and alpha by:
    #   FWHM = 2 * gamma * sqrt(2^(1/alpha) - 1)
    # so we solve for gamma:
    gamma = fwhm / (2.0 * np.sqrt(2.0**(1.0/moffat_alpha) - 1.0))

    # Create Moffat2D model:
    # amplitude=1.0, x_0=x0, y_0=y0, gamma=..., alpha=... 
    # Note 'alpha' in Moffat2D = power index
    moffat_model = Moffat2D(amplitude=1.0, x_0=x0, y_0=y0,
                            gamma=gamma, alpha=moffat_alpha)
    psf = moffat_model(x, y)

    # Normalize so total flux = 1.0
    psf /= psf.sum()
    return psf

def sky_brightness_to_electrons(
    sky_mag_per_arcsec2,
    plate_scale_arcsec_per_pix,
    exposure_time,
    quantum_efficiency,
    aperture_area,
    mag_zero_point=20.0
):
    """
    Convert a sky background in magnitudes per square arcsecond into
    photoelectrons per pixel for a given exposure, telescope area, and sensor QE.

    Parameters
    ----------
    sky_mag_per_arcsec2 : float
        Sky surface brightness in mag/arcsec^2 (e.g. ~21 for dark sky).
    plate_scale_arcsec_per_pix : float
        Arcseconds per pixel (e.g. 1.0 arcsec/pix).
    exposure_time : float
        Exposure time in seconds.
    quantum_efficiency : float
        Combined quantum efficiency of the system (0..1).
    aperture_area : float
        Effective collecting area of your telescope (e.g., in cm^2).
    mag_zero_point : float
        Zero-point magnitude used in the flux calculation.

    Returns
    -------
    sky_e_per_pixel : float
        The total electrons from sky background in one pixel over the exposure.
    """
    # 1. Relative flux for the sky brightness vs. zero point
    relative_flux_per_arcsec2 = 10.0 ** (-0.4 * (sky_mag_per_arcsec2 - mag_zero_point))

    # 2. Pixel area in arcsec^2
    pixel_area_arcsec2 = plate_scale_arcsec_per_pix**2

    # 3. Flux per pixel (relative units)
    relative_flux_per_pixel = relative_flux_per_arcsec2 * pixel_area_arcsec2

    # 4. Convert to electrons
    sky_e_per_pixel = (exposure_time
                       * aperture_area
                       * quantum_efficiency
                       * relative_flux_per_pixel)

    return sky_e_per_pixel

