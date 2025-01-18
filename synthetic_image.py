import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.modeling.models import Gaussian2D, Moffat2D
from astropy.io import fits
from skyfield.api import load, wgs84, Star
from datetime import datetime, timezone, timedelta
from astropy.table import Table

def get_star_catalog(ra_center, dec_center, radius=0.3, max_mag=16.0, local_fits_path="celestial_equator_band.fits"):
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Load the local FITS-based star catalog
    local_data = Table.read(local_fits_path)

    # Force columns 'ra' and 'dec' to be unitless, in case the FITS header says 'deg2'
    local_data['ra'].unit = None
    local_data['dec'].unit = None

    # Now multiply by degrees explicitly
    ra_values = local_data['ra'] * u.deg
    dec_values = local_data['dec'] * u.deg

    # Build sky coords for everything in local catalog
    star_coords = SkyCoord(ra=ra_values, dec=dec_values)

    # Center coordinate
    center_coord = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg)

    # Filter by radius
    sep = center_coord.separation(star_coords).deg
    mask = (sep <= radius)
    data = local_data[mask]

    # Rename columns consistently
    data.rename_column("ra", "RA")
    data.rename_column("dec", "Dec")
    data.rename_column("phot_g_mean_mag", "Mag")

    # Filter by magnitude
    if max_mag is not None:
        data = data[data["Mag"] <= max_mag]

    data.index = range(len(data))
    return data

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
    # Removed the 'NAXIS', 'NAXIS1', 'NAXIS2' entries to avoid conflicts 
    # with the primary HDU creation in generate-fits.ipynb.
    refpix_x = image_width / 2.0
    refpix_y = image_height / 2.0
    header["CRPIX1"] = refpix_x
    header["CRPIX2"] = refpix_y
    header["CRVAL1"] = ra_center
    header["CRVAL2"] = dec_center
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    deg_per_pixel = pixscale_arcsec / 3600.0
    import numpy as np
    theta = np.deg2rad(-rotation_degrees)
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
    """
    x, y = wcs_obj.all_world2pix(ra, dec, 0)
    return x, y

def mag_to_flux(mag, exposure_time, aperture_area, quantum_efficiency,
                mag_zero_point=20.0):
    """
    Convert magnitude to flux in photo-electrons per exposure.
    """
    relative_flux = 10.0 ** (-0.4 * (mag - mag_zero_point))
    flux_electrons = exposure_time * aperture_area * quantum_efficiency * relative_flux
    return flux_electrons

def add_psf_to_image(image, x_star, y_star, flux_electrons, psf_kernel):
    """
    Add the flux for a point source to the image, distributing it by psf_kernel.
    """
    kernel_size_y, kernel_size_x = psf_kernel.shape
    half_x = kernel_size_x // 2
    half_y = kernel_size_y // 2
    x_min = int(np.floor(x_star) - half_x)
    x_max = x_min + kernel_size_x
    y_min = int(np.floor(y_star) - half_y)
    y_max = y_min + kernel_size_y
    if x_max < 0 or y_max < 0 or x_min >= image.shape[1] or y_min >= image.shape[0]:
        return image
    overlap_x_min = max(x_min, 0)
    overlap_y_min = max(y_min, 0)
    overlap_x_max = min(x_max, image.shape[1])
    overlap_y_max = min(y_max, image.shape[0])
    image_slice = (slice(overlap_y_min, overlap_y_max), slice(overlap_x_min, overlap_x_max))
    kernel_x_min = overlap_x_min - x_min
    kernel_y_min = overlap_y_min - y_min
    kernel_x_max = kernel_x_min + (overlap_x_max - overlap_x_min)
    kernel_y_max = kernel_y_min + (overlap_y_max - overlap_y_min)
    kernel_slice = (slice(kernel_y_min, kernel_y_max), slice(kernel_x_min, kernel_x_max))
    image[image_slice] += flux_electrons * psf_kernel[kernel_slice]
    return image

def make_gaussian_psf(size=21, fwhm=3.0):
    """
    Create a normalized 2D Gaussian PSF kernel
    of shape (size, size), with the given FWHM in pixels.
    """
    from astropy.modeling.models import Gaussian2D
    y, x = np.mgrid[0:size, 0:size]
    amp = 1.0
    x0 = y0 = (size - 1) / 2.0
    import math
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = Gaussian2D(amp, x0, y0, sigma, sigma)
    psf = gauss(x, y)
    psf /= np.sum(psf)
    return psf

def make_moffat_psf(size=21, fwhm=3.0, moffat_alpha=2.5):
    """
    Create a normalized 2D Moffat PSF kernel of shape (size, size).
    """
    from astropy.modeling.models import Moffat2D
    y, x = np.mgrid[0:size, 0:size]
    x0 = y0 = (size - 1) / 2
    import math
    gamma = fwhm / (2.0 * math.sqrt(2.0**(1.0/moffat_alpha) - 1.0))
    moffat_model = Moffat2D(amplitude=1.0, x_0=x0, y_0=y0,
                            gamma=gamma, alpha=moffat_alpha)
    psf = moffat_model(x, y)
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
    photoelectrons per pixel for a given exposure.
    """
    relative_flux_per_arcsec2 = 10.0 ** (-0.4 * (sky_mag_per_arcsec2 - mag_zero_point))
    pixel_area_arcsec2 = plate_scale_arcsec_per_pix**2
    sky_e_per_pixel = (exposure_time
                       * aperture_area
                       * quantum_efficiency
                       * relative_flux_per_arcsec2
                      )
    return sky_e_per_pixel

def add_satellite_substepping(
    image, wcs_obj,
    start_ra_deg, start_dec_deg,
    end_ra_deg, end_dec_deg,
    total_exposure_s,
    total_satellite_flux,
    psf_kernel,
    num_steps=20
):
    """
    Simulate a satellite crossing the field by linear motion in RA/Dec.
    """
    flux_per_step = total_satellite_flux / num_steps
    ra_values = np.linspace(start_ra_deg, end_ra_deg, num_steps)
    dec_values = np.linspace(start_dec_deg, end_dec_deg, num_steps)

    for i in range(num_steps):
        ra_i = ra_values[i]
        dec_i = dec_values[i]
        x_pix, y_pix = wcs_obj.all_world2pix(ra_i, dec_i, 0)
        add_psf_to_image(image, x_pix, y_pix, flux_per_step, psf_kernel)
    return image

def add_star_trails_skyfield(
    image,
    star_catalog,
    wcs_obj,
    lat_deg,
    lon_deg,
    alt_m,
    start_time,
    end_time,
    num_sub_steps,
    aperture_area,
    quantum_efficiency,
    mag_zero_point,
    psf_kernel,
    extinction=0.0
):
    """
    Use Skyfield to compute star trails in RA/Dec, for a possibly *tracking* WCS.
    If you want a truly fixed camera, consider using alt/az below.
    """
    from skyfield.api import load, wgs84, Star
    from synthetic_image import mag_to_flux, add_psf_to_image
    planets = load('de421.bsp')
    earth = planets['earth']
    site = earth + wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_m)

    ts = load.timescale()
    t0 = ts.from_datetime(start_time)
    t1 = ts.from_datetime(end_time)
    time_array = [t0 + (t1 - t0)*frac for frac in np.linspace(0, 1, num_sub_steps)]

    total_exposure_s = (end_time - start_time).total_seconds()

    for star_row in star_catalog[0:100]:
        star_ra_deg = star_row['RA']
        star_dec_deg = star_row['Dec']
        star_mag = star_row['Mag'] + extinction
        star_obj = Star(ra_hours=(star_ra_deg/15.0), dec_degrees=star_dec_deg)
        star_flux_total = mag_to_flux(
            mag=star_mag,
            exposure_time=total_exposure_s,
            aperture_area=aperture_area,
            quantum_efficiency=quantum_efficiency,
            mag_zero_point=mag_zero_point
        )
        flux_per_step = star_flux_total / num_sub_steps

        for i, t_skyfield in enumerate(time_array):
            obs = site.at(t_skyfield)
            astrometric = obs.observe(star_obj)
            apparent = astrometric.apparent()
            ra_app, dec_app, _dist = apparent.radec(epoch=t_skyfield)
            ra_deg_app = ra_app.hours * 15.0
            dec_deg_app = dec_app.degrees
            x_pix, y_pix = wcs_obj.all_world2pix(ra_deg_app, dec_deg_app, 0)
            add_psf_to_image(image, x_pix, y_pix, flux_per_step, psf_kernel)

    return image

def add_star_trails_fixed_camera(
    image,
    star_catalog,
    lat_deg,
    lon_deg,
    alt_m,
    start_time,
    end_time,
    num_sub_steps,
    aperture_area,
    quantum_efficiency,
    mag_zero_point,
    psf_kernel,
    extinction=0.0,
    camera_rotation_deg=0.0,
    arcsec_per_pix=1.0,
    alt0=45.0,
    az0=180.0
):
    """
    Simulate star trails for a *fixed* camera that does not track the sky.
    At each sub-step, we compute the star's Alt/Az and convert it to a 
    2D 'camera' coordinate system. Then we add the corresponding flux
    to the image using psf_kernel.

    If the star is below the horizon (alt_deg < 0), we skip adding it
    to the image (though you can remove that check to visualize if you prefer).
    """
    from skyfield.api import load, wgs84, Star
    planets = load('de421.bsp')
    earth = planets['earth']
    site = earth + wgs84.latlon(lat_deg, lon_deg, elevation_m=alt_m)

    ts = load.timescale()
    t0 = ts.from_datetime(start_time)
    t1 = ts.from_datetime(end_time)
    time_array = [t0 + (t1 - t0)*frac for frac in np.linspace(0, 1, num_sub_steps)]
    total_exposure_s = (end_time - start_time).total_seconds()

    # Precompute rotation matrix for camera orientation
    import math
    theta = math.radians(-camera_rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for star_row in star_catalog:
        star_ra_deg = star_row['RA']
        star_dec_deg = star_row['Dec']
        star_mag = star_row['Mag'] + extinction
        star_obj = Star(ra_hours=(star_ra_deg / 15.0), dec_degrees=star_dec_deg)
        star_flux_total = mag_to_flux(
            mag=star_mag,
            exposure_time=total_exposure_s,
            aperture_area=aperture_area,
            quantum_efficiency=quantum_efficiency,
            mag_zero_point=mag_zero_point
        )
        flux_per_step = star_flux_total / num_sub_steps

        for t_skyfield in time_array:
            obs = site.at(t_skyfield).observe(star_obj)
            app = obs.apparent()
            alt, az, distance = app.altaz()
            alt_deg = alt.degrees
            az_deg = az.degrees

            # Skip if below horizon
            if alt_deg < 0:
                continue

            # Convert alt_deg, az_deg to offsets from alt0, az0 in arcseconds
            d_alt_arcsec = (alt_deg - alt0) * 3600.0
            d_az_arcsec  = (az_deg  - az0) * 3600.0

            # Convert to pixel space (before rotation)
            x0 = d_az_arcsec / arcsec_per_pix
            y0 = d_alt_arcsec / arcsec_per_pix

            # Apply camera rotation
            x_cam = x0 * cos_t - y0 * sin_t
            y_cam = x0 * sin_t + y0 * cos_t

            add_psf_to_image(
                image,
                x_cam + image.shape[1]/2.0,
                y_cam + image.shape[0]/2.0,
                flux_per_step,
                psf_kernel
            )

    return image