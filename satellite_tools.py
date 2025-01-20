import numpy as np
from datetime import datetime, timezone, timedelta
import math

from skyfield.api import load, wgs84, EarthSatellite
from skyfield.functions import to_spherical

# We reuse helper methods from synthetic_image.py
from synthetic_image import mag_to_flux, add_psf_to_image

def add_satellite_from_tle(
    image,
    wcs_obj,
    tle_line1,
    tle_line2,
    lat_deg,
    lon_deg,
    alt_m,
    start_time,
    end_time,
    psf_kernel,
    satellite_mag=7.0,
    aperture_area=326.85,
    quantum_efficiency=0.75,
    mag_zero_point=20.0,
    num_steps=20
):
    """
    Render a satellite into 'image' using a real TLE, seen from an Earth-based
    observer at (lat_deg, lon_deg, alt_m).

    We fetch each position in the Earth-centered ICRF frame, subtract them to
    get the topocentric vector from the site to the satellite, then convert
    that vector to RA/Dec.

    NOTE: We now pass a single 3-vector to `to_spherical(dx)` to avoid the
    TypeError in older Skyfield versions that do not accept three arguments.
    """
    # Create timescale
    ts = load.timescale()
    t0 = ts.from_datetime(start_time)
    t1 = ts.from_datetime(end_time)

    # Create EarthSatellite from TLE lines
    satellite = EarthSatellite(tle_line1, tle_line2, name="MySatellite", ts=ts)

    # Build an Earth reference so we can create an Earth-based site
    planets = load('de421.bsp')
    earth = planets['earth']

    site = earth + wgs84.latlon(latitude_degrees=lat_deg,
                                longitude_degrees=lon_deg,
                                elevation_m=alt_m)

    # Compute total exposure time in seconds
    total_exposure_s = (end_time - start_time).total_seconds()

    # Compute total flux for the entire exposure
    satellite_flux = mag_to_flux(
        mag=satellite_mag,
        exposure_time=total_exposure_s,
        aperture_area=aperture_area,
        quantum_efficiency=quantum_efficiency,
        mag_zero_point=mag_zero_point
    )
    flux_per_step = satellite_flux / num_steps

    for frac in np.linspace(0, 1, num_steps):
        t_now = t0 + (t1 - t0) * frac

        # 1) Satellite Earth-centered ICRF position
        satpos_icrf = satellite.at(t_now).position.au  # shape=(3,)
        # 2) Site Earth-centered ICRF position
        sitepos_icrf = site.at(t_now).position.au      # shape=(3,)

        # 3) Vector from site => satellite in ICRF
        dx = satpos_icrf - sitepos_icrf  # shape=(3,)

        # 4) Convert to spherical with one argument
        #    to_spherical(xyz) => (r, lat, lon)
        r_au, lat_rad, lon_rad = to_spherical(dx)

        # 5) lat_rad => dec, lon_rad => ra (in degrees)
        dec_deg = math.degrees(lat_rad)
        ra_deg  = math.degrees(lon_rad)
        if ra_deg < 0:
            ra_deg += 360.0

        # Convert RA/Dec to pixel coords
        x_pix, y_pix = wcs_obj.all_world2pix(ra_deg, dec_deg, 0)

        # If either is NaN, skip
        if np.isnan(x_pix) or np.isnan(y_pix):
            continue  # skip drawing

        # Add a fraction of flux
        add_psf_to_image(image, x_pix, y_pix, flux_per_step, psf_kernel)

    return image