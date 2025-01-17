import math

class TelescopeConfig:
    """
    Stores basic telescope parameters and provides
    methods to compute derived properties.
    """
    def __init__(self, focal_length_mm, aperture_mm, image_circle_mm=None):
        """
        :param focal_length_mm: Focal length of the telescope in millimeters
        :param aperture_mm: Aperture diameter in millimeters
        :param image_circle_mm: Optional image circle diameter in millimeters;
                                if None, a simple approximate formula or a 
                                user-provided value can be used.
        """
        self.focal_length_mm = focal_length_mm
        self.aperture_mm = aperture_mm
        
        # If no image circle is provided, we make a naive guess or leave it as None
        if image_circle_mm is None:
            # Simple guess: just set it to half the aperture or any formula you prefer
            self.image_circle_mm = 0.5 * self.aperture_mm
        else:
            self.image_circle_mm = image_circle_mm

    @property
    def aperture_area_cm2(self):
        """
        Compute the area of the telescope aperture in cm^2.
        """
        radius_mm = self.aperture_mm / 2.0
        area_mm2 = math.pi * (radius_mm ** 2)
        return area_mm2 / 100.0  # convert mm^2 to cm^2

    @property
    def f_ratio(self):
        """
        Compute the focal ratio (f-number).
        """
        return self.focal_length_mm / self.aperture_mm


class SensorConfig:
    """
    Stores basic sensor parameters and provides
    methods to compute derived properties.
    """
    def __init__(
        self,
        pixel_size_um,
        sensor_width_px,
        sensor_height_px,
        full_well_capacity_e,
        quantum_efficiency=0.8,
        read_noise_e=5.0,
        gain=1.0
    ):
        """
        :param pixel_size_um: Pixel size in micrometers (um)
        :param sensor_width_px: Number of pixels in the horizontal dimension
        :param sensor_height_px: Number of pixels in the vertical dimension
        :param full_well_capacity_e: Full well capacity in electrons
        :param quantum_efficiency: Quantum efficiency (0.0 to 1.0)
        :param read_noise_e: Read noise in electrons
        :param gain: System gain in e-/ADU (electrons per ADU)
        """
        self.pixel_size_um = pixel_size_um
        self.sensor_width_px = sensor_width_px
        self.sensor_height_px = sensor_height_px
        self.full_well_capacity_e = full_well_capacity_e
        self.quantum_efficiency = quantum_efficiency
        self.read_noise_e = read_noise_e
        self.gain = gain

    def compute_pixel_scale_arcsec(self, telescope_config):
        """
        Compute the pixel scale in arcseconds/pixel given a telescope configuration.
        Formula: pixel_scale_arcsec = 206265 * (pixel_size_mm / focal_length_mm)
        where pixel_size_mm = pixel_size_um / 1000.
        """
        pixel_size_mm = self.pixel_size_um / 1000.0
        return 206265.0 * (pixel_size_mm / telescope_config.focal_length_mm)