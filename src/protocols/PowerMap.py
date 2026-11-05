from protocols.PunchOut import PunchOutMeasurement
from protocols.PowerSweep import PowerSweepMeasurement
import numpy as np

class PowerMapMeasurement:
    def __init__(self, attenuation, *args, **kwargs):
        self.attenuation = attenuation

        if isinstance(attenuation, (int, float)):
            print("[PowerMap] Initializing PunchOutMeasurement instance")
            self._measurement_instance = PunchOutMeasurement(attenuation=attenuation, *args, **kwargs)

        elif isinstance(attenuation, (list, tuple, np.ndarray)):
            print("[PowerMap] Initializing PowerSweepMeasurement instance")
            self._measurement_instance = PowerSweepMeasurement(attenuation_range=attenuation, *args, **kwargs)

        else:
            raise ValueError("[PowerMap] Invalid attenuation type. Must be int/float or list/tuple.")
        

    def __getattr__(self, name):
        """
        Dynamic forwarding of all attribute/method calls to the selected measurement instance.
        This avoids having to manually wrap every method (measure, plot, save...).
        """
        return getattr(self._measurement_instance, name)
