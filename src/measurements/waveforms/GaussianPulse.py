from measurements.waveforms.Pulse import Pulse
import numpy as np


class GaussianPulse(Pulse):
    def __init__(self, amplitude, frequency, phase, lengthP1, delay, lengthP2):
        super().__init__(lengthP1+delay+lengthP2)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.lengthP1 = lengthP1
        self.delay = delay
        self.lengthP2 = lengthP2
        self.marker_on = False

    def build(self, timestep, initial_time = 0):
        total_time = self.lengthP1 + self.delay + self.lengthP2
        
        t = np.arange(initial_time,
                      initial_time + total_time,
                      timestep)

        # --- Coseno contínuo no tempo ---
        carrier = self.amplitude * np.sin(2*np.pi*self.frequency*t + self.phase)

        # --- Envelope: dois pulsos separados pelo delay ---
        envelope = np.zeros_like(t)

        # Primeiro pulso
        idx1 = (t >= initial_time) & (t < initial_time + self.lengthP1)
        envelope[idx1] = 1.0

        # Segundo pulso
        idx2 = (t >= initial_time + self.lengthP1 + self.delay) & \
               (t <  initial_time + self.lengthP1 + self.delay + self.lengthP2)
        envelope[idx2] = 1.0

        # Pulso final com o mesmo coseno mas janelado
        final_pulse = carrier * envelope

        return t, final_pulse

    """"
    This class defines a pulse to be sent to the AWG.
    """
    """
    def __init__(self, length, amplitude, sigma, phase=0):
        super().__init__(length)
        self.amplitude = amplitude
        self.sigma = sigma
        self.phase = phase

    def build(self, timestep, initial_time=0):
        t = np.arange(initial_time, initial_time + self.length, timestep)

        s = self.amplitude * np.exp(
            -(((t - initial_time - self.length) / 2) ** 2) / self.sigma**2 / 2
        )

        return t, s
    """