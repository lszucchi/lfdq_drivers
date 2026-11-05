from measurements.waveforms.Pulse import Pulse
import numpy as np


class GaussianCosPulse(Pulse):
    """
    This class defines a pulse to be sent to the AWG.
    """

    def __init__(
        self, amplitude, sigma, frequency, length=-1, length_factor=7, phase=0
    ):
        """Gaussian enveloped cosine wave

        Args:
            amplitude (_type_): vp
            sigma (_type_): gaussian std
            frequency (_type_): cos frequency
            length (_type_,optional): total length in seconds, if set ignores length_factor
            length_factor (_type_,optional): length in number of times the std sigma
            phase (int, optional): cosine phase. Defaults to 0.
        """
        # sigma dependant length
        if length <= 0:
            self.length_factor = length_factor
            super().__init__(length_factor * sigma)
        # length explicitely set by user size independant of sigma internally
        else:
            super().__init__(length)

        self.amplitude = amplitude
        self.sigma = sigma
        self.frequency = frequency
        self.phase = phase

    def curve(self, t):
        """Envelope in relative time

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        s = self.amplitude * np.exp(
            -(((t + (-self.length) / 2)) ** 2) / self.sigma**2 / 2
        )
        return s

    def oscillation(self, t):
        """Oscillation to be used in absolute time to maintain phase coherence

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.cos(2 * np.pi * self.frequency * (t) + self.phase)

    def build(self, timestep, initial_time=0):
        t = np.arange(initial_time, initial_time + self.length, timestep)

        return t, self.curve(t - initial_time) * self.oscillation(t)
