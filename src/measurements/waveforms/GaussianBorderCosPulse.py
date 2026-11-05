from measurements.waveforms.Pulse import Pulse
import numpy as np


class GaussianBorderCosPulse(Pulse):
    """
    This class defines a pulse to be sent to the AWG.

    """

    def __init__(
        self,
        length,
        amplitude,
        frequency,
        sigma_factor=0.003,
        phase=0,
        border_lenght=3.5,
    ):
        """_summary_

        Args:
            length: Total lenght (s)
            amplitude: total max amplitude
            frequency: cos freq (s)
            sigma_factor: gaussian border std in factor of the total lenght
            phase (float, optional): cos phase (rad). Defaults to 0.
            border_lenght (int, optional): border length in factor of gaussian stds.
                Defaults to 3.
        """
        super().__init__(length)
        self.amplitude = amplitude
        self.sigma_factor = sigma_factor
        self.frequency = frequency
        self.phase = phase
        self.border_length = border_lenght

        if length < 2 * border_lenght * sigma_factor * length:
            raise ValueError("Length smaller than total border length.")

    def envelope(self, t):
        """Auxiliary method to draw envelope relative in time

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        sigma = self.sigma_factor * self.length
        border_time = self.border_length * sigma
        if self.length == 0:
            return 0

        return self.amplitude * (
            np.where(t < border_time, 1, 0)  # left  heaviside
            * np.exp(-(((t - border_time) / 2) ** 2) / sigma**2 / 2)  # left gaussian
            + np.where(
                (t < self.length - border_time) & (t > border_time), 1, 0
            )  # heaviside rectangular middle
            + np.where(t > self.length - border_time, 1, 0)  # right heaviside
            * np.exp(-(((t - self.length + border_time) / 2) ** 2) / sigma**2 / 2)
        )

    def oscillation(self, t):
        """Generates the oscillation to be used with absolute
          time to ensure phase coherence between pulses

        Args:
            t (_type_): times of each value (s)

        Returns:
            _type_: pulse heights
        """
        s = np.cos(2 * np.pi * self.frequency * (t) + self.phase)

        return s

    def build(self, timestep, initial_time=0):
        t = np.arange(initial_time, initial_time + self.length + timestep, timestep)
        return t, (self.oscillation(t) * self.envelope(t - initial_time))
