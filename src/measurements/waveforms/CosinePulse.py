from measurements.waveforms.Pulse import Pulse
import numpy as np

class CosinePulse(Pulse):

    def __init__(self, length, amplitude=1, phase=0):
        super().__init__(length)
        self.amplitude = amplitude
        self.phase = phase

    def build(self, timestep, initial_time = 0):
        t = np.arange(initial_time, initial_time + self.length, timestep)
        pulse = self.amplitude*np.ones(len(t)) * np.abs(np.cos(t-initial_time + self.phase))

        return t, pulse
