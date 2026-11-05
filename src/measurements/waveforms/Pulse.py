from abc import ABC # this defines the Abstract Base Class


class Pulse(ABC):
    '''
        This class defines an absctract class that is the bases for all waveforms sent do AWG.
        amplitude = ranges from 0 to 1
    '''

    def __init__(self, length):
        self.length = length
        self.marker_on = False  # This is used to define if the marker is on or off during the pulse.

    def build(self, timestep, initial_time = 0):
        '''
        The build method receives the timestep used to discretize pulse.
        initial_time is used to define the initial time of the pulse, to correct the phase.
        '''
        pass
    