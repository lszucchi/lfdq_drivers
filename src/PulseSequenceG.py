import numpy as np
from matplotlib.pyplot import plot, show, figure, gca
from measurements.waveforms.Pulse import Pulse

class PulseSequenceG:
    """
    This class allows one to build a synchronized sequence of pulses for different channels
    by tracking the absolute start time of each pulse from t=0.
    """
    def __init__(self, name, startup_delay):
        self.name = name
        self.channels = {}
        self.startup_delay = startup_delay

    def add(self, p, channel, delay=0.1e-9):
        """
        Adds a pulse to a channel with a specified delay preceding it.
        """
        if not isinstance(p, Pulse):
            raise TypeError('p must be a Pulse object.')

        c = channel.lower()

        # Initialize channel if it's new
        if c not in self.channels:
            self.channels[c] = {
                'pulses': [],
                'delays': [],
                'absolute_start_times': [], # We will now store absolute start times
                'current_time': 0.0 # Track the end of the last pulse in this channel
            }
        
        # The delay occurs *before* the pulse
        pulse_start_time = self.channels[c]['current_time'] + delay

        self.channels[c]['pulses'].append(p)
        self.channels[c]['delays'].append(delay)
        self.channels[c]['absolute_start_times'].append(pulse_start_time)
        
        # Update the channel's current time to be the end of the new pulse
        self.channels[c]['current_time'] = pulse_start_time + p.length

    def get_channel_duration(self, channel):
        """
        Gets the total duration of a single channel's sequence.
        """
        c = channel.lower()
        if c in self.channels:
            return self.channels[c]['current_time']
        return 0

    def get_max_duration(self):
        """
        Gets the longest duration among all channels, which defines the
        total experiment time.
        """
        max_duration = 0
        for channel in self.channels.keys():
            duration = self.get_channel_duration(channel)
            if duration > max_duration:
                max_duration = duration
        return max_duration

    def clear(self):
        self.channels = {}

    def show_all(self, timestep=0.01e-9, y_offset=2.05, ignore=[]):
        """
        Plots the sequences for all channels using absolute start times.
        """
        fig = figure()
        ax = gca()
        yorder = 0

        # Get the total duration from the new helper method
        total_duration = self.get_max_duration()
        if total_duration == 0:
            print("Sequence is empty, nothing to show.")
            return

        t = np.arange(0, total_duration, timestep)

        for c in self.channels.keys():
            if c in ignore:
                continue
            
            # Create a baseline for this channel's waveform
            sequence_wave = np.zeros(len(t)) + yorder

            # Place each pulse at its absolute start time
            for idx, p in enumerate(self.channels[c]['pulses']):
                start_time = self.channels[c]['absolute_start_times'][idx]
                start_index = int(start_time / timestep)

                # Build the individual pulse waveform
                _, p_wave = p.build(timestep)
                
                # Ensure we don't write past the end of the sequence array
                end_index = start_index + len(p_wave)
                if end_index > len(sequence_wave):
                    p_wave = p_wave[:len(sequence_wave) - start_index]
                
                # Add the pulse to the channel's waveform
                sequence_wave[start_index:start_index + len(p_wave)] += p_wave

            ax.plot(t, sequence_wave, label=f'Channel: {c}')
            yorder += y_offset
        
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (arb. units) + Offset")
        ax.set_title(f"Pulse Sequence: {self.name}")
        show()