import numpy as np
from numpy import array, zeros, int8, ravel, column_stack, bitwise_or
from PulseSequenceG import PulseSequenceG

class DataChannelManagerG:
    """
    Manages the translation of a PulseSequence object into binary data streams
    for an Arbitrary Waveform Generator (AWG). This version supports flexible,
    many-to-one mapping of logical channels to physical AWG outputs.
    """
    def __init__(self, awg):
        self.awg = awg
        self.awgChannels = {}
        self._channelData = {}

    def labelAwgChannel(self, channel, channelName, markerValue=0):
        self.awgChannels[channelName.lower()] = {'channel': channel, 'markerValue': markerValue}

    def clearAwgChannel(self):
        self.awg.freeMemory()
        self.awgChannels.clear()

    def allocAwgMemory(self):
        # Allocate memory for ONE channel stream. The AWG duplicates this for the second channel.
        self.awg.allocMemory(self._channelData['totalSizeMeasurement'])

    def _get_logical_channel_waveform(self, sequence, channel_name, awg_rate, total_duration):
        """
        Generates the waveform and marker data for a single logical channel.
        Returns silence if the channel is not in the sequence.
        """
        c = channel_name.lower()
        wave_data_size = int(total_duration * awg_rate)
        if wave_data_size % 256 != 0:
            wave_data_size += (256 - wave_data_size % 256)

        # Default to silence
        wave_data = zeros(wave_data_size, dtype=int8)
        marker_data = zeros(wave_data_size, dtype=int8)

        # If channel exists in the sequence, build its waveform
        if c in sequence.channels:
            bytes_amplitude = 127
            config = self.awgChannels.get(c, {})
            markerValue = config.get('markerValue', 0)

            for idx, pulse in enumerate(sequence.channels[c]['pulses']):
                start_time = sequence.channels[c]['absolute_start_times'][idx]
                i = int(start_time * awg_rate)
                _, p_wave = pulse.build(1 / awg_rate, 0)
                data = array(bytes_amplitude * p_wave, dtype=int8)

                if i + len(data) > len(wave_data):
                    data = data[:len(wave_data) - i]
                
                wave_data[i:i + len(data)] = data
                if pulse.marker_on:
                    marker_data[i:i + len(data)] = np.ones(len(data), dtype=int8) * markerValue
        
        return wave_data, marker_data

    def prepareChannelData(self, sequence: PulseSequenceG):
        """
        Prepares the final data streams for each physical AWG channel by summing
        all logical channels mapped to it.
        """
        awg_rate = self.awg.getSampleRate() / 2
        total_duration = sequence.get_max_duration()
        total_size_samples = int(total_duration * awg_rate)
        if total_size_samples % 256 != 0:
             total_size_samples += (256 - total_size_samples % 256)

        self._channelData = {
            'awgRate': awg_rate,
            'totalSizeMeasurement': total_size_samples * 2, # x2 for interleaved analog+marker
            'streams': {}
        }
        
        # This will hold the final combined data for each physical channel (e.g., 1 and 2)
        physical_channel_data = {}

        # For each physical channel, create empty accumulator arrays
        for phys_ch_num in [1, 2]:
            physical_channel_data[phys_ch_num] = {
                'wave': zeros(total_size_samples, dtype=np.int16), # Use 16-bit for summing to prevent overflow
                'marker': zeros(total_size_samples, dtype=int8)
            }

        # Find all logical channels that map to each physical channel and combine them
        for logical_name, config in self.awgChannels.items():
            phys_ch_num = config['channel']
            if phys_ch_num in physical_channel_data:
                # Generate this logical channel's waveform
                wave, marker = self._get_logical_channel_waveform(sequence, logical_name, awg_rate, total_duration)

                # Add this waveform to the physical channel's accumulator
                physical_channel_data[phys_ch_num]['wave'] += wave
                
                # Combine marker data with a bitwise OR
                physical_channel_data[phys_ch_num]['marker'] = bitwise_or(
                    physical_channel_data[phys_ch_num]['marker'], marker
                )

        # Now, create the final interleaved streams for loading
        for phys_ch_num, data in physical_channel_data.items():
            # Clip the summed wave data back to 8-bit range
            final_wave_clipped = np.clip(data['wave'], -127, 127).astype(np.int8)
            
            # Interleave analog and marker data
            self._channelData['streams'][phys_ch_num] = ravel(
                column_stack((final_wave_clipped, data['marker']))
            )

    def load_all_channels(self):
        """
        Loads the final, combined data streams to their respective physical AWG channels.
        """
        for phys_ch_num, stream in self._channelData['streams'].items():
            print(f"Loading stream to physical AWG channel {phys_ch_num}...")
            self.awg.loadData(stream, phys_ch_num, 0)