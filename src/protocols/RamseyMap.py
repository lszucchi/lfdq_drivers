"""
RamseyMap Measurement Protocol

This module implements a RamseyMap measurement using the standardized Handler framework.
It combines AWG, Alazar digitizer, RF sources, and other instruments to perform 
measurements of qubit Ramsey interference across different excitation frequencies.

A RamseyMap is a 2D sweep of delay time vs excitation frequency, showing
how the qubit dephasing behaves at different frequencies.

Author: Measurement System
Date: June 2025
"""

# Standard library imports
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Framework imports
from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

# Measurement framework imports
from measurements.DataChannelManager import DataChannelManager
from measurements.PulseSequence import PulseSequence
from measurements.waveforms.SquarePulse import SquarePulse
from measurements.waveforms.ZeroPulse import ZeroPulse
from measurements.waveforms.GaussianPulse import GaussianPulse
from measurements.waveforms.CosinePulse import CosinePulse
from measurements.waveforms.DragPulse import DragPulse

# Helper function (commonly used in the original script)
def dBmtoV(dBm):
    """Convert dBm to Volts."""
    return 10**(dBm/20) * 1e-3 * np.sqrt(50)

MEASURE_NAME = "RamseyMap"
DIR_NAME = "ramseymap"

class RamseyMapMeasurement(BaseMeasurement):
    """
    RamseyMap measurement class using the Handler framework.
    
    This class performs qubit Ramsey interference measurements across a range of
    excitation frequencies, creating a 2D map of magnitude and phase data.
    The x-axis represents delay times, while the y-axis represents
    different excitation frequencies.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    
    def __init__(self, header: str,
                 # Frequency range parameters
                 excitation_start_freq: float,
                 excitation_stop_freq: float,
                 excitation_step_freq: float,
                 # Delay range parameters
                 min_delay: float,
                 max_delay: float,
                 delay_step: float,
                 # Instrument parameters
                 attenuation: float,
                 rf_measurement_amplitude: float,
                 rf_measurement_frequency: float,
                 rf_excitation_amplitude: float,
                 # AWG parameters
                 awg_sample_rate: float = 64e9,
                 awg_ref_clock: float = 10e6,
                 # Pulse parameters
                 pi_half_pulse_length: float = 0.475e-6/2,
                 rf_measurement_length: float = 5e-6,
                 total_experiment_duration: float = 500e-6,
                 pulse_amplitude: float = 1.0,
                 pulse_channel: str = 'I',  # I or Q channel
                 # Alazar parameters
                 records_per_buffer: int = 150,
                 buffers_per_acquisition: int = 100,
                 amplitude_reference_alazar: float = -8,
                 trigger_level_volts: float = 0.7,
                 trigger_range_volts: float = 1.0,
                 waveform_head_cut: int = 1500,
                 period: int = 4,
                 buffer_timeout_ms: int = 5000,
                 decimation_value: int = 1,
                 # Handler instances (optional)
                 awg_handler: Optional[Handler] = None,
                 alazar_handler: Optional[Handler] = None,
                 rf_measurement_handler: Optional[Handler] = None,
                 rf_excitation_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None):
        
        # Generate x-axis (delay times) and y-axis (frequencies) arrays
        delays = np.arange(min_delay, max_delay, delay_step)
        frequencies = np.arange(excitation_start_freq, excitation_stop_freq, excitation_step_freq)
        
        # Store measurement parameters
        self.attenuation = attenuation
        self.rf_measurement_amplitude = rf_measurement_amplitude
        self.rf_measurement_frequency = rf_measurement_frequency
        self.rf_excitation_amplitude = rf_excitation_amplitude
        self.pi_half_pulse_length = pi_half_pulse_length
        self.rf_measurement_length = rf_measurement_length
        self.total_experiment_duration = total_experiment_duration
        self.pulse_amplitude = pulse_amplitude
        self.pulse_channel = pulse_channel
        
        # AWG parameters
        self.awg_sample_rate = awg_sample_rate
        self.awg_ref_clock = awg_ref_clock
        
        # Alazar parameters
        self.records_per_buffer = records_per_buffer
        self.buffers_per_acquisition = buffers_per_acquisition
        self.amplitude_reference_alazar = amplitude_reference_alazar
        self.trigger_level_volts = trigger_level_volts
        self.trigger_range_volts = trigger_range_volts
        self.waveform_head_cut = waveform_head_cut
        self.period = period
        self.buffer_timeout_ms = buffer_timeout_ms
        self.decimation_value = decimation_value
        
        # Additional data storage for phase information
        self.phases = np.zeros((len(frequencies), len(delays)))
        
        # Initialize handlers with default values if not provided
        handlers = {
            'awg': awg_handler or Handler(InstrumentType.AWG, "169.254.101.100"),
            'alazar': alazar_handler or Handler(InstrumentType.ALAZAR, "ALAZAR_BOARD::0"),
            'rfm': rf_measurement_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1),
            'rfe': rf_excitation_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.104::inst0::INSTR", source_number=2),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        
        # Initialize BaseMeasurement with delays as x-axis and frequencies as y-axis
        super().__init__(header, delays, frequencies, handlers)
    
    def prepare(self):
        """Prepare all instruments with configuration parameters."""
        # Access handlers
        awg = self.handlers['awg']
        alazar = self.handlers['alazar']
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        att = self.handlers['att']
        
        # Configure AWG
        self._configure_awg()
        
        # Configure RF sources
        self._configure_rf_sources()
        
        # Configure attenuator
        att.set_attenuation(self.attenuation)
        
        # Configure Alazar
        self._configure_alazar()
        
        # Initialize data channel manager
        self.data_manager = DataChannelManager(awg._driver)
        self._setup_awg_channels()
        
        # Update information string
        self.info = f"""
        RAMSEY MAP MEASUREMENT
        Delay times: {self.x[0]*1e6:.1f} to {self.x[-1]*1e6:.1f} μs ({len(self.x)} pts)
        Frequencies: {self.y[0]/1e6:.1f} to {self.y[-1]/1e6:.1f} MHz ({len(self.y)} pts)
        Attenuation: {self.attenuation} dB
        RF Measurement Frequency: {self.rf_measurement_frequency/1e9:.6f} GHz
        RF Measurement Amplitude: {self.rf_measurement_amplitude} dBm
        RF Excitation Amplitude: {self.rf_excitation_amplitude} dBm
        Pi/2 Pulse Length: {self.pi_half_pulse_length*1e9:.2f} ns
        RF Measurement Length: {self.rf_measurement_length*1e6:.2f} μs
        Pulse Channel: {self.pulse_channel}
        """
    
    def _configure_awg(self):
        """Configure AWG-specific settings via direct driver calls."""
        awg = self.handlers['awg']
        try:
            # Sample rate and clock configuration
            if hasattr(awg._driver, 'setSampleRate'):
                awg._safe_driver_call('setSampleRate', self.awg_sample_rate)
            if hasattr(awg._driver, 'setRefInClockExternal'):
                awg._safe_driver_call('setRefInClockExternal')
            if hasattr(awg._driver, 'setRefInClockFrequency'):
                awg._safe_driver_call('setRefInClockFrequency', self.awg_ref_clock)
            if hasattr(awg._driver, 'setDualWithMarker'):
                awg._safe_driver_call('setDualWithMarker')
            if hasattr(awg._driver, 'setMemoryDivision'):
                awg._safe_driver_call('setMemoryDivision', 2)
            if hasattr(awg._driver, 'setChannelMemoryToExtended'):
                awg._safe_driver_call('setChannelMemoryToExtended', 2)
            
            # Configure marker channels
            for channel in [3, 4]:  # Markers
                if hasattr(awg._driver, 'setVoltage'):
                    awg._safe_driver_call('setVoltage', channel, 1)
                if hasattr(awg._driver, 'setVoltageOffset'):
                    awg._safe_driver_call('setVoltageOffset', channel, 0.5)
            
            # Enable all channels
            for channel in range(1, 5):  # Channels 1-4
                if hasattr(awg._driver, 'enableChanneloutput'):
                    awg._safe_driver_call('enableChanneloutput', channel)
            
            # Get any errors
            if hasattr(awg._driver, 'getError'):
                awg._safe_driver_call('getError')
            
            # Stop AWG
            if hasattr(awg._driver, 'stop'):
                awg._safe_driver_call('stop')
        except Exception as e:
            print(f"Error configuring AWG: {e}")
            raise
    
    def _configure_rf_sources(self):
        """Configure RF sources with appropriate parameters."""
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        
        try:
            # Measurement source
            rfm.set_pulse_trigger_external()
            rfm.set_pulse_polarity_inverted()
            rfm.start_modulation()
            rfm.set_amplitude(self.rf_measurement_amplitude)
            rfm.set_frequency(self.rf_measurement_frequency)
            rfm.stop_rf()
            
            # Excitation source
            rfe.set_pulse_trigger_external()
            rfe.set_pulse_polarity_normal()
            rfe.start_modulation()
            rfe.set_amplitude(self.rf_excitation_amplitude)
            # Frequency will be set per point in _set_y
            rfe.stop_rf()
        except Exception as e:
            print(f"Error configuring RF sources: {e}")
            raise
    
    def _configure_alazar(self):
        """Configure Alazar digitizer with appropriate parameters."""
        alazar = self.handlers['alazar']
        
        # Calculate sampling rate
        sampling_rate = alazar.get_default_sampling_rate() / self.decimation_value
        
        # Calculate post-trigger samples
        post_trigger_samples = int(self.rf_measurement_length * sampling_rate / 256) * 256
        
        # Create alazar parameters dictionary
        self.alazar_params = {
            'preTriggerSamples': 0,
            'postTriggerSamples': post_trigger_samples,
            'recordsPerBuffer': self.records_per_buffer,
            'buffersPerAcquisition': self.buffers_per_acquisition,
            'powerReference_dBm': self.amplitude_reference_alazar,
            'inputLoad': 50,
            'triggerLevel_volts': self.trigger_level_volts,
            'triggerRange_volts': self.trigger_range_volts,
            'waveformHeadCut': self.waveform_head_cut,
            'period': self.period,
            'buffer_timeout_ms': self.buffer_timeout_ms,
            'decimation_value': self.decimation_value,
            'TTL': True,
            'save': False
        }
    
    def _setup_awg_channels(self):
        """Set up AWG channels for Ramsey sequence."""
        # Clear channels
        self.data_manager.clearAwgChannel()
        
        # Label channels
        if self.pulse_channel == 'I':
            self.data_manager.labelAwgChannel(channel=1, channelName='I', markerValue=1)
            self.data_manager.labelAwgChannel(channel=1, channelName='m', markerValue=2)
        else:  # Q channel
            self.data_manager.labelAwgChannel(channel=2, channelName='Q', markerValue=1)
            self.data_manager.labelAwgChannel(channel=1, channelName='m', markerValue=2)
    
    def _set_y(self, frequency):
        """Set excitation frequency for the RF source."""
        rfe = self.handlers['rfe']
        rfe.set_frequency(frequency)
    
    def measure_point(self, delay, frequency):
        """
        Measure a single point at the given delay and frequency.
        
        Args:
            delay (float): The delay time between pi/2 pulses
            frequency (float): The excitation frequency
            
        Returns:
            float: The magnitude measurement result
        """
        awg = self.handlers['awg']
        alazar = self.handlers['alazar']
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        
        # Stop instruments before sequence programming
        awg.stop()
        rfm.stop_rf()
        rfe.stop_rf()
        time.sleep(0.5)
        
        # Create pulse sequence
        p1 = SquarePulse(length=self.pi_half_pulse_length, amplitude=int(self.pulse_amplitude))
        p2 = ZeroPulse(length=self.rf_measurement_length)
        
        s1 = PulseSequence('RamseyMap', 10e-9)
        s1.clear()
        
        # Add first pi/2 pulse at time 0
        s1.add(p=p1, channel=self.pulse_channel, delay=0)
        # Add second pi/2 pulse after the delay
        s1.add(p=p1, channel=self.pulse_channel, delay=delay)
        # Add measurement marker pulse
        s1.add(p=p2, channel='m')
        
        # If this is the first delay, prepare channel data
        if delay == self.x[0]:
            self.data_manager.prepareChannelData(
                sequence=s1,
                totalExperimentDuration=self.total_experiment_duration
            )
            time.sleep(1)
            
            # Free memory and allocate
            if hasattr(awg._driver, 'freeMemory'):
                awg._safe_driver_call('freeMemory')
            time.sleep(0.5)
            
            self.data_manager.allocAwgMemory()
            time.sleep(0.5)
            
            # Load channel data
            self.data_manager.loadChannelDataToAwg(self.pulse_channel)
            time.sleep(0.5)
            self.data_manager.loadChannelDataToAwg('m')
            time.sleep(0.5)
            
            # Set markers
            self.data_manager.setInstrumentsMarker(1)
            time.sleep(0.5)
        else:
            # Update channel data for new delay
            self.data_manager.updateChannelData(sequence=s1, channelName=self.pulse_channel)
            time.sleep(0.5)
            
            # Load updated channel data
            self.data_manager.loadChannelDataToAwg(self.pulse_channel)
            time.sleep(0.5)
            self.data_manager.loadChannelDataToAwg('m')
            time.sleep(0.5)
            
            # Set markers
            self.data_manager.setInstrumentsMarker(1)
            time.sleep(0.5)
        
        # Start instruments
        awg.start()
        rfm.start_rf()
        rfe.start_rf()
        time.sleep(0.5)
        
        # Capture data
        i, q = alazar._driver.capture(**self.alazar_params)
        time.sleep(0.5)
        
        # Calculate magnitude and phase
        magnitude = 20 * np.log(np.sqrt(i**2 + q**2))
        phase = np.arctan2(q, i)
        
        # Store phase data in additional array
        freq_idx = np.where(self.y == frequency)[0][0]
        delay_idx = np.where(self.x == delay)[0][0]
        self.phases[freq_idx, delay_idx] = phase
        
        return magnitude
    
    def show(self):
        """Display current measurement information."""
        print(self.info)
    
    def plot(self, save=True):
        """
        Plot the RamseyMap data as a 3D heatmap.
        
        Args:
            save (bool): Whether to save the generated plot
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        # Convert delay times to microseconds for better display
        x_display = self.x * 1e6  # Convert to μs
        
        # Convert frequencies to MHz for better display
        y_display = self.y / 1e6  # Convert to MHz
        
        # Create plot with magnitude data
        fig = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name="Delay (μs)",
            yaxis_name="Frequency (MHz)",
            name_info=[f"{self.MEASURE_NAME} measurement"],
            title=f"{self.MEASURE_NAME} Measurement",
            data_x=x_display,
            data_y=y_display,
            data_z=self.data,
            zaxis_name="Magnitude (dB)"
        ).plot_3d(save=save)
        
        return fig

    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, magnitude, phase, header, and info.
        """
        measure_name = self.MEASURE_NAME
        dir_name = self.DIR_NAME
        out_dir = Path(dir_name) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{measure_name.lower()}.npz"
        path = out_dir / fname
        np.savez(path,
                 header=self.header,
                 info=self.info,
                 x=self.x,
                 y=self.y,
                 data=self.data,
                 phases=self.phases)
        print(f"Data saved to {path}")
        return path
    
    def cleanup(self):
        """Stop all instruments and clean up resources."""
        for name, handler in self.handlers.items():
            if name == 'rfm' or name == 'rfe':
                handler.stop_rf()
            elif name == 'awg':
                handler.stop()
        print(f"RamseyMap measurement completed and instruments stopped.")
    
    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot measurement data from a file.
        
        Args:
            filepath: Path to the saved data file (.npz)
            y_info: Type of data to plot ("magnitude" or "phase")
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        try:
            # Load data
            data = np.load(filepath)
            
            # Extract data using standard keys
            x = data['x']
            y = data['y']
            
            # Determine which data to plot based on y_info
            if y_info.lower() == "phase" and 'phases' in data:
                measurement_data = data['phases']
                z_axis_name = "Phase (rad)"
            else:
                measurement_data = data['data']
                z_axis_name = "Magnitude (dB)"
            
            # Get info
            info = str(data.get('info', "No info available"))
            
            # Convert delay times to microseconds for better display
            x_display = x * 1e6  # Convert to μs
            
            # Convert frequencies to MHz for better display
            y_display = y / 1e6  # Convert to MHz
            
            # Create 3D plot
            fig = Plot(
                measure_name=cls.MEASURE_NAME,
                measurement_info=info,
                dir_name=cls.DIR_NAME,
                xaxis_name="Delay (μs)",
                yaxis_name="Frequency (MHz)",
                name_info=[f"{cls.MEASURE_NAME} measurement"],
                title=f"{cls.MEASURE_NAME} Measurement ({y_info})",
                data_x=x_display,
                data_y=y_display,
                data_z=measurement_data,
                zaxis_name=z_axis_name
            ).plot_3d(save=save)
            
            return fig
            
        except Exception as e:
            print(f"Error plotting from file: {e}")
            raise


















