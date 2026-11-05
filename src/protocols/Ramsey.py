"""
Ramsey Measurement Protocol

This module implements Ramsey interference measurements using the standardized Handler framework.
It combines AWG, Alazar digitizer, RF sources, and other instruments to perform time-domain
measurements of qubit dephasing by applying two π/2 pulses separated by a variable delay time.

The Ramsey measurement characterizes the phase coherence time (T2*) of a qubit.

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

MEASURE_NAME = "Ramsey"
DIR_NAME = "ramsey"

class RamseyMeasurement(BaseMeasurement):
    """
    Ramsey measurement class using the Handler framework.
    
    This class performs qubit Ramsey interference measurements by applying two π/2 pulses
    separated by a variable delay time. The measurement produces a decay curve that can be
    fitted to extract the T2* dephasing time.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    
    def __init__(self, header: str,
                 # Delay range parameters
                 min_delay: float,
                 max_delay: float,
                 delay_step: float,
                 # Instrument parameters
                 attenuation: float,
                 rf_measurement_amplitude: float,
                 rf_measurement_frequency: float,
                 rf_excitation_amplitude: float,
                 rf_excitation_frequency: float,
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
                 records_per_buffer: int = 200,
                 buffers_per_acquisition: int = 150,
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
        
        # Generate x-axis (delay times) array
        delays = np.arange(min_delay, max_delay, delay_step)
        
        # We'll use a 1-element y-axis for compatibility with BaseMeasurement
        # This will effectively create a 1D measurement (delay vs. magnitude)
        y_dummy = np.array([attenuation])  # Single element array for y-axis
        
        # Store measurement parameters
        self.attenuation = attenuation
        self.rf_measurement_amplitude = rf_measurement_amplitude
        self.rf_measurement_frequency = rf_measurement_frequency
        self.rf_excitation_amplitude = rf_excitation_amplitude
        self.rf_excitation_frequency = rf_excitation_frequency
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
        self.phases = np.zeros(len(delays))
        
        # Initialize handlers with default values if not provided
        handlers = {
            'awg': awg_handler or Handler(InstrumentType.AWG, "169.254.101.100"),
            'alazar': alazar_handler or Handler(InstrumentType.ALAZAR, "ALAZAR_BOARD::0"),
            'rfm': rf_measurement_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1),
            'rfe': rf_excitation_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.104::inst0::INSTR", source_number=2),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        
        # Initialize BaseMeasurement with delays as x-axis and dummy y-axis
        super().__init__(header, delays, y_dummy, handlers)
    
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
        RAMSEY MEASUREMENT
        Delay times: {self.x[0]*1e9:.1f} to {self.x[-1]*1e9:.1f} ns ({len(self.x)} pts)
        Attenuation: {self.attenuation} dB
        RF Measurement Frequency: {self.rf_measurement_frequency/1e9:.6f} GHz
        RF Measurement Amplitude: {self.rf_measurement_amplitude} dBm
        RF Excitation Frequency: {self.rf_excitation_frequency/1e9:.6f} GHz
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
                    awg._safe_driver_call('setVoltage', channel, 1.0)
                if hasattr(awg._driver, 'setVoltageOffset'):
                    awg._safe_driver_call('setVoltageOffset', channel, 0.5)
            
            # Enable all channels
            for channel in [1, 2, 3, 4]:
                if hasattr(awg._driver, 'enableChanneloutput'):
                    awg._safe_driver_call('enableChanneloutput', channel)
            
            # Stop AWG initially
            if hasattr(awg._driver, 'stop'):
                awg._safe_driver_call('stop')
                
        except Exception as e:
            print(f"Warning: AWG configuration failed: {e}")
    
    def _configure_rf_sources(self):
        """Configure RF source parameters."""
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        
        # Configure measurement RF source
        if hasattr(rfm._driver, 'set_pulse_trigger_external'):
            rfm._safe_driver_call('set_pulse_trigger_external')
        if hasattr(rfm._driver, 'setPulsePolarityInverted'):
            rfm._safe_driver_call('setPulsePolarityInverted')
        if hasattr(rfm._driver, 'start_mod'):
            rfm._safe_driver_call('start_mod')
        if hasattr(rfm._driver, 'set_amplitude'):
            rfm._safe_driver_call('set_amplitude', self.rf_measurement_amplitude)
        if hasattr(rfm._driver, 'set_frequency'):
            rfm._safe_driver_call('set_frequency', self.rf_measurement_frequency)
        
        # Configure excitation RF source
        if hasattr(rfe._driver, 'set_pulse_trigger_external'):
            rfe._safe_driver_call('set_pulse_trigger_external')
        if hasattr(rfe._driver, 'setPulsePolarityNormal'):
            rfe._safe_driver_call('setPulsePolarityNormal')
        if hasattr(rfe._driver, 'start_mod'):
            rfe._safe_driver_call('start_mod')
        if hasattr(rfe._driver, 'set_amplitude'):
            rfe._safe_driver_call('set_amplitude', self.rf_excitation_amplitude)
        if hasattr(rfe._driver, 'set_frequency'):
            rfe._safe_driver_call('set_frequency', self.rf_excitation_frequency)
        
        # Stop RF initially
        if hasattr(rfm._driver, 'stop_rf'):
            rfm._safe_driver_call('stop_rf')
        if hasattr(rfe._driver, 'stop_rf'):
            rfe._safe_driver_call('stop_rf')
    
    def _configure_alazar(self):
        """Configure Alazar data acquisition card."""
        alazar = self.handlers['alazar']
        
        # If the Handler has its own configure_alazar method, use it
        if hasattr(alazar, 'configure_alazar'):
            alazar.configure_alazar(
                records_per_buffer=self.records_per_buffer,
                buffers_per_acquisition=self.buffers_per_acquisition,
                amplitude_reference_alazar=self.amplitude_reference_alazar,
                trigger_level_volts=self.trigger_level_volts,
                trigger_range_volts=self.trigger_range_volts,
                waveform_head_cut=self.waveform_head_cut,
                period=self.period,
                buffer_timeout_ms=self.buffer_timeout_ms,
                decimation_value=self.decimation_value,
                rf_measurement_length=self.rf_measurement_length
            )
        # Otherwise, try to configure via the driver directly
        elif hasattr(alazar._driver, 'configure') and callable(alazar._driver.configure):
            alazar_params = {
                'recordsPerBuffer': self.records_per_buffer,
                'buffersPerAcquisition': self.buffers_per_acquisition,
                'amplitudeReferenceAlazar': self.amplitude_reference_alazar,
                'inputLoad': 50,  # Default impedance
                'triggerLevel_volts': self.trigger_level_volts,
                'triggerRange_volts': self.trigger_range_volts,
                'waveformHeadCut': self.waveform_head_cut,
                'period': self.period,
                'buffer_timeout_ms': self.buffer_timeout_ms,
                'decimationValue': self.decimation_value,
                'TTL': True,
                'save': False,
                'DefaultAlazarSamplingRate': 1e9
            }
            
            # Calculate post-trigger samples based on RF measurement length
            sampling_rate = alazar_params['DefaultAlazarSamplingRate'] / alazar_params['decimationValue']
            post_trigger_samples = int(self.rf_measurement_length * sampling_rate / 256) * 256
            alazar_params['postTriggerSamples'] = post_trigger_samples
            alazar_params['preTriggerSamples'] = 0
            
            # Apply configuration
            alazar._driver.configure(alazar_params)
    
    def _setup_awg_channels(self):
        """Set up the AWG channels for the pulse sequence."""
        # Clear existing channel configuration
        self.data_manager.clearAwgChannel()
        
        # Configure AWG channels based on selected pulse channel
        self.data_manager.labelAwgChannel(channel=2, channelName='Q', markerValue=1)
        self.data_manager.labelAwgChannel(channel=1, channelName='I', markerValue=1)
        self.data_manager.labelAwgChannel(channel=1, channelName='m', markerValue=2)
        
        # Stop AWG
        self.handlers['awg']._driver.stop()
    
    def _setup_initial_sequence(self):
        """Set up the initial pulse sequence for the first measurement."""
        # Define the pi/2 pulse and measurement pulse
        p1 = SquarePulse(length=self.pi_half_pulse_length, amplitude=int(self.pulse_amplitude))
        p2 = ZeroPulse(length=self.rf_measurement_length)
        
        # Create a pulse sequence
        seq = PulseSequence('Ramsey', 10e-9)
        seq.clear()
        
        # Add pulses to the sequence - two π/2 pulses separated by delay
        seq.add(p=p1, channel=self.pulse_channel, delay=0)
        seq.add(p=p1, channel=self.pulse_channel, delay=self.rf_measurement_length)
        seq.add(p=p2, channel='m')
        
        # Prepare and load the sequence to AWG
        self.data_manager.prepareChannelData(sequence=seq, totalExperimentDuration=self.total_experiment_duration)
        time.sleep(1)
        
        # Free AWG memory
        self.handlers['awg']._driver.freeMemory()
        time.sleep(0.5)
        
        # Allocate AWG memory
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
        
        return seq, p1, p2
    
    def measure_point(self, delay: float, y_val=None):
        """Measure a point with the given delay between the two π/2 pulses."""
        # Update the pulse sequence with the current delay
        p1 = SquarePulse(length=self.pi_half_pulse_length, amplitude=int(self.pulse_amplitude))
        p2 = ZeroPulse(length=self.rf_measurement_length)
        
        seq = PulseSequence('Ramsey', 10e-9)
        seq.clear()
        
        # Add pulses to the sequence - two π/2 pulses separated by the current delay
        seq.add(p=p1, channel=self.pulse_channel, delay=0)
        seq.add(p=p1, channel=self.pulse_channel, delay=self.rf_measurement_length + delay)
        seq.add(p=p2, channel='m')
        
        # Update AWG sequence
        self.data_manager.updateChannelData(sequence=seq, channelName=self.pulse_channel)
        time.sleep(0.5)
        self.data_manager.loadChannelDataToAwg(self.pulse_channel)
        time.sleep(0.5)
        self.data_manager.loadChannelDataToAwg('m')
        time.sleep(0.5)
        self.data_manager.setInstrumentsMarker(1)
        time.sleep(0.5)
        
        # Capture data
        I, Q = self.handlers['alazar']._driver.capture(
            preTriggerSamples=0,
            postTriggerSamples=int(self.rf_measurement_length * 1e9 / 256) * 256,
            recordsPerBuffer=self.records_per_buffer,
            buffersPerAcquisition=self.buffers_per_acquisition,
            powerReference_dBm=self.amplitude_reference_alazar,
            inputLoad=50,
            triggerLevel_volts=self.trigger_level_volts,
            triggerRange_volts=self.trigger_range_volts,
            waveformHeadCut=self.waveform_head_cut,
            period=self.period,
            buffer_timeout_ms=self.buffer_timeout_ms,
            decimation_value=self.decimation_value,
            TTL=True,
            save=False
        )
        
        # Calculate magnitude and store phase for this point
        magnitude = np.sqrt(I**2 + Q**2)
        
        # Find the index of current delay (using nearest-value approach for floating point safety)
        delay_idx = np.abs(self.x - delay).argmin()
        
        # Store phase information
        self.phases[delay_idx] = np.arctan2(Q, I)
        
        # Return the magnitude in dB
        return 20 * np.log10(magnitude)

    def run(self):
        """
        Run the Ramsey measurement, overriding the base implementation to 
        handle AWG and RF source initialization.
        """
        # Display measurement info
        self.show()
        
        # Set up AWG sequence
        seq, p1, p2 = self._setup_initial_sequence()
        
        # Start instruments
        self.handlers['awg']._driver.start()
        self.handlers['rfm']._driver.start_rf()
        self.handlers['rfe']._driver.start_rf()
        time.sleep(1)
        
        # Run the measurement
        total = len(self.x)
        count = 0
        
        for i, delay in enumerate(self.x):
            result = self.measure_point(delay)
            self.data[0, i] = result  # Store in first row (y=0)
            count += 1
            self._show_progress(count, total)
            
            # Plot intermediate results
            plt.figure(figsize=(10, 6))
            plt.plot(self.x[:i+1] * 1e6, dBmtoV(self.data[0, :i+1]), 'o-')
            plt.xlabel('Delay (μs)')
            plt.ylabel('Amplitude (V)')
            plt.title('Ramsey Measurement')
            plt.pause(0.05)
            plt.close()
        
        # Save and plot final results
        self.save_data()
        self.show()
        self.plot(save=True)
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Perform cleanup of all instruments after measurement."""
        try:
            # Stop RF sources
            rfm = self.handlers['rfm']
            rfe = self.handlers['rfe']
            if hasattr(rfm._driver, 'stop_rf'):
                rfm._safe_driver_call('stop_rf')
            if hasattr(rfe._driver, 'stop_rf'):
                rfe._safe_driver_call('stop_rf')
            
            # Stop AWG
            awg = self.handlers['awg']
            if hasattr(awg._driver, 'stop'):
                awg._safe_driver_call('stop')
            
            print("Measurement resources cleaned up.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def plot(self, save=True):
        """
        Plot the Ramsey measurement result as a 2D line plot.
        Overrides the base method to ensure proper display of Ramsey data.
        """
        # Extract the data from the first row (y=0)
        y_data = self.data[0, :]
        
        fig = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name="Delay (s)",
            yaxis_name="Magnitude (dB)",
            name_info=["Ramsey Measurement"],
            title=f"{self.MEASURE_NAME}",
            data_x=self.x,
            data_y=y_data
        ).plot(save=save)
        
        fig.show()
        return fig
    
    def save_data(self):
        """
        Save raw data to numpy .npz with x, data, phases, header, and info.
        """
        out_dir = Path(self.DIR_NAME) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        fname = f"{ts}_{self.MEASURE_NAME.lower()}.npz"
        path = out_dir / fname
        
        np.savez(path,
                header=self.header,
                info=self.info,
                x=self.x,
                y=self.y,
                data=self.data,
                phases=self.phases,
                delays=self.x,
                mags=self.data[0, :],  # Just the first row
                measure_name=self.MEASURE_NAME)
        
        print(f"Data saved to {path}")
        return path
    
    @classmethod
    def plot_measurement(cls, idx: int, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement by index.
        
        Args:
            idx: Index of the measurement file from list_measurements()
            y_info: Type of data to plot ("magnitude" or "phase")
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        return super(RamseyMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
    
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
        return super(RamseyMeasurement, cls).plot_from_file(filepath, y_info=y_info, save=save)
