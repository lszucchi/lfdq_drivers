"""
Rabi Measurement Protocol

This module implements Rabi oscillation measurements using the standardized Handler framework.
It combines AWG, Alazar digitizer, RF sources, and other instruments to perform time-domain
measurements of qubit Rabi oscillations.

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output

# Framework imports
from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

# Measurement framework imports (optional)
try:
    from measurements.DataChannelManager import DataChannelManager
    from measurements.PulseSequence import PulseSequence
    from measurements.waveforms.SquarePulse import SquarePulse
    from measurements.waveforms.ZeroPulse import ZeroPulse
    from measurements.waveforms.GaussianPulse import GaussianPulse
    from measurements.waveforms.CosinePulse import CosinePulse
    from measurements.waveforms.DragPulse import DragPulse
except ImportError:
    print("Warning: Measurement classes not found. Some functionality may be limited.")
    DataChannelManager = None
    PulseSequence = None
    SquarePulse = None
    ZeroPulse = None
    GaussianPulse = None
    CosinePulse = None
    DragPulse = None

MEASURE_NAME = "Rabi"
DIR_NAME = "rabi"


class RabiMeasurement(BaseMeasurement):
    """
    Rabi oscillation measurement class using the Handler framework.
    
    This class performs qubit Rabi oscillation measurements by varying the duration
    of the excitation pulse and measuring the qubit state.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    
    def __init__(self, header: str,
                 # Pulse duration range
                 min_duration: float,
                 max_duration: float,
                 n_duration_points: int,
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
                 rf_measurement_length: float = 5e-6,
                 total_experiment_duration: float = 500e-6,
                 pulse_type: str = "square",  # square, gaussian, cosine, drag
                 # Alazar parameters
                 records_per_buffer: int = 100,
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
        
        # Generate duration array for the BaseMeasurement
        durations = np.linspace(min_duration, max_duration, n_duration_points)
        
        # Store measurement parameters
        self.attenuation = attenuation
        self.rf_measurement_amplitude = rf_measurement_amplitude
        self.rf_measurement_frequency = rf_measurement_frequency
        self.rf_excitation_amplitude = rf_excitation_amplitude
        self.rf_excitation_frequency = rf_excitation_frequency
        self.rf_measurement_length = rf_measurement_length
        self.total_experiment_duration = total_experiment_duration
        self.pulse_type = pulse_type
        
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
        
        # Data arrays for storing phases
        self.phases = np.zeros(n_duration_points)
        
        # Initialize handlers with default values if not provided
        handlers = {
            'awg': awg_handler or Handler(InstrumentType.AWG, "169.254.101.100"),
            'alazar': alazar_handler or Handler(InstrumentType.ALAZAR, "ALAZAR_BOARD::0"),
            'rfm': rf_measurement_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1),
            'rfe': rf_excitation_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.104::inst0::INSTR", source_number=2),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        
        # Initialize BaseMeasurement with durations as x-axis and single attenuation value as y-axis
        super().__init__(header, durations, np.array([attenuation]), handlers)
        
        # Initialize data channel manager if available
        if DataChannelManager is not None:
            self.data_manager = DataChannelManager(self.handlers['awg']._driver)
            self._setup_pulse_sequence()
        else:
            self.data_manager = None
    
    def prepare(self):
        """Prepare all instruments with configuration parameters."""
        # Access handlers
        awg = self.handlers['awg']
        alazar = self.handlers['alazar']
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        att = self.handlers['att']
        
        # Configure AWG
        awg.configure_measurement(
            awg_amplitude=1.0,
            awg_offset=0.0,
            ref_clock_frequency=self.awg_ref_clock
        )
        self._configure_awg_specific(self.awg_sample_rate, self.awg_ref_clock)
        
        # Configure RF measurement source
        rfm.configure_measurement(
            rf_amplitude=self.rf_measurement_amplitude,
            rf_frequency=self.rf_measurement_frequency
        )
        self._configure_rf_source_pulse_mode(rfm, source_number=1)
        
        # Configure RF excitation source
        rfe.configure_measurement(
            rf_amplitude=self.rf_excitation_amplitude,
            rf_frequency=self.rf_excitation_frequency
        )
        self._configure_rf_source_pulse_mode(rfe, source_number=2)
        
        # Configure attenuator
        att.configure_measurement(attenuation=self.attenuation)
        
        # Configure Alazar
        self._configure_alazar_specific(
            self.records_per_buffer, 
            self.buffers_per_acquisition,
            self.amplitude_reference_alazar, 
            self.trigger_level_volts,
            self.trigger_range_volts, 
            self.waveform_head_cut, 
            self.period,
            self.buffer_timeout_ms, 
            self.decimation_value
        )
        
        # Update information string
        self.info = f"""
        RABI OSCILLATION MEASUREMENT
        Pulse durations: {self.x[0]*1e9:.1f} to {self.x[-1]*1e9:.1f} ns ({len(self.x)} pts)
        Pulse type: {self.pulse_type}
        Attenuation: {self.attenuation} dB
        RF Measurement Frequency: {self.rf_measurement_frequency/1e9:.6f} GHz
        RF Measurement Amplitude: {self.rf_measurement_amplitude} dBm
        RF Excitation Frequency: {self.rf_excitation_frequency/1e9:.6f} GHz
        RF Excitation Amplitude: {self.rf_excitation_amplitude} dBm
        RF Measurement Length: {self.rf_measurement_length*1e6:.2f} μs
        AWG Sample Rate: {self.awg_sample_rate/1e9:.1f} GS/s
        Alazar Records/Buffer: {self.records_per_buffer}
        Alazar Buffers/Acquisition: {self.buffers_per_acquisition}
        """
    
    def _configure_awg_specific(self, sample_rate: float, ref_clock: float):
        """Configure AWG-specific settings via direct driver calls."""
        awg = self.handlers['awg']
        try:
            # Sample rate and clock configuration
            if hasattr(awg._driver, 'setSampleRate'):
                awg._safe_driver_call('setSampleRate', sample_rate)
            if hasattr(awg._driver, 'setRefInClockExternal'):
                awg._safe_driver_call('setRefInClockExternal')
            if hasattr(awg._driver, 'setRefInClockFrequency'):
                awg._safe_driver_call('setRefInClockFrequency', ref_clock)
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
    
    def _configure_rf_source_pulse_mode(self, rf_handler: Handler, source_number: int):
        """Configure RF source for pulse mode operation."""
        try:
            # Set pulse mode and polarities
            if hasattr(rf_handler._driver, 'set_pulse_trigger_external'):
                rf_handler._safe_driver_call('set_pulse_trigger_external')
            
            # Set polarity based on source number
            if source_number == 1:
                # Measurement source gets inverted polarity
                if hasattr(rf_handler._driver, 'setPulsePolarityInverted'):
                    rf_handler._safe_driver_call('setPulsePolarityInverted')
            elif source_number == 2:
                # Excitation source gets normal polarity
                if hasattr(rf_handler._driver, 'setPulsePolarityNormal'):
                    rf_handler._safe_driver_call('setPulsePolarityNormal')
            
            # Start modulation
            if hasattr(rf_handler._driver, 'start_mod'):
                rf_handler._safe_driver_call('start_mod')
                
        except Exception as e:
            print(f"Warning: RF source {source_number} pulse configuration failed: {e}")
    
    def _configure_alazar_specific(self, records_per_buffer: int, buffers_per_acquisition: int,
                                 amplitude_reference_alazar: float, trigger_level_volts: float,
                                 trigger_range_volts: float, waveform_head_cut: int,
                                 period: int, buffer_timeout_ms: int, decimation_value: int):
        """Configure Alazar-specific settings."""
        alazar = self.handlers['alazar']
        
        # Calculate sampling parameters
        default_sampling_rate = 1e9
        sampling_rate = default_sampling_rate / decimation_value
        post_trigger_samples = int(self.rf_measurement_length * sampling_rate / 256) * 256
        
        # Use Handler's configure_measurement with Alazar parameters
        alazar.configure_measurement(
            pre_trigger_samples=0,
            post_trigger_samples=post_trigger_samples,
            records_per_buffer=records_per_buffer,
            buffers_per_acquisition=buffers_per_acquisition,
            power_reference_dbm=amplitude_reference_alazar,
            input_load=50,
            trigger_level_volts=trigger_level_volts,
            trigger_range_volts=trigger_range_volts,
            waveform_head_cut=waveform_head_cut,
            period=period,
            buffer_timeout_ms=buffer_timeout_ms,
            decimation_value=decimation_value,
            ttl_trigger=True
        )
    
    def _setup_pulse_sequence(self):
        """Setup initial pulse sequence for AWG."""
        if self.data_manager is None:
            return
        
        try:
            # Clear and setup AWG channels
            if hasattr(self.data_manager, 'clearAwgChannel'):
                self.data_manager.clearAwgChannel()
            
            # Label AWG channels
            if hasattr(self.data_manager, 'labelAwgChannel'):
                self.data_manager.labelAwgChannel(channel=2, channelName='Q', markerValue=1)
                self.data_manager.labelAwgChannel(channel=1, channelName='I', markerValue=1)
                self.data_manager.labelAwgChannel(channel=1, channelName='m', markerValue=2)
            
            # Create initial pulse sequence
            if PulseSequence is not None:
                # Define excitation pulse and measurement pulse attributes to avoid AttributeError
                self.excitation_pulse = None
                self.measurement_pulse = None
                self.pulse_sequence = None
                
                # Define pulses based on the pulse_type
                if self.pulse_type == "square" and SquarePulse is not None:
                    self.excitation_pulse = SquarePulse(length=0, amplitude=31)  # Integer amplitude
                elif self.pulse_type == "gaussian" and GaussianPulse is not None:
                    self.excitation_pulse = GaussianPulse(length=0, amplitude=31, sigma=1)  # Integer parameters
                elif self.pulse_type == "cosine" and CosinePulse is not None:
                    self.excitation_pulse = CosinePulse(length=0, amplitude=31)
                elif self.pulse_type == "drag" and DragPulse is not None:
                    self.excitation_pulse = DragPulse(length=0, amplitude=31, sigma=1, beta=1)  # Integer parameters
                else:
                    # If requested type is not available, try the default square pulse
                    if SquarePulse is not None:
                        self.excitation_pulse = SquarePulse(length=0, amplitude=31)
                
                # Create measurement pulse (always a zero pulse with length for the cavity measurement)
                if ZeroPulse is not None:
                    self.measurement_pulse = ZeroPulse(length=self.rf_measurement_length)
                
                # Create pulse sequence if both pulses are available
                if self.excitation_pulse is not None and self.measurement_pulse is not None and PulseSequence is not None:
                    self.pulse_sequence = PulseSequence('Rabi', 10e-9)
                    self.pulse_sequence.clear()
                    
                    # Add initial pulses to sequence
                    self.pulse_sequence.add(p=self.excitation_pulse, channel='I', delay=self.rf_measurement_length)
                    self.pulse_sequence.add(p=self.measurement_pulse, channel='m')
                    
                    # Prepare and load initial sequence
                    if hasattr(self.data_manager, 'prepareChannelData'):
                        self.data_manager.prepareChannelData(
                            sequence=self.pulse_sequence,
                            totalExperimentDuration=self.total_experiment_duration
                        )
                
                # Allocate and load memory
                if hasattr(self.handlers['awg']._driver, 'freeMemory'):
                    self.handlers['awg']._safe_driver_call('freeMemory')
                
                if hasattr(self.data_manager, 'allocAwgMemory'):
                    self.data_manager.allocAwgMemory()
                    
                if hasattr(self.data_manager, 'loadChannelDataToAwg'):
                    self.data_manager.loadChannelDataToAwg('I')
                    self.data_manager.loadChannelDataToAwg('m')
                
                if hasattr(self.data_manager, 'setInstrumentsMarker'):
                    self.data_manager.setInstrumentsMarker(1)
                
        except Exception as e:
            print(f"Warning: Pulse sequence setup failed: {e}")
    
    def _start_measurement(self):
        """Start AWG and RF sources."""
        # Start AWG
        awg = self.handlers['awg']
        if hasattr(awg._driver, 'start'):
            awg._safe_driver_call('start')
        
        # Start RF sources
        rfm = self.handlers['rfm']
        rfe = self.handlers['rfe']
        
        if hasattr(rfm._driver, 'start_rf'):
            rfm._safe_driver_call('start_rf')
        if hasattr(rfe._driver, 'start_rf'):
            rfe._safe_driver_call('start_rf')
        
        # Allow time for instruments to stabilize
        time.sleep(1)
    
    def measure_point(self, duration: float, att: float):
        """
        Measure a single point using the specified pulse duration.
        
        Args:
            duration: Pulse duration to measure
            att: Attenuation value (unused in this implementation since we fix attenuation)
            
        Returns:
            float: Magnitude measurement
        """
        if self.data_manager is None or self.pulse_sequence is None or self.excitation_pulse is None or self.measurement_pulse is None:
            return 0.0
        
        try:
            # Update pulse duration
            self.excitation_pulse.length = duration
            
            # Clear and recreate pulse sequence
            self.pulse_sequence.clear()
            self.pulse_sequence.add(p=self.excitation_pulse, channel='I', delay=self.rf_measurement_length)
            self.pulse_sequence.add(p=self.measurement_pulse, channel='m')
            
            # Update and load sequence to AWG
            if hasattr(self.data_manager, 'updateChannelData'):
                self.data_manager.updateChannelData(sequence=self.pulse_sequence, channelName='I')
                time.sleep(0.5)
            
            if hasattr(self.data_manager, 'loadChannelDataToAwg'):
                self.data_manager.loadChannelDataToAwg('I')
                time.sleep(0.5)
                self.data_manager.loadChannelDataToAwg('m')
                time.sleep(0.5)
            
            if hasattr(self.data_manager, 'setInstrumentsMarker'):
                self.data_manager.setInstrumentsMarker(1)
                time.sleep(0.5)
            
            # Capture data from Alazar
            I, Q = self.handlers['alazar'].capture_alazar_data()
            time.sleep(0.5)
            
            # Calculate magnitude and phase
            idx = np.abs(self.x - duration).argmin()  # Find nearest index for safety
            magnitude = 20 * np.log10(np.sqrt(I**2 + Q**2))
            phase = np.arctan2(Q, I)
            
            # Store phase for later use
            self.phases[idx] = phase
            
            return magnitude
            
        except Exception as e:
            print(f"Error measuring point at duration {duration}: {e}")
            return 0.0
    
    def run(self):
        """
        Override BaseMeasurement.run() to implement specialized startup/shutdown.
        
        Returns:
            dict: Measurement results
        """
        # Start instruments
        self._start_measurement()
        
        try:
            # Let BaseMeasurement handle the loop
            super().run()
        finally:
            # Stop instruments
            self.cleanup()
        
        return {
            'durations': self.x,
            'mags': self.data[0],  # First row of data is magnitudes
            'phases': self.phases,
            'header': self.header,
            'info': self.info,
            'attenuation': self.attenuation
        }
    
    def plot(self, save=True, y_info="magnitude"):
        """
        Plot measurement results.
        
        Args:
            save: Whether to save the plot
            y_info: Type of data to plot ("magnitude" or "phase")
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        data_x = self.x * 1e9  # Convert to ns for display
        data_y = self.data[0] if y_info == "magnitude" else self.phases
        
        xaxis_name = "Pulse Duration (ns)"
        yaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (rad)"
        
        # Create name info
        name_info = [
            f"Att {self.attenuation} dB",
            f"Pulse type: {self.pulse_type}",
            f"Excitation: {self.rf_excitation_frequency/1e9:.6f} GHz"
        ]
        
        title = f"{self.MEASURE_NAME} Oscillation Measurement"
        
        # Create plot using standardized Plot class
        plot_obj = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name=xaxis_name,
            yaxis_name=yaxis_name,
            name_info=name_info,
            title=title,
            data_x=data_x,
            data_y=data_y
        )
        
        fig = plot_obj.plot(save=save)
        fig.show()
        return fig
    
    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, data, header, and info.
        Override to ensure consistent format for plot_from_file functionality.
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
                 durations=self.x,
                 attenuation=self.y[0],
                 mags=self.data[0],
                 phases=self.phases,
                 pulse_type=self.pulse_type,
                 measure_name=self.MEASURE_NAME)
        print(f"Data saved to {path}")
        return path
        
    @classmethod
    def plot_measurement(cls, idx: int, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement by index.
        
        Args:
            idx: Index of the measurement file from list_measurements()
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        return super(RabiMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
        
    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot measurement data from a file.
        
        Args:
            filepath: Path to the saved data file (.npz or .npy)
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Extract data using standard keys
        durations = data['x'] if 'x' in data else data['durations']
        durations_ns = durations * 1e9  # Convert to ns for display
        
        # Get magnitude data
        if 'data' in data and len(data['data'].shape) > 1:
            mags = data['data'][0]  # First row contains magnitudes
        else:
            mags = data['mags'] if 'mags' in data else None
        
        # Get phase data
        phases = data['phases'] if 'phases' in data else np.zeros_like(durations)
        
        # Get info and header
        info = str(data.get('info', 'No information available'))
        attenuation = float(data.get('attenuation', 0))
        pulse_type = str(data.get('pulse_type', 'square'))
        
        # Prepare plot data
        data_x = durations_ns
        data_y = mags if y_info == "magnitude" else phases
        
        xaxis_name = "Pulse Duration (ns)"
        yaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (rad)"
        
        # Create name info
        name_info = [
            f"Att {attenuation} dB",
            f"Pulse type: {pulse_type}"
        ]
        
        title = f"{cls.MEASURE_NAME} Oscillation Measurement"
        
        # Create plot using standardized Plot class
        plot_obj = Plot(
            measure_name=cls.MEASURE_NAME,
            measurement_info=info,
            dir_name=cls.DIR_NAME,
            xaxis_name=xaxis_name,
            yaxis_name=yaxis_name,
            name_info=name_info,
            title=title,
            data_x=data_x,
            data_y=data_y
        )
        
        fig = plot_obj.plot(save=save)
        fig.show()
        return fig

    def cleanup(self):
        """
        Perform cleanup of all instruments after measurement.
        """
        try:
            # Specific cleanup for RF sources (turn off RF)
            if hasattr(self.handlers['rfm']._driver, 'stop_rf'):
                self.handlers['rfm']._safe_driver_call('stop_rf')
            if hasattr(self.handlers['rfe']._driver, 'stop_rf'):
                self.handlers['rfe']._safe_driver_call('stop_rf')
            
            # Stop AWG
            if hasattr(self.handlers['awg']._driver, 'stop'):
                self.handlers['awg']._safe_driver_call('stop')
            
            # Set all instruments to idle state
            for handler in self.handlers.values():
                handler.set_idle()
                
            print("Measurement resources cleaned up.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

