"""
PunchOutAWG Measurement Protocol

This module implements AWG-based punchout measurements using the standardized Handler framework.
It combines AWG, Alazar digitizer, RF sources, and other instruments to perform cavity punchout
measurements with precise timing control.

The class follows the same pattern as PunchOut.py but uses AWG+Alazar instead of VNA.

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
except ImportError:
    print("Warning: Measurement classes not found. Some functionality may be limited.")
    DataChannelManager = None
    PulseSequence = None
    SquarePulse = None
    ZeroPulse = None

MEASURE_NAME = "PunchOutAWG"
DIR_NAME = "punchoutawg"


class PunchOutAWGMeasurement(BaseMeasurement):
    """
    AWG-based punchout measurement class using the Handler framework.
    
    This class performs cavity punchout measurements using AWG for pulse generation,
    Alazar for data acquisition, RF sources for signal generation, and other instruments
    for system control.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    
    def __init__(self, header: str,
                 # Frequency range for measurement
                 start_freq: float,
                 stop_freq: float,
                 n_freq_points: int,
                 # Instrument parameters
                 if_frequency: float,
                 if_level: float,
                 attenuation: float,
                 rf_measurement_amplitude: float,
                 rf_measurement_frequency: float,
                 rf_measurement_phase: float,
                 rf_excitation_amplitude: float,
                 rf_excitation_frequency: float,
                 rf_excitation_phase: float,
                 # AWG parameters
                 awg_sample_rate: float,
                 awg_ref_clock: float,
                 # Pulse parameters
                 rf_measurement_length: float,
                 total_experiment_duration: float,
                 # Alazar parameters
                 pre_trigger_samples: int,
                 records_per_buffer: int,
                 buffers_per_acquisition: int,
                 amplitude_reference_alazar: float,
                 trigger_level_volts: float,
                 trigger_range_volts: float,
                 waveform_head_cut: int,
                 buffer_timeout_ms: int,
                 decimation_value: int,
                 input_load: int,
                 pulse_sequence=None,
                 # Voltage source
                 volt_source_voltage: float = 0.0,
                 # Handler instances (optional)
                 awg_handler: Optional[Handler] = None,
                 alazar_handler: Optional[Handler] = None,
                 rf_measurement_handler: Optional[Handler] = None,
                 rf_excitation_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None,
                 voltage_source_handler: Optional[Handler] = None,
                 data_channel_handler: Optional[Handler] = None,
                 if_signal_handler: Optional[Handler] = None,
                 # Additional notes
                 notes: Optional[str] = None):
        
        self.notes = notes
        self._interrupted = False
        
        # Generate frequency array for the BaseMeasurement
        freqs = np.linspace(start_freq, stop_freq, n_freq_points)

        # Store measurement parameters      31/10/25 foi comentado TODO
        self.if_frequency = if_frequency
        self.if_level = if_level
        self.attenuation = attenuation
        if isinstance(attenuation, (list, np.ndarray)):
            self.y = np.array(attenuation)
        else:
            self.y = np.array([attenuation])
            
        self.rf_measurement_amplitude = rf_measurement_amplitude
        self.rf_measurement_frequency = rf_measurement_frequency
        self.rf_measurement_phase = rf_measurement_phase
        self.rf_excitation_amplitude = rf_excitation_amplitude
        self.rf_excitation_frequency = rf_excitation_frequency
        self.rf_excitation_phase = rf_excitation_phase
        self.rf_measurement_length = rf_measurement_length
        self.total_experiment_duration = total_experiment_duration
        
        # AWG parameters
        self.awg_sample_rate = awg_sample_rate
        self.awg_ref_clock = awg_ref_clock
        self.pulse_sequence = pulse_sequence
            
        # Alazar parameters
        self.pre_trigger_samples = pre_trigger_samples
        self.records_per_buffer = records_per_buffer
        self.buffers_per_acquisition = buffers_per_acquisition
        self.amplitude_reference_alazar = amplitude_reference_alazar
        self.trigger_level_volts = trigger_level_volts
        self.trigger_range_volts = trigger_range_volts
        self.waveform_head_cut = waveform_head_cut
        self.buffer_timeout_ms = buffer_timeout_ms
        self.decimation_value = decimation_value
        self.input_load = input_load
        
        # Voltage source parameter
        self.volt_source_voltage = volt_source_voltage
        
        # Initialize handlers with default values if not provided
        handlers = {
            'awg': awg_handler,
            'data_channel': data_channel_handler,
            'alazar': alazar_handler,
            'rf_measurement': rf_measurement_handler,
            'rf_excitation': rf_excitation_handler,
            'attenuator': attenuator_handler,
            'if_signal': if_signal_handler,
            'volt_source': voltage_source_handler
        }
        
        # Add voltage source if needed
        if self.volt_source_voltage != 0:
            handlers['volt_source'] = voltage_source_handler or Handler(InstrumentType.VOLTAGE_SOURCE, "GPIB0::2::INSTR", port=8, step_voltage=0.001)
        
        # Data arrays for storing phases - make 2D like TwoTone.py for 3D plotting
        self.phases = np.zeros((self.y.size, n_freq_points))
        
        # Initialize BaseMeasurement with the full attenuation array
        super().__init__(header, freqs, self.y, handlers)
        
    
    def prepare(self):
        """Prepare all instruments with configuration parameters."""
        # Access handlers
        awg = self.handlers['awg']
        awg._driver.open()
        data_channel = self.handlers['data_channel']
        alazar = self.handlers['alazar']
        rf_measurement = self.handlers['rf_measurement']
        rf_excitation = self.handlers['rf_excitation']
        attenuator = self.handlers['attenuator']
        if_signal = self.handlers['if_signal']   # 31/10/25 foi comentado TODO
        #if_signal._driver.reconnect()  # Ensure IF signal generator is connected
        volt_source = self.handlers['volt_source']
        
        # Configure AWG
        awg.configure_measurement(
            sample_rate = self.awg_sample_rate, 
            ref_clock_frequency = self.awg_ref_clock, 
            ref_in_clock_external = True,
            channel_setting = {1: True, 2: True, 3: True, 4: True},
            channel_voltage = {1: 0.5, 2: 0.5, 3: 1.0, 4: 1.0},
            channel_voltage_offset = {1: 0.0, 2: 0.0, 3: 0.5, 4: 0.5}
        )

        data_channel.configure_measurement(
            rf_measurement_length = self.rf_measurement_length,
            total_experiment_duration = self.total_experiment_duration,
            pulse_sequence=self.pulse_sequence
            )
        
        # Configure RF measurement source
        if rf_measurement is not None:
            rf_measurement.configure_measurement(
                rf_amplitude=self.rf_measurement_amplitude,
                rf_frequency=self.rf_measurement_frequency,
                rf_mod=True,
                pulse_trigger_source='EXT',
                rf_phase = self.rf_measurement_phase
            )
        
        # Configure RF excitation source
        if rf_excitation is not None:
            rf_excitation.configure_measurement(
                rf_amplitude=self.rf_excitation_amplitude,
                rf_frequency=self.rf_excitation_frequency,
                rf_mod=True,
                pulse_trigger_source='EXT',
                rf_phase = self.rf_excitation_phase
            )
        
        # Configure Alazar
        alazar.configure_measurement(
            if_frequency = self.if_frequency,
            rf_measurement_length = self.rf_measurement_length,
            pre_trigger_samples = self.pre_trigger_samples,
            records_per_buffer = self.records_per_buffer,
            buffers_per_acquisition = self.buffers_per_acquisition,
            amplitude_reference_alazar = self.amplitude_reference_alazar,
            trigger_level_volts = self.trigger_level_volts,
            trigger_range_volts = self.trigger_range_volts,
            waveform_head_cut = self.waveform_head_cut,
            buffer_timeout_ms = self.buffer_timeout_ms,
            decimation_value = self.decimation_value
        )
        
        # Configure attenuator
        attenuator.configure_measurement(attenuation=self.y[0])

        # Configure IF signal generator        31/10/25 foi comentado TODO
        #if_signal.configure_measurement(
        #    rf_frequency = self.if_frequency,
        #    rf_level = self.if_level
        #)

        # Configure voltage source if present
        if volt_source is not None:
            volt_source.configure_measurement(voltage=self.volt_source_voltage)

        self.estimated_duration = 1.7 * self.total_experiment_duration * len(self.x) * len(self.y) \
            * (self.records_per_buffer * self.buffers_per_acquisition)
        
        minutes, seconds = divmod(self.estimated_duration, 60)
        hours, minutes = divmod(minutes, 60)

        if isinstance(self.attenuation, np.ndarray):
            self.att_info = f"Attenuation: {self.attenuation[0]} - {self.attenuation[-1]} dB, {len(self.attenuation)} points"
        else:
            self.att_info = f"Attenuation: {self.attenuation} dB"
        
        if rf_measurement is not None:
            rf_measurement_info =  f"{rf_measurement.config.rf_amplitude} dBm, {rf_measurement.config.rf_frequency:.2e} Hz"
        else:
            rf_measurement_info = "No RF measurement source configured."
        
        if rf_excitation is not None:
            rf_excitation_info = f"{rf_excitation.config.rf_amplitude} dBm, {rf_excitation.config.rf_frequency:.2e} Hz"
        else:
            rf_excitation_info = "No RF excitation source configured."
        
        if volt_source is not None:
            volt_source_info = f"{volt_source.config.voltage} V"
        else:
            volt_source_info = "No Voltage source configured."
        
        # Update information string
        self.info = f"""
        PUNCHOUT AWG MEASUREMENT
        {self.att_info}
        Frequency Range: {self.x[0]:.2e} - {self.x[-1]:.2e} Hz
        Number of Frequency Points: {len(self.x)}
        IF Frequency: {self.if_frequency:.2e} Hz
        IF Level: {self.if_level} dBm
        RF Measurement Amp/Freq: {rf_measurement_info}
        RF Excitation Amp/Freq: {rf_excitation_info}
        Voltage Source Voltage: {volt_source_info}
        Total Experiment Duration: {self.total_experiment_duration:.2e} s
        AWG Sample Rate: {self.awg_sample_rate:.2e} Samples/s
        AWG Reference Clock: {self.awg_ref_clock:.2e} Hz
        Alazar Records/Buffer: {self.records_per_buffer}
        Alazar Buffers/Acquisition: {self.buffers_per_acquisition}
        Estimated Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s
        """

        # Add notes to self.info if provided
        if self.notes:
            self.info += f"Notes: {self.notes}\n\t\t"

        self.show()
    
    
    def _start_measurement(self):
        """Start AWG and RF sources."""
        # Get handlers
        awg = self.handlers['awg']
        awg._driver.open()
        #if_signal = self.handlers['if_signal']    31/10/25 foi comentado TODO
        #if_signal._driver.reconnect()  # Ensure IF signal generator is connected
        
        # First stop AWG
        if hasattr(awg._driver, 'stop'):
            try:
                awg._safe_driver_call('stop')
                time.sleep(0.5)  # Allow AWG to fully stop
            except Exception as e:
                print(f"Error stopping AWG: {e}")
        
        # Start AWG
        if hasattr(awg._driver, 'start'):
            try:
                awg._safe_driver_call('start')
            except Exception as e:
                print(f"Error starting AWG: {e}")
        
        # Start RF measurement source
        if 'rf_measurement' in self.handlers and self.handlers['rf_measurement'] is not None:
            rf_measurement = self.handlers['rf_measurement']
            if hasattr(rf_measurement._driver, 'start_rf'):
                try:
                    rf_measurement._safe_driver_call('start_rf')
                except Exception as e:
                    print(f"Error starting RF measurement source: {e}")
        
        # Start RF excitation if available
        if 'rf_excitation' in self.handlers and self.handlers['rf_excitation'] is not None:
            rf_excitation = self.handlers['rf_excitation']
            if hasattr(rf_excitation._driver, 'start_rf'):
                try:
                    rf_excitation._safe_driver_call('start_rf')
                except Exception as e:
                    print(f"Error starting RF excitation source: {e}")
        
        # Start IF signal if available            31/10/25 foi comentado TODO
        #if 'if_signal' in self.handlers and self.handlers['if_signal'] is not None:
        #    if_signal = self.handlers['if_signal']
        #    if hasattr(if_signal._driver, 'rf_on'):
        #        try:
        #            if_signal._safe_driver_call('rf_on')
        #        except Exception as e:
        #            print(f"Error starting IF signal source: {e}")
        
        time.sleep(1)  # Allow instruments to stabilize
    
    def _start_rf_sources(self):
        """Start RF sources."""
        if 'rf_measurement' in self.handlers and self.handlers['rf_measurement'] is not None:
            rf_measurement = self.handlers['rf_measurement']
            if hasattr(rf_measurement._driver, 'start_rf'):
                try:
                    rf_measurement._safe_driver_call('start_rf')
                except Exception as e:
                    print(f"Error starting RF measurement source: {e}")
        
        if 'rf_excitation' in self.handlers and self.handlers['rf_excitation'] is not None:
            rf_excitation = self.handlers['rf_excitation']
            if hasattr(rf_excitation._driver, 'start_rf'):
                try:
                    rf_excitation._safe_driver_call('start_rf')
                except Exception as e:
                    print(f"Error starting RF excitation source: {e}")
    
    def _set_y(self, att):
        """
        Set the y value for the measurement.
        
        Args:
            y_val: Attenuation value to set
        """
        attenuator = self.handlers['attenuator']
        if attenuator is not None:
            attenuator.configure_measurement(attenuation=att)
        else:
            raise RuntimeError("Attenuator handler not initialized.")
        
    
    def measure_point(self, freq: float, att: float):
        """
        Measure a single point using the specified frequency.
        
        Args:
            freq: RF frequency to measure
            att: Attenuation value (unused in this implementation since we fix attenuation)
            
        Returns:
            float: Magnitude measurement
        """
        # Get handlers
        if 'rf_measurement' not in self.handlers or 'alazar' not in self.handlers:
            raise RuntimeError("RF measurement or Alazar handler not initialized.")
        if 'rf_measurement' in self.handlers:
            rf_measurement = self.handlers['rf_measurement']
        elif 'rf_excitation' in self.handlers and 'rf_measurement' not in self.handlers:
            rf_measurement = self.handlers['rf_excitation']

        alazar = self.handlers['alazar']
        attenuator = self.handlers['attenuator'] 

        
        # Set measurement frequency (with offset for mixing if needed)
        measurement_freq = freq - self.if_frequency  # Based on notebook implementation
        rf_measurement.configure_measurement(rf_frequency=measurement_freq)
        
        attenuator.configure_measurement(attenuation=att)
        
        time.sleep(0.05)  # Allow frequency to settle            # Capture data using the direct Alazar driver with parameters
        try:
            alazar_data = alazar.get_data()
            if isinstance(alazar_data, dict) and 'I' in alazar_data and 'Q' in alazar_data:
                I, Q = alazar_data['I'], alazar_data['Q']
            else:
                print(f"Warning: Unexpected data format from Alazar: {type(alazar_data)}")
                I, Q = 0.0, 0.0
        except Exception as e:
            print(f"Error capturing data: {e}")
            I, Q = 0.0, 0.0
        
        # Calculate magnitude and phase
        magnitude = 20 * np.log10(np.sqrt(I**2 + Q**2))
        phase = np.arctan2(Q, I)
        
        # Find the current y index (row) where this attenuation value is being used
        j = np.where(self.y == att)[0][0]
        # Find the current x index (column) where this frequency is being used
        i = np.where(self.x == freq)[0][0]
        
        # Store phase in 2D array for proper 3D plotting
        self.phases[j, i] = phase
        
        # Return magnitude
        return magnitude
    
    # The capture_data method has been removed as we're directly using 
    # the Alazar driver's capture method in measure_point
    
    def run(self):
        """
        Override BaseMeasurement.run() to implement specialized startup/shutdown
        with custom partial plotting.
        
        Returns:
            dict: Measurement results
        """
        # Start instruments
        self._start_measurement()
        
        try:
            # Display measurement info
            self.show()
            
            total = len(self.y) * len(self.x)
            count = 0
            start_time = datetime.now()
            
            # Pre-allocate full arrays with NaN values for partial plotting
            # This ensures the plot always uses the same scale
            full_x_data = self.x  # Use full range for x-axis
            
            # Initialize min/max values for consistent color scale in 3D plots
            vmin = float('inf')
            vmax = float('-inf')
            
            for j, yv in enumerate(self.y):
                # Set attenuation value
                self._set_y(yv)
                
                # Initialize full data array with NaNs for progressive plotting
                full_y_data = np.full(len(self.x), np.nan)
                
                for i, xv in enumerate(self.x):
                    # Measure at current frequency
                    result = self.measure_point(xv, yv)
                    
                    # Store results in the main data arrays
                    self.data[j, i] = result
                    
                    # Update the plotting array with the new point
                    full_y_data[i] = result
                    
                    # Update min/max values for consistent color scale
                    if not np.isnan(result):
                        vmin = min(vmin, result)
                        vmax = max(vmax, result)
                    
                    # Update progress
                    count += 1
                    
                    # Clear output and show updated info at regular intervals
                    if count % 2 == 0 or i == len(self.x) - 1 or count == 1:
                        clear_output(wait=True)
                        
                        # For single attenuation, show line plot updates
                        if len(self.y) == 1:
                            # Estimate y-axis limits based on current data to keep scale consistent
                            valid_data = full_y_data[~np.isnan(full_y_data)]
                            if len(valid_data) > 0:
                                y_min = np.nanmin(valid_data) - 1  # Add margin
                                y_max = np.nanmax(valid_data) + 1  # Add margin
                            else:
                                y_min, y_max = -80, 0  # Default range if no data yet
                            
                            # Create the figure with full x-axis range but only plot measured points
                            fig = go.Figure()
                            
                            # Plot measured data with connected line where we have measurements
                            fig.add_trace(go.Scatter(
                                x=full_x_data,
                                y=full_y_data,
                                mode='lines+markers',
                                name=f"Attenuation: {float(self.y[0])} dB",
                                connectgaps=False  # Don't connect across NaN values
                            ))
                            
                            # Configure the layout to maintain the full range
                            fig.update_layout(
                                title=f"PunchOut AWG Spectroscopy (in progress: {count}/{total} points)",
                                xaxis_title="Frequency (Hz)",
                                yaxis_title="Magnitude (dB)",
                                xaxis=dict(
                                    range=[min(full_x_data), max(full_x_data)]
                                ),
                                yaxis=dict(
                                    range=[y_min, y_max]
                                ),
                                height=600,
                                width=800
                            )
                            display(fig)
                            self._show_progress(count, total, start_time)
                            self.show()
                        # For multiple attenuation values, after completing a full row, show 3D plot update
                        elif count % 2 == 0 or i == len(self.x) - 1:
                            # Create a full-sized data array with NaN values for unmeasured points
                            # This ensures consistent color scale and axis limits
                            full_sized_data = np.full((len(self.y), len(self.x)), np.nan)
                            # Fill in only the measured values
                            full_sized_data[:j+1, :] = self.data[:j+1, :]
                            partial_data = full_sized_data.copy()
                            partial_data[j, i+1:] = np.nan  # Keep current point NaN for partial plot
                            

                            # Use Plot class for visualization of 3D data
                            progress_fig = Plot(
                                measure_name=self.MEASURE_NAME,
                                measurement_info=self.info,
                                dir_name=self.DIR_NAME,
                                xaxis_name="Frequency (Hz)",
                                yaxis_name="Attenuation (dB)",
                                name_info=[f"Measured {j+1} of {len(self.y)} attenuation values"],
                                title=f"PunchOut AWG Spectroscopy",
                                data_x=self.x,
                                data_y=self.y,
                                data_z=partial_data,
                                zaxis_name="Magnitude (dB)"
                            ).plot_3d(save=False)  # Don't save intermediate plots
                            
                            # Show the 3D progress plot
                            display(progress_fig)
                            self._show_progress(count, total, start_time)
                            self.show()
                        else:
                            # For iterations between plot updates, just print a compact status update
                            # without clearing the screen, so the plot remains visible
                            elapsed = (datetime.now() - start_time).total_seconds()
                            points_per_second = count / elapsed if elapsed > 0 else 0
                            remaining_points = total - count
                            remaining_seconds = remaining_points / points_per_second if points_per_second > 0 else 0
                            remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
                            print(f"\rPoint {count}/{total} ({count/total*100:.1f}%) - {int(remaining_minutes)}m {int(remaining_seconds)}s remaining", end="", flush=True)
                    
                    # Check if measurement was interrupted
                    if hasattr(self, '_interrupted') and self._interrupted:
                        self._interrupted = True
                        raise KeyboardInterrupt
            
            # Find maximum point (useful for resonance identification)
            if np.any(~np.isnan(self.data)):
                if len(self.y) == 1:
                    max_idx = np.nanargmax(self.data[0, :])
                    self.max_freq = self.x[max_idx]
                    self.max_mag = self.data[0, max_idx]
                    print(f"\nMaximum response at frequency: {self.max_freq:.2e} Hz")
                else:
                    # For 3D data, find global maximum
                    max_indices = np.unravel_index(np.nanargmax(self.data), self.data.shape)
                    self.max_freq = self.x[max_indices[1]]
                    self.max_att = float(self.y[max_indices[0]])
                    self.max_mag = float(self.data[max_indices])
                    print(f"\nMaximum response at frequency: {self.max_freq:.2e} Hz, attenuation: {self.max_att} dB")
            
            # Clear last output before saving and plotting final results
            clear_output(wait=True)
            
            # Calculate and display total elapsed time
            elapsed_time = datetime.now() - start_time
            minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
            hours, minutes = divmod(minutes, 60)
            elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            # Save and plot final results
            self.save_data()
            self.show()
            print(f"\nMeasurement completed successfully! Total time: {elapsed_str}")
            
            # Plot final results
            final_plot = self.plot(save=True, partial=False)
            display(final_plot)
            
        except KeyboardInterrupt:
            print("\nMeasurement interrupted by user.")
            elapsed_time = datetime.now() - start_time
            minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
            hours, minutes = divmod(minutes, 60)
            elapsed_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            print(f"Elapsed time before interruption: {elapsed_str}")
            self._interrupted = True
            # Plot partial data when interrupted
            self.plot(save=True, partial=True)
        except Exception as e:
            print(f"\nError during measurement: {e}")
            import traceback
            traceback.print_exc()
            self._interrupted = True
            # Plot partial data when error occurs
            self.plot(save=True, partial=True)
        finally:
            # Stop instruments
            self.cleanup()

            
    def plot(self, save=True, y_info="magnitude", partial=False):
        """
        Plot measurement results, optionally supporting partial plotting during measurement.
        
        Args:
            save: Whether to save the plot
            y_info: Type of data to plot ("magnitude" or "phase")
            partial: Whether this is a partial plot during measurement (True) or final plot (False)
            
        Returns:
            Plotly figure object
        """
        # Prepare data based on whether it's a 2D or 3D plot
        measure_name = getattr(self, 'MEASURE_NAME', self.__class__.__name__)
        dir_name = getattr(self, 'DIR_NAME', self.DIR_NAME)
        name_info = [f'{self.att_info}', f'{self.x[0]:.2e} - {self.x[-1]:.2e} Hz']
        
        # Choose plot data based on y_info
        plot_data = self.data if y_info == "magnitude" else self.phases
        
        # Set axis labels
        xaxis_name = "Frequency (Hz)"
        yaxis_name = "Attenuation (dB)"
        zaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (rad)"
        
        # Determine if we should do a 2D or 3D plot
        if len(self.y) <= 1:
            # For 2D plots with a single attenuation value
            if partial:
                # For partial plots, create a NaN array for full scale plotting
                # but show only measured points
                full_y_data = np.full(len(self.x), np.nan)
                
                # Identify measured points (non-zero and non-NaN)
                mask = ~np.isclose(plot_data[0], 0) & ~np.isnan(plot_data[0])
                
                # Fill in measured points only
                full_y_data[mask] = plot_data[0, mask]
                
                # Use full x-axis range but only show measured y values
                data_x = self.x
                data_y = full_y_data
                
                # Add "(Partial)" to title
                plot_title = f"{measure_name} - {y_info.capitalize()} (Partial)"
            else:
                # Full plot with all data
                data_x = self.x
                data_y = plot_data[0]
                plot_title = f"{measure_name} - {y_info.capitalize()}"
            
            # Create plot using the Plot utility class
            fig = Plot(
                measure_name=measure_name,
                measurement_info=self.info,
                dir_name=dir_name,
                xaxis_name=xaxis_name,
                yaxis_name=zaxis_name,
                name_info=name_info,
                title=plot_title,
                data_x=data_x,
                data_y=data_y
            ).plot(save=save)
            
            # Find and annotate maximum point if plotting magnitude and we have data
            if y_info == "magnitude":
                # Filter out NaN values for finding the maximum
                valid_mask = ~np.isnan(data_y)
                if np.any(valid_mask):
                    # Find maximum only among valid (measured) points
                    valid_data = data_y[valid_mask]
                    valid_x = data_x[valid_mask]
                    max_idx = np.argmax(valid_data)
                    max_freq = valid_x[max_idx]
                    max_mag = valid_data[max_idx]
                    
                    # Add annotation for maximum point
                    fig.add_annotation(
                        x=max_freq,
                        y=max_mag,
                        text=f"Max: {max_freq:.2e} Hz, {max_mag:.2f} dB",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2
                    )
        else:
            # For 3D plots with multiple attenuation values
            if partial:
                # Create a full-sized NaN array for consistent scale
                full_sized_data = np.full((len(self.y), len(self.x)), np.nan)
                
                # Identify which data points have been measured
                # Assumption: unmeasured values are exactly 0 or NaN
                measured_mask = ~np.isclose(plot_data, 0) & ~np.isnan(plot_data)
                
                # Fill in only the measured values
                full_sized_data[measured_mask] = plot_data[measured_mask]
                
                plot_title = f"{measure_name} - {y_info.capitalize()} (Partial)"
                plot_z_data = full_sized_data
            else:
                # For final complete plot
                plot_title = f"{measure_name} - {y_info.capitalize()}"
                plot_z_data = plot_data
            
            # Create 3D plot
            fig = Plot(
                measure_name=measure_name,
                measurement_info=self.info,
                dir_name=dir_name,
                xaxis_name=xaxis_name,
                yaxis_name=yaxis_name,
                name_info=name_info,
                title=plot_title,
                data_x=self.x,
                data_y=self.y,
                data_z=plot_z_data,
                zaxis_name=zaxis_name
            ).plot_3d(save=save)
            
            # For 3D data, find global maximum if plotting magnitude and we have data
            if y_info == "magnitude":
                # Use a mask to find maximum only in measured points
                measured_mask = ~np.isnan(plot_z_data) & ~np.isclose(plot_z_data, 0)
                if np.any(measured_mask):
                    # Get only measured values, setting unmeasured to -inf for argmax
                    measured_data = np.where(measured_mask, plot_z_data, -np.inf)
                    max_indices = np.unravel_index(np.argmax(measured_data), measured_data.shape)
                    max_freq = self.x[max_indices[1]]
                    max_att = self.y[max_indices[0]]
                    max_mag = plot_z_data[max_indices]
                    print(f"Maximum response at frequency: {max_freq:.2e} Hz, attenuation: {max_att} dB, magnitude: {max_mag:.2f} dB")
        
        fig.show()

    
    @classmethod
    def plot_from_file(cls, filename: str, y_info: str = "magnitude", save: bool = False):
        """
        Load and plot measurement data from a file.
        
        Args:
            filename: Path to the saved data file (.npz or .npy)
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        data = np.load(filename, allow_pickle=True)
        header = str(data['header'])
        info = str(data['info'])
        attenuation = float(data['attenuation'])
        freqs = data['x'] if 'x' in data else data['freqs']
        
        # Get magnitude data
        if 'data' in data:
            mags = data['data'][0]  # First row contains magnitudes
        else:
            mags = data['mags']
        
        # Get phase data
        if 'phases' in data:
            phases = data['phases']
        else:
            phases = np.zeros_like(freqs)
        
        data_x = freqs
        data_y = mags if y_info == "magnitude" else phases
        
        xaxis_name = "Frequency (Hz)"
        yaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (rad)"
        
        # Find maximum magnitude point
        max_idx = np.argmax(mags)
        max_freq = freqs[max_idx]
        max_mag = mags[max_idx]
        
        title = f"{cls.MEASURE_NAME} Measurement"
        name_info = [f"Att {attenuation} dB", f"Max Freq {max_freq:.2e} Hz"]
        
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
        
        fig = plot_obj.plot(save=False)
        
        # Add maximum point marker
        fig.add_scatter(
            x=[max_freq],
            y=[max_mag],
            mode='markers+text',
            text=[f"({max_freq:.2e}, {max_mag:.2f})"],
            textposition="top center",
            name="Max Point"
        )
        
        return fig
    
    def cleanup(self):
        """
        Perform cleanup of all instruments after measurement.
        """
        try:
            # Use the handler's own cleanup methods for anything else
            for name, handler in self.handlers.items():
                if name == 'if_signal':
                    continue
                if hasattr(handler, 'cleanup'):
                    try:
                        handler.cleanup()
                    except Exception as e:
                        print(f"Error during {name} cleanup: {e}")
                
            print("Measurement resources cleaned up.")
            
        except Exception as e:
            print(f"Warning during cleanup: {e}")
    
    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, data, header, info, and phases.
        Override to ensure consistent format for plot_from_file functionality.
        """
        # Use more robust path handling to ensure directories are created properly
        try:
            # Create data directory with explicit path
            out_dir = Path(self.DIR_NAME) / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filename
            ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            fname = f"{ts}_{self.MEASURE_NAME.lower()}.npz"
            path = out_dir / fname
            
            # Save data with all necessary fields
            np.savez(str(path),
                    header=self.header,
                    info=self.info,
                    x=self.x,
                    y=self.y,
                    data=self.data,
                    freqs=self.x,
                    attenuation=self.y,
                    mags=self.data,
                    phases=self.phases,
                    measure_name=self.MEASURE_NAME)
            
            # Add howtoplot string for backward compatibility with notebook
            howtoplot = f"""
            data = np.load('{fname}')
            freqs = data['x']  # or data['freqs']
            mags = data['data']  # or data['mags']
            phases = data['phases']
            """
            
            print(f"Data saved to {path}")
            print("To plot this data:")
            print(howtoplot)
            
            return path
            
        except Exception as e:
            print(f"Error saving data: {e}")
            # Try an alternative location if the primary fails
            try:
                alt_dir = Path("measurement_data")
                alt_dir.mkdir(exist_ok=True)
                alt_path = alt_dir / f"{ts}_{self.MEASURE_NAME.lower()}.npz"
                
                np.savez(str(alt_path),
                        header=self.header,
                        info=self.info,
                        x=self.x,
                        y=self.y,
                        data=self.data,
                        freqs=self.x,
                        attenuation=self.y,
                        mags=self.data,
                        phases=self.phases,
                        measure_name=self.MEASURE_NAME)
                
                print(f"Data saved to alternative location: {alt_path}")
                return alt_path
            except Exception as e2:
                print(f"Critical error: Failed to save data to any location: {e2}")
                return None
        
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
        return super(PunchOutAWGMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
    
    @staticmethod
    def template_default():
        """
        Default template for PunchOutAWGMeasurement.
        
        Returns:
            dict: Default parameters for PunchOutAWGMeasurement
        """
        return {
            'header': "PunchOut AWG Measurement",
            'start_freq': 7e9,
            'stop_freq': 8e9,
            'n_freq_points': 100,
            'if_frequency': 240e6,
            'if_level': 11.0,
            'attenuation': 0.0,
            'rf_measurement_amplitude': 18,
            'rf_measurement_frequency': 5e9,
            'rf_excitation_amplitude': 18,
            'rf_excitation_frequency': 5e9,
            'awg_sample_rate': 64e9,
            'awg_ref_clock': 10e6,
            'rf_measurement_length': 5e-6,
            'total_experiment_duration': 200e-3,
            'pre_trigger_samples': 0,
            'records_per_buffer': 500,
            'buffers_per_acquisition': 100,
            'amplitude_reference_alazar': -8,
            'trigger_level_volts': 0.7,
            'trigger_range_volts': 1.0,
            'waveform_head_cut': 1500,
            'buffer_timeout_ms': 5000,
            'decimation_value': 1,
            'input_load': 50,
            'pulse_sequence': None,  # Placeholder for PulseSequence object
            'volt_source_voltage': 0.0,  # Default voltage source off
        }
    
    def _show_progress(self, count, total, start_time=None):
        """
        Display a progress bar and estimated remaining time.
        
        Args:
            count: Current count of processed points
            total: Total number of points
            start_time: Start time of the measurement (optional)
        """
        # Calculate percentage progress
        pct = count / total * 100
        
        # Create a progress bar
        bar_len = 40
        filled = int(round(bar_len * pct / 100))
        bar = '#' * filled + '-' * (bar_len - filled)
        
        # Print progress
        print(f"Progress: [{bar}] {pct:.1f}% ({count}/{total})")
        
        # Calculate remaining time if start_time is provided
        if start_time is not None:
            elapsed = (datetime.now() - start_time).total_seconds()
            points_per_second = count / elapsed if elapsed > 0 else 0
            remaining_points = total - count
            remaining_seconds = remaining_points / points_per_second if points_per_second > 0 else 0
            remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
            remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
            
            # Print estimated remaining time
            print(f"Estimated remaining time: {int(remaining_hours)}h {int(remaining_minutes)}m {int(remaining_seconds)}s")
            
            return elapsed
        