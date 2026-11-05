"""
TwoTone Measurement Protocol

This module implements two-tone spectroscopy for superconducting qubits using 
the standardized Handler framework and BaseMeasurement class. It combines VNA, 
RF source, attenuator, and optional voltage source to perform qubit spectroscopy 
by measuring the resonator response while an RF tone excites the qubit.

Author: Measurement System
Date: June 2025
"""

import os
import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
import plotly.graph_objects as go
from IPython.display import display, clear_output
import logging
import traceback
import plotly.graph_objects as go

from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType, InstrumentState
from protocols.Plot import Plot

MEASURE_NAME = "TwoTone"
DIR_NAME = "twotone"

class TwoToneMeasurement(BaseMeasurement):
    """
    Two-tone spectroscopy measurement using VNA and RF source for qubit spectroscopy.
    
    This class performs qubit spectroscopy by measuring the dispersive shift of a 
    readout resonator (using VNA) while an RF source excites the qubit.
    The RF source frequency is swept to find the qubit transition frequency.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME

    def __init__(self, header: str,
                 # Fixed parameters for single measurement
                 attenuation: float,
                 # VNA parameters 
                 vna_amplitude: float,
                 vna_parameter: str,
                 # RF source parameters
                 rf_amplitude: Union[int, np.ndarray],
                 # Frequency sweep parameters
                 start_freq: float,
                 stop_freq: float,
                 step: float,
                 cavity_frequency: float,
                 # Measurement settings
                 if_bandwidth: float,
                 data_array_format: str,
                 average_points: int,
                 vna_smoothing: int = 0,
                 vna_averaging: int = 1,
                 # Voltage source (optional)
                 volt_source_voltage: float = 0,
                 # Handler instances (optional)
                 vna_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None,
                 rf_source_handler: Optional[Handler] = None,
                 voltage_source_handler: Optional[Handler] = None,
                 notes: Optional[str] = None):
        """
        Initialize two-tone spectroscopy measurement.
        
        Args:
            header: Measurement header for file naming
            attenuation: Fixed attenuation value
            vna_amplitude: VNA source power in dBm
            vna_parameter: VNA parameter to measure (S21, S11, etc.)
            rf_amplitude: RF source power in dBm
            start_freq: Start frequency for RF source sweep in Hz
            stop_freq: Stop frequency for RF source sweep in Hz
            step: Frequency step size in Hz
            cavity_frequency: Resonator frequency to probe with VNA in Hz
            if_bandwidth: VNA IF bandwidth in Hz
            data_array_format: VNA data format ("MLOG", "RI", etc.)
            average_points: Number of averaging points
            vna_smoothing: VNA smoothing factor
            vna_averaging: VNA averaging enabled (1) or disabled (0)
            volt_source_voltage: Optional DC voltage bias in V
            vna_handler: Optional pre-configured VNA handler
            attenuator_handler: Optional pre-configured attenuator handler
            rf_source_handler: Optional pre-configured RF source handler
            voltage_source_handler: Optional pre-configured voltage source handler
        """
        self.notes = notes
        # Build frequency array for x-axis (RF source sweep frequencies)
        x = np.arange(start_freq, stop_freq + step, step)
        # Set y-axis as fixed RF amplitude value
        if isinstance(rf_amplitude, (int)):
            y = np.array([rf_amplitude])
        elif isinstance(rf_amplitude, (list, tuple, np.ndarray)):
            y = np.array(rf_amplitude)
        else:
            raise ValueError("rf_amplitude must be an int or a list/tuple/array of values.")
        
        # Initialize handlers for all required instruments
        handlers = {
            'vna': vna_handler or Handler(InstrumentType.VNA, "TCPIP0::169.254.101.105::inst0::INSTR"),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR"),
            'rf': rf_source_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1)
        }
        
        # Add voltage source handler if needed
        if volt_source_voltage != 0:
            handlers['volt_source'] = voltage_source_handler or Handler(InstrumentType.VOLTAGE_SOURCE, "TCPIP0::169.254.101.102::inst0::INSTR", port=1)
        
        # Store measurement parameters
        self.vna_amplitude = vna_amplitude
        self.vna_parameter = vna_parameter
        self.rf_amplitude = rf_amplitude
        self.cavity_frequency = cavity_frequency
        self.if_bandwidth = if_bandwidth
        self.data_array_format = data_array_format
        self.average_points = average_points
        self.vna_smoothing = vna_smoothing
        self.vna_averaging = vna_averaging
        self.volt_source_voltage = volt_source_voltage
        self.attenuation = attenuation
        
        # Data storage for phases (magnitudes stored in self.data by BaseMeasurement)
        self.phases = np.zeros((len(y), len(x)))
        
        # Initialize BaseMeasurement with axes and handlers
        super().__init__(header, x, y, handlers)

    def prepare(self):
        """
        Prepare all instruments for two-tone measurement.
        
        This method configures the VNA to measure the cavity resonance and
        sets up the RF source for sweeping near the qubit frequency.
        """
        # Get handler references
        vna = self.handlers['vna']
        rf = self.handlers['rf']
        att = self.handlers['att']
        
        # Set all instruments to idle first
        self._vna_idle()
        self._rf_idle()
        rf._driver.initial_frequency_config()
        
        # Set attenuator
        att.set_attenuation(self.attenuation)
        
        # Configure VNA to measure at the cavity frequency
        # VNA is set to single point measurement at cavity frequency
        vna.configure_measurement(
            amplitude=self.vna_amplitude,
            parameter=self.vna_parameter,
            center_frequency=self.cavity_frequency,
            start_frequency=self.cavity_frequency,  # Center frequency for zero span
            stop_frequency=self.cavity_frequency,   # Zero span for single frequency
            span_frequency=1e3,  # Zero span for single frequency
            sweep_points=3,    # Single point measurement
            if_bandwidth=self.if_bandwidth,
            data_array_format=self.data_array_format,
            average_points=self.average_points,
            smoothing=self.vna_smoothing,
            averaging_enabled=self.vna_averaging
        )
        
        # Set VNA to single sweep mode
        if hasattr(vna._driver, 'set_sweep_continuous'):
            vna._safe_driver_call('set_sweep_continuous', False)
        
        # Configure RF source for frequency sweep
        rf.configure_measurement(
            rf_amplitude=self.y[0],
            rf_frequency=self.x[0]
        )
        
        # Set voltage source if needed
        if 'volt_source' in self.handlers:
            self._volt_source_idle()
            volt = self.handlers['volt_source']
            volt.set_voltage(self.volt_source_voltage)
        
        # Calculate measurement delay
        self.suggested_delay = vna.config.calculate_delay()  
        
        # Calculate estimated measurement time
        total_estimated_time = self.suggested_delay * len(self.x) * len(self.y)
        minutes, seconds = divmod(total_estimated_time, 60)
        hours, minutes = divmod(minutes, 60)

        # Adjust rf_amplitude info if it's an array
        # Show only the first and last values and number of points
        if isinstance(self.rf_amplitude, (list, np.ndarray)):
            rf_info = f"RF Amplitude: {self.rf_amplitude[0]} to {self.rf_amplitude[-1]} dBm, {len(self.rf_amplitude)} points"
        else:
            rf_info = f"RF Amplitude: {self.rf_amplitude} dBm"
            
        # Update information string
        self.info = f"""
            Two-Tone Spectroscopy Measurement
            ----------------------------------
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Attenuation: {self.attenuation} dB
            VNA Power: {self.vna_amplitude} dBm
            RF Power: {rf_info}
            Cavity Frequency: {self.cavity_frequency/1e9:.6f} GHz
            RF Sweep: {self.x[0]/1e9:.6f} to {self.x[-1]/1e9:.6f} GHz, step {(self.x[1]-self.x[0])/1e6:.3f} MHz
            Data points: {len(self.x)}
            Delay per point: {self.suggested_delay:.3f} s
            Estimated time: {int(hours)}h {int(minutes)}m {int(seconds)}s
            """
        
        # Add notes to self.info if provided
        if self.notes:
            self.info += f"Notes: {self.notes}\n"

        if self.volt_source_voltage != 0:
            self.info += f"DC Voltage: {self.volt_source_voltage} V\n"
        
        self.show()
    
    def _vna_idle(self):
        """Set VNA to idle state."""
        vna = self.handlers['vna']
        if not vna.is_connected():
            vna.connect()
        vna.state = InstrumentState.IDLE
    
    def _rf_idle(self):
        """Set RF source to idle state."""
        rf = self.handlers['rf']
        if not rf.is_connected():
            rf.connect()
        rf.state = InstrumentState.IDLE
    
    def _volt_source_idle(self):
        """Set voltage source to idle state if available."""
        if 'volt_source' in self.handlers:
            volt = self.handlers['volt_source']
            if not volt.is_connected():
                volt.connect()
            volt.state = InstrumentState.IDLE

    def measure_point(self, rf_freq: float, att: float):
        """
        Measure a single point in the two-tone spectroscopy.
        
        Args:
            rf_freq: RF source frequency in Hz
            att: Attenuation value in dB (not used directly in this method)
            
        Returns:
            Tuple containing measurement data (magnitude, phase)
        """
        # Set RF source to the current frequency
        rf = self.handlers['rf']
        rf.set_frequency(rf_freq)
        
        # Allow time for the system to stabilize
        time.sleep(0.05)
        
        # Trigger a single VNA measurement
        vna = self.handlers['vna']
        
        # Validate the measurement configuration
        vna._validate_connection()
        vna.clear_averaging()
        
        # Wait for sweep completion with appropriate delay
        time.sleep(self.suggested_delay)
        
        # Get measurement data
        result = vna.get_data()
        
        # Handle case where get_data returns None
        if result is None:
            print(f"Warning: VNA returned None data at RF frequency {rf_freq/1e9:.6f} GHz")
            return 0, 0
        
        # Process result based on data format
        if isinstance(result, dict):
            # New handler format with named fields
            mag = np.mean(result.get('mags', [0]))  
            phase = np.mean(result.get('phases', [0]))  
            return mag, phase
        elif isinstance(result, tuple):
            # Legacy format with tuple of arrays
            if len(result) == 3:
                mags, phases, _ = result
                return np.mean(mags[0]), np.mean(phases[0])  # Return first point only
            elif len(result) == 2:
                mags, phases = result
                return np.mean(mags[0]), np.mean(phases[0])  # Return first point only

        # Default fallback for unexpected formats
        print(f"Warning: Unexpected data format from VNA: {type(result)}")
        return 0, 0

    def _set_y(self, rf_amplitude: float):
        """
        Set the RF amplitude value for the current measurement.

        Args:
            rf_amplitude: RF amplitude value in dB
        """
        rf = self.handlers['rf']
        rf.configure_measurement(
            rf_amplitude=rf_amplitude
        )

    def run(self):
        """
        Run the two-tone measurement by sweeping RF source frequency.
        
        This method sweeps the RF source frequency while keeping the VNA 
        frequency fixed at the cavity resonance. For each RF frequency, 
        it measures the cavity response which changes due to the qubit's 
        dispersive shift when the qubit is excited.
        """
        # Import required modules
        from IPython.display import clear_output, display
        import plotly.graph_objects as go
        import time
        import logging
        import traceback
        
        # Display measurement info
        self.show()
        
        total = len(self.y) * len(self.x)
        count = 0
        start_time = datetime.now()
        
        # Pre-allocate full arrays with NaN values for partial plotting
        # This ensures the plot always uses the same scale
        full_x_data = self.x / 1e9  # Convert to GHz for plotting
        
        try:
            # Power on RF source
            rf = self.handlers['rf']
            rf.power_on()
            vna = self.handlers['vna']
            vna.power_on()
            
            # Enable voltage source if needed
            if 'volt_source' in self.handlers and self.volt_source_voltage != 0:
                volt = self.handlers['volt_source']
                volt.set_voltage(self.volt_source_voltage)
            
            # Initialize color scale ranges for consistent visualization
            vmin = float('inf')
            vmax = float('-inf')
                
            for j, yv in enumerate(self.y):
                # Set RF amplitude value
                self._set_y(yv)
                
                # Initialize full data array with NaNs for progressive plotting
                full_y_data = np.full(len(self.x), np.nan)
                
                for i, xv in enumerate(self.x):
                    # Measure at current RF frequency
                    mag, phase = self.measure_point(xv, yv)
                    
                    # Store results in the main data arrays
                    self.data[j, i] = mag
                    self.phases[j, i] = phase
                    
                    # Update the plotting array with the new point
                    full_y_data[i] = mag
                    
                    # Update progress
                    count += 1
                    
                    # Calculate remaining time
                    elapsed = (datetime.now() - start_time).total_seconds()
                    points_per_second = count / elapsed if elapsed > 0 else 0
                    remaining_points = total - count
                    remaining_seconds = remaining_points / points_per_second if points_per_second > 0 else 0
                    remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)
                    remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
                    
                    # Progress bar
                    pct = count / total * 100
                    bar_len = 40
                    filled = int(round(bar_len * pct / 100))
                    bar = '#' * filled + '-' * (bar_len - filled)
                    
                    # Update the plot with current data at regular intervals
                    if (i > 0 and i % 2 == 0) or i == len(self.x) - 1:  # Update regularly or at end of row
                        # For single RF amplitude, show line plot updates
                        if len(self.y) == 1:
                            # Clear the output and show updated info and plot
                            clear_output(wait=True)
                            self.show()
                            print(f"Progress: [{bar}] {pct:.1f}% ({count}/{total})")
                            print(f"Estimated remaining time: {int(remaining_hours)}h {int(remaining_minutes)}m {int(remaining_seconds)}s")
                            
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
                                name=f"RF Power: {float(self.y[0])} dBm",
                                connectgaps=False  # Don't connect across NaN values
                            ))
                            
                            # Configure the layout to maintain the full range
                            fig.update_layout(
                                title=f"Two-Tone Spectroscopy (in progress: {count}/{total} points)",
                                xaxis_title="RF Frequency (GHz)",
                                yaxis_title="Magnitude (dB)",
                                xaxis=dict(
                                    range=[min(full_x_data), max(full_x_data)]
                                ),
                                yaxis=dict(
                                    range=[y_min, y_max]
                                ),
                                height=500,
                                width=800
                            )
                            display(fig)
                        # For multiple RF amplitudes, after completing a full row, show 3D plot update
                        elif i == len(self.x) - 1:
                            # Create a full-sized data array with NaN values for unmeasured points
                            # This ensures consistent color scale and axis limits
                            full_sized_data = np.full((len(self.y), len(self.x)), np.nan)
                            # Fill in only the measured values
                            full_sized_data[:j+1, :] = self.data[:j+1, :]
                            
                            # Clear previous output
                            clear_output(wait=True)
                            self.show()
                            print(f"Progress: [{bar}] {pct:.1f}% ({count}/{total})")
                            print(f"Completed RF amplitude {j+1}/{len(self.y)}: {float(yv)} dBm")
                            print(f"Estimated remaining time: {int(remaining_hours)}h {int(remaining_minutes)}m {int(remaining_seconds)}s")
                            
                            # Use Plot class for visualization of 3D data
                            progress_fig = Plot(
                                measure_name=self.MEASURE_NAME,
                                measurement_info=self.info,
                                dir_name=self.DIR_NAME,
                                xaxis_name="RF Frequency (GHz)",
                                yaxis_name="RF Amplitude (dBm)",
                                name_info=[f"Measured {j+1} of {len(self.y)} RF amplitudes"],
                                title=f"Two-Tone Spectroscopy",
                                data_x=self.x / 1e9,  # Convert to GHz
                                data_y=self.y,        # RF amplitudes
                                data_z=full_sized_data,  # Partial data with NaNs
                                zaxis_name="Magnitude (dB)"
                            ).plot_3d(save=False)  # Don't save intermediate plots
                            
                            # Show the 3D progress plot
                            display(progress_fig)
                    else:
                        # For iterations between plot updates, just print a compact status update
                        # without clearing the screen, so the plot remains visible
                        print(f"\rPoint {count}/{total} ({pct:.1f}%) - {int(remaining_minutes)}m {int(remaining_seconds)}s remaining", end="", flush=True)
            
            # Find maximum point (useful for qubit frequency identification)
            if np.any(~np.isnan(self.data)):
                if len(self.y) == 1:
                    max_idx = np.nanargmax(self.data[0, :])
                    self.max_freq = self.x[max_idx]
                    self.max_mag = self.data[0, max_idx]
                    print(f"\nMaximum response at RF frequency: {self.max_freq/1e9:.6f} GHz")
                else:
                    # For 3D data, find global maximum
                    max_indices = np.unravel_index(np.nanargmax(self.data), self.data.shape)
                    self.max_freq = self.x[max_indices[1]]
                    self.max_amp = float(self.y[max_indices[0]])
                    self.max_mag = float(self.data[max_indices])
                    print(f"\nMaximum response at RF frequency: {self.max_freq/1e9:.6f} GHz, RF amplitude: {self.max_amp} dBm")
            
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
            final_plot = self.plot()
            display(final_plot)
            
        except KeyboardInterrupt:
            print("\nMeasurement interrupted by user.")
        except Exception as e:
            logging.error(f"Error during two-tone measurement: {str(e)}")
            print(f"\nError during measurement: {e}")
            traceback.print_exc()
            raise
        finally:
            # Always clean up resources
            self.cleanup()
    
    def plot(self, save=True, y_info="magnitude"):
        """
        Plot the two-tone spectroscopy measurement results.
        
        Args:
            save: Whether to save the plot files
            y_info: Data to plot ("magnitude" or "phase")
            
        Returns:
            Plotly figure object
        """
        
        # Prepare data for plotting
        if len(self.y) == 1:
            # Single RF amplitude value: plot 1D line
            if y_info.lower() == "phase":
                plot_data = self.phases[0]
                y_label = "Phase (rad)"
            else:
                plot_data = self.data[0]
                y_label = "Magnitude (dB)"
            
            # Frequency in GHz for x-axis
            x_data = self.x / 1e9
            
            # Create plot object
            p = Plot(
                measure_name=self.MEASURE_NAME,
                measurement_info=self.info,
                dir_name=self.DIR_NAME,
                xaxis_name="RF Frequency (GHz)",
                yaxis_name=y_label,
                name_info=[f"att {self.attenuation} dB, RF amp {float(self.y[0])} dBm, cavity {self.cavity_frequency/1e9:.6f} GHz"],
                title="Two-Tone Spectroscopy",
                data_x=x_data,
                data_y=plot_data
            )
            
            # Generate and return plot
            fig = p.plot(save=save)
            
            # Add maximum point marker if available
            if hasattr(self, 'max_freq') and hasattr(self, 'max_mag') and y_info == "magnitude":
                fig.add_trace(
                    go.Scatter(
                        x=[self.max_freq/1e9],
                        y=[self.max_mag],
                        mode='markers+text',
                        text=[f"({self.max_freq/1e9:.6f} GHz, {self.max_mag:.2f} dB)"],
                        textposition="top center",
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='circle'
                        ),
                        showlegend=False
                    )
                )
            
            return fig
        else:
            # Multiple RF amplitude values: plot 2D heatmap
            if y_info.lower() == "phase":
                plot_data = self.phases
                z_label = "Phase (rad)"
            else:
                plot_data = self.data
                z_label = "Magnitude (dB)"
            
            # Calculate min/max values for info
            max_indices = np.unravel_index(np.nanargmax(plot_data), plot_data.shape)
            max_amp = float(self.y[max_indices[0]])
            max_freq = self.x[max_indices[1]]
            max_val = float(plot_data[max_indices])
            
            # Create name info for saving
            name_info = [
                f"att {self.attenuation} dB",
                f"cavity {self.cavity_frequency/1e9:.6f} GHz",
                f"max {y_info}: {max_val:.2f} at {max_freq/1e9:.6f} GHz, {max_amp} dBm"
            ]
            
            # Frequency in GHz for x-axis
            x_data = self.x / 1e9
            
            # Create plot object
            p = Plot(
                measure_name=self.MEASURE_NAME,
                measurement_info=self.info,
                dir_name=self.DIR_NAME,
                xaxis_name="RF Frequency (GHz)",
                yaxis_name="RF Amplitude (dBm)",
                zaxis_name=z_label,
                name_info=name_info,
                title="Two-Tone Spectroscopy",
                data_x=x_data,
                data_y=self.y,
                data_z=plot_data
            )
            
            # Generate and return 3D plot
            fig = p.plot_3d(save=save)
            return fig

    def save_data(self):
        """
        Save measurement data to NPZ file.
        
        The saved data includes frequency arrays, measurement data,
        and all relevant measurement parameters.
        """
        # Create output directory if it doesn't exist
        output_dir = f"{self.DIR_NAME}/data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with key parameters
        if len(self.y) == 1:
            # Single RF amplitude
            filename = f"{timestamp}_{self.MEASURE_NAME}_Att{self.attenuation}dB_RF{self.y[0]}dBm_CavityFreq{self.cavity_frequency:.2e}Hz.npz"
        else:
            # Multiple RF amplitudes - use range
            min_rf = np.min(self.y)
            max_rf = np.max(self.y)
            filename = f"{timestamp}_{self.MEASURE_NAME}_Att{self.attenuation}dB_RF{min_rf}to{max_rf}dBm_CavityFreq{self.cavity_frequency:.2e}Hz.npz"
        
        filepath = f"{output_dir}/{filename}"
        
        # Save data and metadata
        np.savez(
            filepath,
            # Arrays
            rf_frequencies=self.x,
            rf_amplitudes=self.y,
            magnitudes=self.data,
            phases=self.phases,
            # Measurement parameters
            attenuation=self.attenuation,
            vna_amplitude=self.vna_amplitude,
            vna_parameter=self.vna_parameter,
            cavity_frequency=self.cavity_frequency,
            if_bandwidth=self.if_bandwidth,
            data_array_format=self.data_array_format,
            average_points=self.average_points,
            vna_smoothing=self.vna_smoothing,
            volt_source_voltage=self.volt_source_voltage,
            # Header and info
            header=self.header,
            info=self.info,
            timestamp=timestamp
        )
        
        print(f"Data saved to {filepath}")
        return filepath

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
        files = cls.list_measurements()
        if not files or idx < 0 or idx >= len(files):
            raise IndexError("Invalid measurement index")
        
        return cls.plot_from_file(str(files[idx]), y_info=y_info, save=save)
        
    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot a saved two-tone measurement from file.
        
        Args:
            filepath: Path to the saved data file (.npz)
            y_info: Type of data to plot ("magnitude" or "phase")
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        try:
            # Load data from file
            data = np.load(filepath)
            
            # Extract required fields
            x = data['x'] if 'x' in data else data['rf_freqs']
            y = data['y'] if 'y' in data else data['attenuation']
            
            if y_info == "magnitude":
                z_data = data['data']
            elif y_info == "phase":
                z_data = data['phases'] if 'phases' in data else np.zeros_like(data['data'])
            else:
                raise ValueError(f"Unsupported y_info type: {y_info}")
            
            # Get metadata for plot
            info = str(data['info']) if 'info' in data else "Two-Tone Measurement"
            cavity_freq = float(data['cavity_frequency']) if 'cavity_frequency' in data else 0
            
            # Find maximum point
            if np.any(~np.isnan(z_data)):
                max_idx = np.nanargmax(z_data[0, :])
                max_freq = x[max_idx]
                max_val = z_data[0, max_idx]
            else:
                max_freq = 0
                max_val = 0
            
            # Determine if this is a 1D or 2D plot
            if len(y) <= 1:
                # 1D line plot
                name_info = [
                    f"Att: {y[0] if len(y) > 0 else 'N/A'} dB",
                    f"Cavity: {cavity_freq/1e9:.6f} GHz"
                ]
                
                # Create plot
                plot_obj = Plot(
                    measure_name=MEASURE_NAME,
                    measurement_info=info,
                    dir_name=DIR_NAME,
                    xaxis_name="RF Frequency (Hz)",
                    yaxis_name="Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)",
                    name_info=name_info,
                    title=f"{MEASURE_NAME} Measurement",
                    data_x=x,
                    data_y=z_data.flatten()
                )
                
                fig = plot_obj.plot(save=save)
                
                # Add maximum point marker
                fig.add_trace(
                    go.Scatter(
                        x=[max_freq],
                        y=[max_val],
                        mode='markers+text',
                        text=[f"({max_freq/1e9:.6f} GHz, {max_val:.2f} dB)"],
                        textposition="top center",
                        marker=dict(color='red', size=10, symbol='circle'),
                        showlegend=False
                    )
                )
                
            else:
                # 2D heatmap plot
                plot_obj = Plot(
                    measure_name=MEASURE_NAME,
                    measurement_info=info,
                    dir_name=DIR_NAME,
                    xaxis_name="RF Frequency (Hz)",
                    yaxis_name="Attenuation (dB)",
                    zaxis_name="Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)",
                    name_info=[f"Cavity: {cavity_freq/1e9:.6f} GHz"],
                    title=f"{MEASURE_NAME} Measurement",
                    data_x=x,
                    data_y=y,
                    data_z=z_data
                )
                
                fig = plot_obj.plot_3d(save=save)
            
            return fig
            
        except Exception as e:
            print(f"Error loading/plotting data from {filepath}: {e}")
            traceback.print_exc()
            return None
    
    @classmethod
    def double_plot(cls, x1, y1, x2, y2, labels=None, save=False):
        """
        Plot two sets of data on the same graph for comparison.
        
        Args:
            x1: X-axis data for first plot
            y1: Y-axis data for first plot
            x2: X-axis data for second plot
            y2: Y-axis data for second plot
            labels: Optional tuple of (title, xaxis, yaxis, trace1, trace2)
            save: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
       
        
        # Default labels
        if labels is None:
            labels = (
                "Two-Tone Comparison",
                "RF Frequency (Hz)",
                "Magnitude (dB)",
                "Measurement 1",
                "Measurement 2"
            )
        
        title, xaxis, yaxis, trace1_name, trace2_name = labels
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=x1, y=y1,
            mode='lines',
            name=trace1_name
        ))
        
        fig.add_trace(go.Scatter(
            x=x2, y=y2,
            mode='lines',
            name=trace2_name
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            legend_title="Measurements",
        )
        
        # Save if requested
        if save:
            output_dir_html = f"{DIR_NAME}/html"
            output_dir_pdf = f"{DIR_NAME}/pdf"
            
            os.makedirs(output_dir_html, exist_ok=True)
            os.makedirs(output_dir_pdf, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{MEASURE_NAME}_comparison"
            
            fig.write_html(f"{output_dir_html}/{filename}.html")
            fig.write_image(f"{output_dir_pdf}/{filename}.pdf")
        
        return fig
    
    def cleanup(self):
        """Clean up all instrument resources."""
        try:
            # Turn off all instruments
            self._vna_idle()
            self._rf_idle()
            
            if 'volt_source' in self.handlers and self.volt_source_voltage != 0:
                self._volt_source_idle()
                
            for name, handler in self.handlers.items():
                if hasattr(handler, 'cleanup'):
                    handler.cleanup()
                
            print("Measurement resources cleaned up.")
        
        except Exception as e:
            print(f"Error during cleanup: {e}")
