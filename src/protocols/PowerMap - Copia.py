"""
PowerMap Measurement Protocol

This module implements a unified interface for both 2D (PunchOut-style) and 3D (PowerSweep-style)
measurements using a Vector Network Analyzer (VNA) and an attenuator. The class automatically
determines the measurement type based on whether a single attenuation value or a range is provided.

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
import traceback

from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType, InstrumentState
from protocols.Plot import Plot

MEASURE_NAME = "PowerMap"
DIR_NAME = "powermap"

class PowerMapMeasurement:
    """
    Unified measurement class that combines PunchOut (2D) and PowerSweep (3D) functionality.
    
    This class automatically selects the appropriate measurement method based on the
    attenuation parameter:
    - If attenuation is a single value (int/float), a 2D measurement (PunchOut) is performed
    - If attenuation is a range/array, a 3D measurement (PowerSweep) is performed
    
    Features:
    - Automatic handling of VNA and attenuator instruments
    - Accurate measurement time estimation
    - Progress tracking during measurement
    - 2D line plots and 3D heatmap visualization of results
    - Optional phase data plotting
    - Automatic data saving for later analysis
    - Line plotting for specific attenuation values in 3D data
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME

    def __init__(self, header: str,
                 # Attenuation parameter (single value or range)
                 attenuation: Union[float, np.ndarray, List[float]],
                 # VNA parameters 
                 vna_amplitude: float,
                 vna_parameter: str,
                 start_frequency: float,
                 stop_frequency: float,
                 vna_n_points: int,
                 vna_if_bandwidth: float,
                 vna_data_array_format: str,
                 vna_average_points: int,
                 vna_smoothing: int = 0,
                 vna_averaging: int = 1,
                 # Voltage source (optional)
                 volt_source_voltage: float = 0,
                 # Handler instances (optional)
                 vna_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None,
                 voltage_source_handler: Optional[Handler] = None):
        """
        Initialize a PowerMap measurement with flexible attenuation parameter.
        
        Args:
            header: Measurement header/title
            attenuation: Single value (for 2D) or array/range (for 3D)
            vna_amplitude: VNA output power in dBm
            vna_parameter: VNA parameter to measure (e.g., 'S21')
            start_frequency: Start frequency in Hz
            stop_frequency: Stop frequency in Hz
            vna_n_points: Number of frequency points to measure
            vna_if_bandwidth: VNA IF bandwidth in Hz
            vna_data_array_format: VNA data format (e.g., 'MLOG')
            vna_average_points: Number of averages per point
            vna_smoothing: VNA smoothing factor (0 = off)
            vna_averaging: VNA averaging factor
            volt_source_voltage: Voltage source setting (0 = off)
            vna_handler: Handler instance for VNA
            attenuator_handler: Handler instance for attenuator
            voltage_source_handler: Handler instance for voltage source
        """
        self.header = header
        # Determine measurement mode based on attenuation parameter type
        if isinstance(attenuation, (int, float)):
            self.mode = "2D"
            self.y = np.array([float(attenuation)])
        else:
            self.mode = "3D"
            self.y = np.array(attenuation)
        
        # Store parameters
        self.vna_amplitude = vna_amplitude
        self.vna_parameter = vna_parameter
        self.start_frequency = start_frequency
        self.stop_frequency = stop_frequency
        self.vna_n_points = vna_n_points
        self.vna_if_bandwidth = vna_if_bandwidth
        self.vna_data_array_format = vna_data_array_format
        self.vna_average_points = vna_average_points
        self.vna_smoothing = vna_smoothing
        self.vna_averaging = vna_averaging
        self.volt_source_voltage = volt_source_voltage
        
        # Create frequency array (x-axis)
        self.x = np.linspace(start_frequency, stop_frequency, vna_n_points)
        
        # Initialize data arrays
        self.data = np.zeros((len(self.y), len(self.x)))
        self.phases = np.zeros((len(self.y), len(self.x)))
        
        # Set up handlers
        self.handlers = {}
        if vna_handler is not None:
            self.handlers['vna'] = vna_handler
        if attenuator_handler is not None:
            self.handlers['attenuator'] = attenuator_handler
        if voltage_source_handler is not None:
            self.handlers['voltage_source'] = voltage_source_handler
            
        # Prepare instruments and calculate delays
        self.prepare()
        
        # For tracking interruptions
        self._interrupted = False

    def prepare(self):
        """
        Prepare instruments and calculate appropriate delays.
        """
        # Validate handlers
        if 'vna' not in self.handlers:
            raise ValueError("VNA handler is required")
        if 'attenuator' not in self.handlers:
            raise ValueError("Attenuator handler is required")
            
        # Set VNA parameters and calculate suggested delay
        vna = self.handlers['vna']
        vna.set_sweep_points(self.vna_n_points)
        vna.set_start_frequency(self.start_frequency)
        vna.set_stop_frequency(self.stop_frequency)
        vna.set_if_bandwidth(self.vna_if_bandwidth)
        vna.set_power(self.vna_amplitude)
        vna.set_parameter(self.vna_parameter)
        vna.set_format(self.vna_data_array_format)
        vna.set_average_count(self.vna_average_points)
        vna.set_smoothing(self.vna_smoothing)
        
        # Calculate suggested delay based on VNA settings
        sweep_time = vna.get_sweep_time() if hasattr(vna, 'get_sweep_time') else 1.0
        averaging_factor = max(1, self.vna_averaging)
        suggested_delay = sweep_time * averaging_factor * 1.2  # Add 20% margin
        
        self.suggested_delay = suggested_delay
        
        # Create info string for display
        measurement_type = "2D (Single attenuation)" if self.mode == "2D" else "3D (Attenuation sweep)"
        
        attenuation_info = f"{self.y[0]:.1f} dB" if self.mode == "2D" else \
                           f"{self.y[0]:.1f} to {self.y[-1]:.1f} dB in {len(self.y)} steps"
        
        self.info = f"PowerMap {measurement_type} Measurement\n"
        self.info += f"Header: {self.header}\n"
        self.info += f"Attenuation: {attenuation_info}\n"
        self.info += f"Frequency range: {self.start_frequency/1e9:.6f} to {self.stop_frequency/1e9:.6f} GHz\n"
        self.info += f"VNA amplitude: {self.vna_amplitude} dBm\n"
        self.info += f"VNA parameter: {self.vna_parameter}\n"
        self.info += f"VNA IF bandwidth: {self.vna_if_bandwidth/1e3:.1f} kHz\n"
        self.info += f"VNA averaging: {self.vna_average_points} points\n"
        
        if self.mode == "3D":
            total_points = len(self.y) * len(self.x)
            total_time_estimate = self.suggested_delay * len(self.y)
            self.info += f"Total points: {total_points}\n"
            self.info += f"Estimated measurement time: {total_time_estimate:.1f} seconds "
            self.info += f"({total_time_estimate/60:.1f} minutes)\n"
        else:
            self.info += f"Points: {len(self.x)}\n"
            self.info += f"Estimated measurement time: {self.suggested_delay:.1f} seconds\n"
            
        if self.volt_source_voltage != 0:
            self.info += f"Voltage source: {self.volt_source_voltage} V\n"

    def run(self):
        """
        Run the measurement using the appropriate mode (2D or 3D).
        """
        if self.mode == "2D":
            return self.run_2d_mode()
        else:
            return self.run_3d_mode()

    def run_2d_mode(self):
        """
        Run a 2D measurement (PunchOut-style) with a single attenuation value.
        This method optimizes performance by using a single VNA sweep.
        """
        vna = self.handlers['vna']
        attenuator = self.handlers['attenuator']
        
        # Track if we've acquired any data
        data_acquired = False
        
        try:
            # Display measurement info
            self.show()
            
            # Enable voltage source if needed
            if self.volt_source_voltage != 0 and 'voltage_source' in self.handlers:
                volt_source = self.handlers['voltage_source']
                volt_source.set_voltage(self.volt_source_voltage)
            
            # Set attenuator
            attenuator.set_attenuation(self.y[0])
            
            # Configure VNA
            vna.set_sweep_points(self.vna_n_points)
            vna.set_start_frequency(self.start_frequency)
            vna.set_stop_frequency(self.stop_frequency)
            vna.set_if_bandwidth(self.vna_if_bandwidth)
            vna.set_power(self.vna_amplitude)
            vna.set_parameter(self.vna_parameter)
            vna.set_format(self.vna_data_array_format)
            vna.set_average_count(self.vna_average_points)
            vna.set_smoothing(self.vna_smoothing)
            
            # Power on VNA and prepare for measurement
            vna._safe_driver_call('power_on')
            vna._safe_driver_call('clear_averaging')
            
            # Wait for settling
            if hasattr(self, 'suggested_delay'):
                print(f"Waiting {self.suggested_delay:.1f} seconds for VNA to settle...")
                start_time = time.time()
                while time.time() - start_time < self.suggested_delay:
                    time.sleep(0.1)
                    # Check for keyboard interrupt
                    if self._interrupted:
                        raise KeyboardInterrupt
            
            # Get data from VNA in a single sweep
            print("Getting data from VNA...")
            magnitude, phase = vna.get_data(get_phase=True)
            
            if magnitude is None or phase is None:
                raise Exception("Failed to get data from VNA")
            
            # Flag that we've acquired data
            data_acquired = True
            
            # Store the data
            self.data[0, :] = magnitude
            self.phases[0, :] = phase
            
            # Save complete data
            self.save_data(partial=False)
            self.plot(save=True)
            
        except KeyboardInterrupt:
            self._interrupted = True
            print("\nMeasurement interrupted by user.")
            
            # Save partial data if we acquired any
            if data_acquired:
                print("Saving partial data...")
                self.save_data(partial=True)
                print("Plotting partial data...")
                self.plot(save=True)
                
        except Exception as e:
            print(f"Error during 2D measurement: {str(e)}")
            traceback.print_exc()
            
            # Still try to save partial data if we acquired any
            if data_acquired:
                print("Attempting to save partial data after error...")
                self.save_data(partial=True)
                
        finally:
            # Always clean up resources
            try:
                self.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            
            # Reset interrupt flag
            self._interrupted = False
            
        return None

    def run_3d_mode(self):
        """
        Run a 3D measurement (PowerSweep-style) with an attenuation range.
        This method loops through each attenuation value and measures frequency response.
        """
        vna = self.handlers['vna']
        attenuator = self.handlers['attenuator']
        
        try:
            # Display measurement info
            self.show()
            
            # Enable voltage source if needed
            if self.volt_source_voltage != 0 and 'voltage_source' in self.handlers:
                volt_source = self.handlers['voltage_source']
                volt_source.set_voltage(self.volt_source_voltage)
            
            # Power on VNA and prepare for sweep
            vna._safe_driver_call('power_on')
            
            # For tracking progress
            total_steps = len(self.y)
            start_time = time.time()
            completed = 0
            
            # Loop through attenuation values
            for i, att in enumerate(self.y):
                if self._interrupted:
                    raise KeyboardInterrupt
                
                # Update progress
                clear_output(wait=True)
                self.show()
                self._show_progress(completed, total_steps, start_time)
                
                # Set attenuator
                attenuator.set_attenuation(att)
                
                # Clear VNA averaging for new measurement
                vna._safe_driver_call('clear_averaging')
                
                # Wait for settling time
                if hasattr(self, 'suggested_delay'):
                    wait_start = time.time()
                    while time.time() - wait_start < self.suggested_delay:
                        time.sleep(0.1)
                        if self._interrupted:
                            raise KeyboardInterrupt
                
                # Get data from VNA
                magnitude, phase = vna.get_data(get_phase=True)
                
                if magnitude is None or phase is None:
                    print(f"Warning: Failed to get data for attenuation {att} dB")
                    continue
                
                # Store the data
                self.data[i, :] = magnitude
                self.phases[i, :] = phase
                
                # Increment completed counter
                completed += 1
            
            # Final update
            clear_output(wait=True)
            self.show()
            self._show_progress(completed, total_steps, start_time)
            
            # Save complete data
            self.save_data(partial=False)
            self.plot(save=True)
            
        except KeyboardInterrupt:
            self._interrupted = True
            print("\nMeasurement interrupted by user.")
            
            # Save partial data
            if completed > 0:
                print(f"Saving partial data ({completed}/{total_steps} steps completed)...")
                # Trim data arrays to completed measurements
                self.data = self.data[:completed, :]
                self.phases = self.phases[:completed, :]
                self.y = self.y[:completed]
                self.save_data(partial=True)
                print("Plotting partial data...")
                self.plot(save=True)
                
        except Exception as e:
            print(f"Error during 3D measurement: {str(e)}")
            traceback.print_exc()
            
            # Try to save partial data
            if completed > 0:
                print(f"Attempting to save partial data ({completed}/{total_steps} steps completed)...")
                self.data = self.data[:completed, :]
                self.phases = self.phases[:completed, :]
                self.y = self.y[:completed]
                self.save_data(partial=True)
                
        finally:
            # Always clean up resources
            try:
                self.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
            
            # Reset interrupt flag
            self._interrupted = False
            
        return None

    def _show_progress(self, completed, total, start_time=None):
        """
        Display a progress bar and estimated time remaining.
        
        Args:
            completed: Number of completed steps
            total: Total number of steps
            start_time: Start time for calculating ETA (optional)
        """
        pct = 100 * completed / total if total > 0 else 0
        bar_len = 40
        filled_len = int(bar_len * completed // total)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        
        if start_time is not None and completed > 0:
            elapsed = time.time() - start_time
            rate = elapsed / completed
            remaining = rate * (total - completed)
            
            if remaining > 3600:
                eta = f"{remaining/3600:.1f} hours"
            elif remaining > 60:
                eta = f"{remaining/60:.1f} minutes"
            else:
                eta = f"{remaining:.1f} seconds"
                
            print(f"Progress: |{bar}| {pct:.1f}% ({completed}/{total}) ETA: {eta}")
        else:
            print(f"Progress: |{bar}| {pct:.1f}% ({completed}/{total})")

    def show(self):
        """
        Display the measurement information string.
        """
        print(self.info)

    def save_data(self, partial=False):
        """
        Save measurement data to a numpy .npz file.
        
        Args:
            partial: Whether this is a partial save due to interruption
        
        Returns:
            Path to the saved file
        """
        out_dir = Path(self.DIR_NAME) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include partial indicator in filename if applicable
        partial_tag = "_partial" if partial else ""
        
        # Include mode and measurement parameters in filename
        if self.mode == "2D":
            filename_parts = [
                ts,
                self.MEASURE_NAME,
                f"Att{self.y[0]:.0f}dB",
                f"MaxFreq{max(self.x):.2e}Hz",
                partial_tag
            ]
        else:
            filename_parts = [
                ts,
                self.MEASURE_NAME,
                f"Att{self.y[0]:.0f}to{self.y[-1]:.0f}dB",
                f"MaxFreq{max(self.x):.2e}Hz",
                partial_tag
            ]
        
        fname = "_".join([p for p in filename_parts if p])
        path = out_dir / f"{fname}.npz"
        
        # Save all relevant data
        data_dict = {
            'header': self.header,
            'info': self.info,
            'x': self.x,
            'y': self.y,
            'data': self.data,
            'phases': self.phases,
            'mode': self.mode,
            'is_partial': partial,
            'timestamp': ts
        }
        
        np.savez(path, **data_dict)
        
        status = "Partial" if partial else "Complete"
        print(f"{status} data saved to {path}")
        return path

    def plot(self, save=True, y_info="magnitude"):
        """
        Plot the measurement data.
        
        Args:
            save: Whether to save the plot to HTML and PDF
            y_info: What to plot - "magnitude" or "phase"
            
        Returns:
            Plotly figure object
        """
        if self.mode == "2D":
            return self._plot_2d(save, y_info)
        else:
            return self._plot_3d(save, y_info)

    def _plot_2d(self, save=True, y_info="magnitude"):
        """
        Create a 2D line plot for single attenuation measurements.
        
        Args:
            save: Whether to save the plot
            y_info: "magnitude" or "phase"
            
        Returns:
            Plotly figure object
        """
        # Convert frequencies to GHz for display
        x_ghz = self.x / 1e9
        
        # Choose data based on y_info
        if y_info.lower() == "phase":
            y_data = self.phases[0, :]
            y_label = "Phase (degrees)"
        else:
            y_data = self.data[0, :]
            y_label = "Magnitude (dB)"
        
        # Prepare plot title and info
        attenuation = self.y[0]
        title = f"{self.MEASURE_NAME} at {attenuation:.1f} dB"
        
        # Create name info for Plot class
        name_info = [
            f"att: {attenuation:.1f} dB",
            f"max_{y_info.lower()}: {np.max(y_data):.2f} at {self.x[np.argmax(y_data)]/1e9:.2e} GHz"
        ]
        
        # Create Plot object
        plotter = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name="Frequency (GHz)",
            yaxis_name=y_label,
            name_info=name_info,
            title=title,
            data_x=x_ghz,
            data_y=y_data
        )
        
        # Generate the plot
        fig = plotter.plot(save=save)
        return fig

    def _plot_3d(self, save=True, y_info="magnitude"):
        """
        Create a 3D heatmap for attenuation sweep measurements.
        
        Args:
            save: Whether to save the plot
            y_info: "magnitude" or "phase"
            
        Returns:
            Plotly figure object
        """
        # Convert frequencies to GHz for display
        x_ghz = self.x / 1e9
        
        # Choose data based on y_info
        if y_info.lower() == "phase":
            z_data = self.phases
            z_label = "Phase (degrees)"
        else:
            z_data = self.data
            z_label = "Magnitude (dB)"
        
        # Prepare plot title and info
        title = f"{self.MEASURE_NAME} 3D Sweep"
        
        # Create name info for Plot class
        name_info = [
            f"att: {self.y[0]:.1f}~{self.y[-1]:.1f} dB",
            f"max_{y_info.lower()}: {np.max(z_data):.2f} at {self.x[np.argmax(z_data)]/1e9:.2e} GHz",
            f"{self.y[0]:.1f} dB"
        ]
        
        # Create Plot object
        plotter = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name="Frequency (GHz)",
            yaxis_name="Attenuation (dB)",
            zaxis_name=z_label,
            name_info=name_info,
            title=title,
            data_x=x_ghz,
            data_y=self.y,
            data_z=z_data
        )
        
        # Generate the plot
        fig = plotter.plot_3d(save=save)
        return fig
        
    def plot_lines(self, attenuation_values, save=True, y_info="magnitude"):
        """
        Plot specific attenuation lines from a 3D measurement.
        
        Args:
            attenuation_values: List of attenuation values to plot
            save: Whether to save the plot
            y_info: "magnitude" or "phase"
            
        Returns:
            Plotly figure object
        """
        if self.mode != "3D":
            print("Line plotting is only available for 3D measurements.")
            return None
            
        # Convert frequencies to GHz for display
        x_ghz = self.x / 1e9
        
        # Choose data based on y_info
        if y_info.lower() == "phase":
            z_data = self.phases
            y_label = "Phase (degrees)"
        else:
            z_data = self.data
            y_label = "Magnitude (dB)"
            
        # Filter for requested attenuation values
        attenuation_indices = []
        valid_attenuations = []
        
        for att in attenuation_values:
            # Find closest attenuation value
            idx = np.abs(self.y - att).argmin()
            actual_att = self.y[idx]
            
            # Check if it's close enough (within 0.1 dB)
            if abs(actual_att - att) <= 0.1:
                attenuation_indices.append(idx)
                valid_attenuations.append(actual_att)
            else:
                print(f"Warning: Attenuation {att} dB not found in data. Closest value: {actual_att} dB")
                
        if not attenuation_indices:
            print("No valid attenuation values found in data.")
            return None
            
        # Create plot
        fig = go.Figure()
        
        # Add each line
        for idx, att in zip(attenuation_indices, valid_attenuations):
            fig.add_trace(go.Scatter(
                x=x_ghz,
                y=z_data[idx, :],
                mode='lines',
                name=f"{att:.1f} dB"
            ))
            
        # Update layout
        fig.update_layout(
            title=f"{self.MEASURE_NAME} - Selected Attenuation Lines",
            xaxis_title="Frequency (GHz)",
            yaxis_title=y_label,
            legend_title="Attenuation",
            width=800,
            height=600
        )
        
        # Save if requested
        if save:
            # Create output directories
            output_dir_html = f"{self.DIR_NAME}/lines/html"
            output_dir_pdf = f"{self.DIR_NAME}/lines/pdf"
            output_dir_data = f"{self.DIR_NAME}/lines/data"
            
            os.makedirs(output_dir_html, exist_ok=True)
            os.makedirs(output_dir_pdf, exist_ok=True)
            os.makedirs(output_dir_data, exist_ok=True)
            
            # Generate timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with attenuation values
            att_values_str = "_".join([f"{att:.0f}" for att in valid_attenuations])
            base_filename = f"{ts}_{self.MEASURE_NAME}_attenuation_lines_{att_values_str}"
            
            # Save plot
            html_path = f"{output_dir_html}/{base_filename}.html"
            pdf_path = f"{output_dir_pdf}/{base_filename}.pdf"
            fig.write_html(html_path)
            fig.write_image(pdf_path)
            
            # Save data
            data_path = f"{output_dir_data}/{base_filename}.npz"
            np.savez(data_path,
                    x=self.x,
                    attenuation_values=np.array(valid_attenuations),
                    data=z_data[attenuation_indices, :],
                    info=self.info,
                    y_info=y_info)
            
            print(f"Line plot saved to {html_path} and {pdf_path}")
            print(f"Line data saved to {data_path}")
            
        return fig

    def cleanup(self):
        """
        Clean up resources and restore instruments to idle state.
        """
        print("Cleaning up instruments...")
        
        try:
            # Return VNA to idle state
            if 'vna' in self.handlers:
                vna = self.handlers['vna']
                if hasattr(vna, '_safe_driver_call'):
                    vna._safe_driver_call('power_off')
            
            # Return voltage source to 0V if used
            if self.volt_source_voltage != 0 and 'voltage_source' in self.handlers:
                volt_source = self.handlers['voltage_source']
                volt_source.set_voltage(0)
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    @classmethod
    def list_measurements(cls):
        """
        List all saved measurements, showing the most recent first.
        
        Returns:
            List of saved measurement file paths
        """
        # Check for data directory
        data_dir = Path(cls.DIR_NAME) / "data"
        if not data_dir.exists():
            print(f"No measurements found in {data_dir}")
            return []
            
        # Find all .npz files
        npz_files = list(data_dir.glob("*.npz"))
        if not npz_files:
            print(f"No measurements found in {data_dir}")
            return []
            
        # Sort by modification time (newest first)
        npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Display measurements
        print(f"Found {len(npz_files)} measurements:")
        for i, file in enumerate(npz_files):
            try:
                # Load header info
                data = np.load(file, allow_pickle=True)
                header = str(data.get('header', "No header"))
                timestamp = file.name.split('_')[0:2]
                timestamp_str = '_'.join(timestamp)
                
                # Check if it's a 2D or 3D measurement
                mode = str(data.get('mode', "Unknown"))
                
                # Get attenuation info
                if 'y' in data:
                    y = data['y']
                    if len(y) == 1:
                        att_info = f"{y[0]:.1f} dB"
                    else:
                        att_info = f"{y[0]:.1f} to {y[-1]:.1f} dB ({len(y)} values)"
                else:
                    att_info = "Unknown"
                    
                print(f"[{i}] {timestamp_str} - {mode} measurement - {att_info} - {header}")
            except Exception as e:
                print(f"[{i}] {file.name} (Error: {str(e)})")
                
        return npz_files

    @classmethod
    def plot_measurement(cls, idx, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement by index.
        
        Args:
            idx: Index from list_measurements()
            y_info: "magnitude" or "phase"
            save: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        # Get list of measurement files
        data_dir = Path(cls.DIR_NAME) / "data"
        if not data_dir.exists():
            print(f"No measurements found in {data_dir}")
            return None
            
        npz_files = list(data_dir.glob("*.npz"))
        if not npz_files:
            print(f"No measurements found in {data_dir}")
            return None
            
        # Sort by modification time (newest first)
        npz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if idx < 0 or idx >= len(npz_files):
            print(f"Invalid index {idx}. Valid range is 0-{len(npz_files)-1}")
            return None
            
        # Load the requested measurement
        return cls.plot_from_file(npz_files[idx], y_info=y_info, save=save)

    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement from a file path.
        
        Args:
            filepath: Path to the .npz file
            y_info: "magnitude" or "phase"
            save: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        # Load the data
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
            
        try:
            data = np.load(filepath, allow_pickle=True)
            
            # Extract required data
            header = str(data.get('header', "No header"))
            info = str(data.get('info', "No info"))
            x = data['x']
            y = data['y']
            
            # Choose data based on y_info
            if y_info.lower() == "phase" and 'phases' in data:
                z_data = data['phases']
                z_label = "Phase (degrees)"
            else:
                z_data = data['data']
                z_label = "Magnitude (dB)"
                
            # Create instance and load data
            instance = cls(header=header, attenuation=y, vna_amplitude=0,
                          vna_parameter="", start_frequency=x[0], stop_frequency=x[-1],
                          vna_n_points=len(x), vna_if_bandwidth=0, vna_data_array_format="",
                          vna_average_points=1)
                          
            instance.x = x
            instance.y = y
            instance.info = info
            instance.data = z_data
            
            if 'phases' in data:
                instance.phases = data['phases']
                
            # Set mode based on data
            instance.mode = "2D" if len(y) == 1 else "3D"
            
            # Plot the data
            return instance.plot(save=save, y_info=y_info)
            
        except Exception as e:
            print(f"Error loading measurement: {str(e)}")
            traceback.print_exc()
            return None
            
    @classmethod
    def double_plot(cls, x1, y1, x2, y2, labels=None, save=False):
        """
        Create a comparison plot of two measurements.
        
        Args:
            x1: x-data for first plot
            y1: y-data for first plot
            x2: x-data for second plot
            y2: y-data for second plot
            labels: List of two labels for the legend
            save: Whether to save the plot
            
        Returns:
            Plotly figure object
        """
        if labels is None:
            labels = ["Measurement 1", "Measurement 2"]
            
        # Convert to GHz if needed
        if np.mean(x1) > 1e6:
            x1 = x1 / 1e9
        if np.mean(x2) > 1e6:
            x2 = x2 / 1e9
            
        # Create plot
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=x1,
            y=y1,
            mode='lines',
            name=labels[0]
        ))
        
        fig.add_trace(go.Scatter(
            x=x2,
            y=y2,
            mode='lines',
            name=labels[1]
        ))
        
        # Update layout
        fig.update_layout(
            title="PowerMap Comparison",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Magnitude (dB)",
            legend_title="Measurements",
            width=800,
            height=600
        )
        
        # Save if requested
        if save:
            # Create output directories
            output_dir_html = f"{cls.DIR_NAME}/html"
            output_dir_pdf = f"{cls.DIR_NAME}/pdf"
            
            os.makedirs(output_dir_html, exist_ok=True)
            os.makedirs(output_dir_pdf, exist_ok=True)
            
            # Generate timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            base_filename = f"{ts}_{cls.MEASURE_NAME}_comparison"
            
            # Save plot
            html_path = f"{output_dir_html}/{base_filename}.html"
            pdf_path = f"{output_dir_pdf}/{base_filename}.pdf"
            fig.write_html(html_path)
            fig.write_image(pdf_path)
            
            print(f"Comparison plot saved to {html_path} and {pdf_path}")
            
        return fig
