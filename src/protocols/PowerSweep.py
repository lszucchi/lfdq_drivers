"""
PowerSweep Measurement Protocol

This module implements VNA-based power sweep measurements using the standardized Handler framework
and BaseMeasurement class. It combines VNA, attenuator, and optional voltage source
to perform cavity power sweep measurements with varying attenuation.

Author: Measurement System
Date: June 2025
"""

import os
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List, Union

from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

MEASURE_NAME = "PowerSweep"
DIR_NAME = "powersweep"

class PowerSweepMeasurement(BaseMeasurement):
    """
    Power sweep measurement using a VNA: sweeps attenuation vs frequency.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME

    def __init__(self, header: str,
                 # Fixed parameters for single measurement (y-axis has length 1)
                 attenuation_range: np.ndarray,
                 # VNA parameters
                 vna_amplitude: float,
                 vna_parameter: str,
                 start_frequency: float,
                 stop_frequency: float,
                 vna_n_points: int,
                 vna_if_bandwidth: float,
                 vna_data_array_format: str = "MLOGarithmic",
                 vna_average_points: int = 64,
                 vna_averaging: int = 1,
                 vna_smoothing: int = 0,
                 # Voltage source (optional)
                 volt_source_voltage: float = 0,
                 # Handler instances (optional)
                 vna_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None,
                 voltage_source_handler: Optional[Handler] = None,
                 notes: Optional[str] = None):
        
        self.notes = notes
        self._interrupted = False
        # Generate frequency array for BaseMeasurement
        freq_array = np.linspace(start_frequency, stop_frequency, vna_n_points)
        
        # Initialize handlers
        handlers = {
            'vna': vna_handler or Handler(InstrumentType.VNA, "TCPIP0::169.254.101.105::inst0::INSTR"),
            'attenuator': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        
        # Voltage source (optional)
        self.volt_source_voltage = volt_source_voltage
        if self.volt_source_voltage != 0:
            handlers['volt_source'] = voltage_source_handler or Handler(
                InstrumentType.VOLTAGE_SOURCE, "GPIB0::2::INSTR", port=8, step_voltage=0.001
            )
        
        # Store VNA parameters
        self.vna_amplitude = vna_amplitude
        self.vna_parameter = vna_parameter
        self.vna_if_bandwidth = vna_if_bandwidth
        self.vna_data_array_format = vna_data_array_format
        self.vna_average_points = vna_average_points
        self.vna_smoothing = vna_smoothing
        self.vna_averaging = vna_averaging
        
        # Data storage for phases (magnitudes stored in self.data)
        self.phases = np.zeros((len(attenuation_range), vna_n_points))
        
        # Initialize BaseMeasurement with attenuation range as y-axis
        super().__init__(header, freq_array, attenuation_range, handlers)

    def prepare(self):
        """Prepare all instruments for measurement."""
        vna = self.handlers['vna']
        attenuator = self.handlers['attenuator']
        
        # Set VNA to idle first
        self._vna_idle()
        
        # Set attenuator to first value in range
        attenuator.set_attenuation(self.y[0])

        self.center_frequency = (self.x[0] + self.x[-1]) / 2
        self.span_frequency = self.x[-1] - self.x[0]
        
        # Configure VNA using Handler's configure_measurement method
        vna.configure_measurement(
            amplitude=self.vna_amplitude,
            parameter=self.vna_parameter,
            start_frequency=self.x[0],  # First frequency
            stop_frequency=self.x[-1],  # Last frequency
            center_frequency=self.center_frequency,
            span_frequency=self.span_frequency,
            sweep_points=len(self.x),
            if_bandwidth=self.vna_if_bandwidth,
            data_array_format=self.vna_data_array_format,
            average_points=self.vna_average_points,
            smoothing=self.vna_smoothing,
            averaging_enabled=self.vna_averaging
        )

        self.suggested_delay = vna.config.calculate_delay()
        
        # Calculate estimated measurement time
        # For each attenuation value, we wait for the suggested delay (one measurement per attenuation)
        total_estimated_time = self.suggested_delay * len(self.y)
        minutes, seconds = divmod(total_estimated_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        # Set voltage source to idle if needed
        if 'volt_source' in self.handlers:
            self._volt_source_idle()
        
        # Update information string
        self.info = f"""
        POWER SWEEP MEASUREMENT
        ----------------------------------
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Attenuation range: {self.y[0]} to {self.y[-1]} dB ({len(self.y)} pts)
        VNA Amplitude: {self.vna_amplitude} dBm
        VNA Parameter: {vna.config.parameter}
        VNA Center frequency: {vna.config.center_frequency/1e9} GHz
        VNA Span: {vna.config.span_frequency/1e6} MHz
        VNA IF bandwidth: {vna.config.if_bandwidth:.2e} Hz
        VNA Sweep points: {vna.config.sweep_points}
        VNA Data array format: {vna.config.data_array_format}
        VNA Smoothing: {vna.config.smoothing}
        Voltage source: {self.volt_source_voltage} V
        Suggested delay per sweep: {self.suggested_delay:.2f} s
        Estimated total time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}
        """

         # Add notes to self.info if provided
        if self.notes:
            self.info += f"Notes: {self.notes}\n"
        
        self.show()

    def measure_point(self, freq: float, att_val: float):
        """
        Measure a single frequency point at given attenuation.
        
        Args:
            freq: Frequency to measure
            att_val: Attenuation value
            
        Returns:
            float: Magnitude at this frequency and attenuation
        """
        vna = self.handlers['vna']
        attenuator = self.handlers['attenuator']
        
        # Set attenuation
        attenuator.set_attenuation(att_val)
        
        # Power on VNA and set frequency
        vna.power_on()
        vna.set_frequency(freq)
        
        # Perform single point measurement
        result = vna.get_data()
        
        if result is None or 'mags' not in result:
            print(f"Warning: No data for freq={freq}, att={att_val}")
            return 0
            
        # Find closest frequency in result
        freqs = result.get('freqs', np.array([freq]))
        mags = result.get('mags', np.zeros_like(freqs))
        phases = result.get('phases', np.zeros_like(freqs))
        
        # Find index of this frequency in the array
        freq_idx = np.argmin(np.abs(freqs - freq))
        
        # Store phase data
        att_idx = np.where(self.y == att_val)[0][0]
        self.phases[att_idx, np.where(self.x == freq)[0][0]] = phases[freq_idx]
        
        # Return magnitude
        return mags[freq_idx]

    def _set_y(self, att_val: float):
        """
        Called per attenuation row in the BaseMeasurement run loop.
        Sets the attenuator to the specified value.
        
        Args:
            att_val: Attenuation value in dB
        """
        # called per attenuation row
        self.handlers['attenuator'].set_attenuation(att_val)

    def plot(self, save=True, y_info="magnitude"):
        """
        Plot measurement results as a 3D heatmap.
        
        Args:
            save: Whether to save the plot. Only the final, complete plot should be saved.
            y_info: Type of data to plot ("magnitude" or "phase")
            
        Returns:
            Plotly figure object
        """
        # Prepare plot data
        plot_data = self.data if y_info == "magnitude" else self.phases
        
        # Determine axis labels
        xaxis_name = "Frequency (Hz)"
        yaxis_name = "Attenuation (dB)"
        zaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)"
        
        # Calculate min/max values for info
        max_indices = np.unravel_index(np.argmax(plot_data), plot_data.shape)
        max_att = self.y[max_indices[0]]
        max_freq = self.x[max_indices[1]]
        max_val = plot_data[max_indices]
        
        # Create name info for saving
        name_info = [
            f"Att: {self.y[0]}~{self.y[-1]} dB",
            f"Max {y_info}: {max_val:.2f} at {max_freq:.2e} Hz, {max_att} dB"
        ]
        
        # Create plot using Plot class
        fig = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name=xaxis_name,
            yaxis_name=yaxis_name,
            name_info=name_info,
            title=f"{self.MEASURE_NAME} Measurement",
            data_x=self.x,
            data_y=self.y,
            data_z=plot_data,
            zaxis_name=zaxis_name
        ).plot_3d(save=save)
        
        fig.show()
        return fig

    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, data, header, and info.
        Override to ensure consistent format for plot_from_file functionality.
        """
        out_dir = Path(self.DIR_NAME) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filename
        max_att = self.y[-1]
        max_freq = self.x[-1]
        fname = f"{ts}_{self.MEASURE_NAME}_Att{int(self.y[0])}dB_MaxFreq{max_freq:.2e}Hz.npz"
        path = out_dir / fname
        
        # Save all data
        np.savez(path,
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
        files = cls.list_measurements()
        if not files or idx < 0 or idx >= len(files):
            raise IndexError("Invalid measurement index")
        
        return cls.plot_from_file(str(files[idx]), y_info=y_info, save=save)
        
    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot measurement data from a file.
        
        Args:
            filepath: Path to the saved data file (.npz or .npy)
            y_info: Type of data to plot ("magnitude" or "phase")
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        try:
            # Load data from .npz file
            data = np.load(filepath, allow_pickle=True)
            
            # Extract measurement parameters
            freqs = data.get('freqs', data.get('x'))
            attenuation = data.get('attenuation', data.get('y'))
            
            if y_info == "magnitude":
                plot_data = data.get('mags', data.get('data'))
            elif y_info == "phase":
                plot_data = data.get('phases', np.zeros_like(data.get('data')))
            else:
                plot_data = data.get('data')
            
            info = str(data.get('info', ''))
            header = str(data.get('header', ''))
            
            # Get measure name or use class name
            measure_name = data.get('measure_name', cls.MEASURE_NAME)
            
            # Calculate min/max values for info
            max_indices = np.unravel_index(np.argmax(plot_data), plot_data.shape)
            max_att = attenuation[max_indices[0]]
            max_freq = freqs[max_indices[1]]
            max_val = plot_data[max_indices]
            
            # Create name info for plot
            name_info = [
                f"Att: {attenuation[0]}~{attenuation[-1]} dB",
                f"Max {y_info}: {max_val:.2f} at {max_freq:.2e} Hz, {max_att} dB"
            ]
            
            # Determine axis labels
            xaxis_name = "Frequency (Hz)"
            yaxis_name = "Attenuation (dB)"
            zaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)"
            
            # Create plot
            fig = Plot(
                measure_name=measure_name,
                measurement_info=info,
                dir_name=cls.DIR_NAME,
                xaxis_name=xaxis_name,
                yaxis_name=yaxis_name,
                name_info=name_info,
                title=f"{measure_name} Measurement from File",
                data_x=freqs,
                data_y=attenuation,
                data_z=plot_data,
                zaxis_name=zaxis_name
            ).plot_3d(save=save)
            
            return None
            
        except Exception as e:
            print(f"Error loading/plotting data: {e}")
            raise

    def cleanup(self):
        """
        Perform cleanup of all instruments after measurement.
        """
        # Clean up VNA
        if 'vna' in self.handlers:
            self._vna_idle()
            
        # Clean up voltage source
        if 'volt_source' in self.handlers:
            self._volt_source_idle()

        # Set attenuator to minimum attenuation
        if 'attenuator' in self.handlers:
            try:
                self.handlers['attenuator'].set_attenuation(0)
            except:
                print("Warning: Failed to reset attenuator to 0 dB")
        
        # Call cleanup methods on all handlers if they exist
        for name, handler in self.handlers.items():
            if hasattr(handler, 'cleanup'):
                handler.cleanup()
        
        print("Measurement resources cleaned up.")

    def _vna_idle(self):
        """Set VNA to idle mode."""
        vna = self.handlers['vna']
        vna.power_off()
        vna.clear_averaging()
    
    def _volt_source_idle(self):
        """Set voltage source to idle mode."""
        if 'volt_source' in self.handlers:
            self.handlers['volt_source'].set_idle()
            
    def _show_progress(self, completed_points, total_points, start_time=None):
        """
        Display progress information including completion percentage and remaining time.
        
        Args:
            completed_points: Number of points measured so far
            total_points: Total number of points in the measurement
            start_time: Optional datetime object when measurement started
        """
        # Calculate percentage complete
        percent_complete = (completed_points / total_points) * 100
        
        # Display basic progress
        progress_msg = f"Progress: {completed_points}/{total_points} points ({percent_complete:.1f}%)"
        
        # Add time information if start_time is provided
        if start_time is not None:
            # Calculate elapsed time
            elapsed = datetime.now() - start_time
            elapsed_seconds = elapsed.total_seconds()
            
            # Only calculate remaining time if we have made some progress
            if completed_points > 0 and elapsed_seconds > 0:
                # Calculate points per second
                points_per_second = completed_points / elapsed_seconds
                
                # Calculate remaining time
                remaining_points = total_points - completed_points
                if points_per_second > 0:
                    remaining_seconds = remaining_points / points_per_second
                    remaining_time = timedelta(seconds=int(remaining_seconds))
                    
                    # Format elapsed and remaining times
                    elapsed_str = self._format_time_delta(elapsed)
                    remaining_str = self._format_time_delta(remaining_time)
                    
                    # Add time information to progress message
                    progress_msg += f"\nElapsed: {elapsed_str}, Remaining: {remaining_str}"
        
        print(progress_msg)
    
    def _format_time_delta(self, delta):
        """
        Format a timedelta object into a readable string.
        
        Args:
            delta: A timedelta object
            
        Returns:
            String in format "HH:MM:SS" or "DD days, HH:MM:SS" for longer durations
        """
        total_seconds = int(delta.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
    def plot_lines(self, attenuation_values, save=True, y_info="magnitude"):
        """
        Plot individual frequency sweeps for selected attenuation values.
        
        Args:
            attenuation_values: List of attenuation values to plot
            save: Whether to save the plot
            y_info: Type of data to plot ("magnitude" or "phase")
            
        Returns:
            Plotly figure object
        """
        # Validate attenuation values
        valid_atts = []
        for att in attenuation_values:
            if att in self.y:
                valid_atts.append(att)
            else:
                print(f"Warning: Attenuation value {att} dB not found in measurement data")
        
        if not valid_atts:
            raise ValueError("No valid attenuation values provided")
            
        # Prepare plot data
        plot_data = self.data if y_info == "magnitude" else self.phases
        
        # Determine axis labels
        xaxis_name = "Frequency (Hz)"
        yaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)"
        
        # Create line data for plotting
        data_y_lines = []
        
        # Get data for each attenuation value
        for att in valid_atts:
            idx = np.where(self.y == att)[0][0]
            data_y_lines.append(plot_data[idx])
        
        # Convert list to dictionary with proper naming for plotly
        data_dict = {}
        for i, att in enumerate(valid_atts):
            data_dict[f"{att} dB"] = data_y_lines[i]
        
        # Create name info for saving
        att_str = ", ".join([f"{att}" for att in valid_atts])
        name_info = [
            f"Attenuation: {att_str} dB",
        ]
        
        # Create plot using Plot class
        from IPython.display import clear_output
        clear_output(wait=True)
        
        # Create a figure with plotly graph_objects for more control
        fig = go.Figure()
        
        # Add a trace for each attenuation value
        for i, att in enumerate(valid_atts):
            idx = np.where(self.y == att)[0][0]
            fig.add_trace(
                go.Scatter(
                    x=self.x,
                    y=plot_data[idx],
                    mode='lines',
                    name=f"{att} dB"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{self.MEASURE_NAME} - Selected Attenuation Lines",
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            width=800,
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # If save is true, use the Plot class to save the figure
        if save:
            # Convert attenuation values to string for filename
            att_str_filename = "_".join([f"{att}" for att in valid_atts])
            
            # Generate timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create dedicated "lines" subdirectories for HTML, PDF, and data
            output_dir_html = f"{self.DIR_NAME}/lines/html"
            output_dir_pdf = f"{self.DIR_NAME}/lines/pdf"
            output_dir_data = f"{self.DIR_NAME}/lines/data"
            os.makedirs(output_dir_html, exist_ok=True)
            os.makedirs(output_dir_pdf, exist_ok=True)
            os.makedirs(output_dir_data, exist_ok=True)
            
            # Create filename
            base_filename = f"{ts}_{self.MEASURE_NAME}_attenuation_lines_{att_str_filename}"
            html_path = f"{output_dir_html}/{base_filename}.html"
            pdf_path = f"{output_dir_pdf}/{base_filename}.pdf"
            data_path = f"{output_dir_data}/{base_filename}.npz"
            
            # Save plot files
            try:
                # Save HTML and PDF
                fig.write_html(html_path)
                fig.write_image(pdf_path, format="pdf")
                
                # Save data as NPZ file
                data_dict = {
                    'header': self.header,
                    'info': self.info,
                    'x': self.x,  # Frequency array
                    'attenuation_values': np.array(valid_atts),
                    'measure_name': self.MEASURE_NAME,
                    'timestamp': ts
                }
                
                # Add data for each attenuation value
                for i, att in enumerate(valid_atts):
                    idx = np.where(self.y == att)[0][0]
                    if y_info == "magnitude":
                        data_dict[f'magnitude_{att}db'] = self.data[idx]
                    else:
                        data_dict[f'phase_{att}db'] = self.phases[idx]
                
                # Save the data dictionary
                np.savez(data_path, **data_dict)
                
                print(f"Plot saved to {html_path} and {pdf_path}")
                print(f"Data saved to {data_path}")
            except Exception as e:
                print(f"Warning: Could not save files: {e}")
        
        fig.show()
        return None

    def run(self):
        """
        Override BaseMeasurement.run() with custom implementation.
        
        For power sweep, we implement row-by-row measurement for each attenuation value.
        Shows partial plot updates after each attenuation step with consistent scale.
        """
        # Display measurement info
        self.show()
        
        vna = self.handlers['vna']
        attenuator = self.handlers['attenuator']
        
        # Enable voltage source if needed
        if 'volt_source' in self.handlers and self.volt_source_voltage != 0:
            volt_source = self.handlers['volt_source']
            volt_source.power_on()
            volt_source.set_voltage(self.volt_source_voltage)
        
        try:
            total_points = len(self.y) * len(self.x)
            point_count = 0
            
            # Initialize start time for remaining time calculation
            start_time = datetime.now()
            
            # Initialize color scale ranges for consistent visualization
            vmin = float('inf')
            vmax = float('-inf')
            vna.power_on()
            
            # Loop through each attenuation value
            for j, att_val in enumerate(self.y):
                # Set attenuator for this row
                attenuator.set_attenuation(att_val)
                vna.clear_averaging()  # Clear any previous averaging
                
                # Perform VNA sweep for this attenuation
                
                start = time.time()
                while time.time() - start < self.suggested_delay:
                    time.sleep(0.1)  # Small sleep
                    # Check for manual interrupt
                    if hasattr(self, '_interrupted') and self._interrupted:
                        self._interrupted = True
                        raise KeyboardInterrupt
                
                # Get full sweep data for this attenuation
                result = vna.get_data()
                
                if result is None or 'mags' not in result:
                    print(f"Warning: No data for attenuation={att_val}")
                    self.data[j, :] = np.zeros(len(self.x))
                    continue
                
                # Extract and store data
                mags = result.get('mags', np.zeros(len(self.x)))
                phases = result.get('phases', np.zeros(len(self.x)))
                freqs = result.get('freqs', self.x)
                
                # Make sure the frequencies match our expected x-axis
                if len(freqs) != len(self.x):
                    # Interpolate if necessary
                    self.data[j, :] = np.interp(self.x, freqs, mags)
                    self.phases[j, :] = np.interp(self.x, freqs, phases)
                else:
                    self.data[j, :] = mags
                    self.phases[j, :] = phases
                
                # Update progress counter
                point_count += len(self.x)
                
                # Import clear_output only when needed
                from IPython.display import clear_output
                
                # Show updated plot after each sweep (but don't save interim results)
                if j > 0:  # Only show plot if we have at least two rows of data
                    # Create a full-sized data array with NaN values for unmeasured points
                    # This ensures consistent color scale and axis limits
                    full_sized_data = np.full((len(self.y), len(self.x)), np.nan)
                    # Fill in only the measured values
                    full_sized_data[:j+1, :] = self.data[:j+1, :]
                    
                    # Clear previous output before showing the plot (avoids overlap with progress bar)
                    clear_output(wait=True)
                               
                    
                    # Use temporary plot object for visualization
                    progress_fig = Plot(
                        measure_name=self.MEASURE_NAME,
                        measurement_info=self.info,
                        dir_name=self.DIR_NAME,
                        xaxis_name="Frequency (Hz)",
                        yaxis_name="Attenuation (dB)",
                        name_info=[f"Measured {j+1} of {len(self.y)} attenuation values"],
                        title=f"{self.MEASURE_NAME} Measurement",
                        data_x=self.x,
                        data_y=self.y,  # Use full y-axis range
                        data_z=full_sized_data,  # Use full-sized data with NaNs
                        zaxis_name="Magnitude (dB)"
                    ).plot_3d(save=False)  # Don't save intermediate plots
                    
                    # Update colorscale to handle NaN values properly
                    progress_fig.update_traces(
                        hovertemplate="Frequency: %{x}<br>Attenuation: %{y}<br>Magnitude: %{z}<br><extra></extra>",
                        colorscale=[
                            [0, "rgb(68, 1, 84)"],
                            [0.2, "rgb(59, 82, 139)"],
                            [0.4, "rgb(33, 145, 140)"],
                            [0.6, "rgb(94, 201, 98)"],
                            [0.8, "rgb(210, 229, 27)"],
                            [1.0, "rgb(253, 231, 37)"]
                        ],
                        # Properly handle NaN values in the heatmap
                        showscale=True,
                        connectgaps=False
                    )

                    # Show progress info
                    print(f"Completed attenuation value {j+1}/{len(self.y)}: {att_val} dB")
                    print("\nMeasurement in progress, please wait...")
                    
                    # Show the plot
                    progress_fig.show()

                # Show progress with elapsed and remaining time
                self._show_progress(point_count, total_points, start_time)

            # Clear the last progress display before showing final results
            clear_output(wait=True)
            
            # Calculate and display total elapsed time
            elapsed_time = datetime.now() - start_time
            elapsed_str = self._format_time_delta(elapsed_time)
            
            # Save and display final results
            self.save_data()
            self.show()
            print(f"\nMeasurement completed successfully! Total time: {elapsed_str}")
            self.plot(save=True)  # Only save the final plot
            
        except KeyboardInterrupt:
            print("Measurement interrupted by user.")
        except Exception as e:
            print(f"Error during measurement: {e}")
        finally:
            # Clean up resources
            self.cleanup()
        
        # Store the data in instance variables, but don't return anything to avoid polluting notebook output
        # This data is already accessible via self.x, self.y, self.data, etc.
        
        return None
