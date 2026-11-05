"""
PunchOut Measurement Protocol

This module implements VNA-based punchout measurements using the standardized Handler framework
and BaseMeasurement class. It combines VNA, attenuator, and optional voltage source
to perform cavity punchout measurements.

Author: Measurement System
Date: June 2025
"""

import os
import time
import numpy as np
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union

from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

MEASURE_NAME = "PunchOut"
DIR_NAME = "punchout"

class PunchOutMeasurement(BaseMeasurement):
    """
    VNA-based punchout measurement class using the BaseMeasurement framework.
    
    This class performs cavity punchout measurements using VNA to sweep
    frequency and measure transmission or reflection.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    
    def __init__(self, header: str,
                 # Fixed parameters for single measurement (y-axis has length 1)
                 attenuation: float,
                 # VNA parameters
                 vna_amplitude: float,
                 vna_parameter: str,
                 start_frequency: float,
                 stop_frequency: float,
                 vna_n_points: int,
                 vna_if_bandwidth: float,
                 vna_data_array_format: str = "MLOGarithmic",
                 vna_average_points: int = 64,
                 vna_smoothing: int = 0,
                 vna_averaging: int = 1,
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
            'attenuator': attenuator_handler or Handler(InstrumentType.ATTENUATOR, 'TCPIP0::169.254.101.101::inst0::INSTR')
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
        self.attenuation = attenuation
        
        # Data storage for phases (magnitudes stored in self.data)
        self.phases = np.zeros(vna_n_points)
        
        # Initialize BaseMeasurement with single y-value (attenuation)
        super().__init__(header, freq_array, np.array([attenuation]), handlers)

    
    def prepare(self):
        """Prepare all instruments for measurement."""
        vna = self.handlers['vna']
        vna._driver.open()
        attenuator = self.handlers['attenuator']
        attenuator._driver.open()
         
        # Set VNA to idle first
        self._vna_idle()
        
        # Set attenuator
        attenuator.set_attenuation(self.attenuation)

        self.center_frequency = (self.x[0] + self.x[-1]) / 2
        self.span_frequency = self.x[-1] - self.x[0]
        
        # Configure VNA using Handler's prepare_punchout method for consistency
        vna.configure_measurement(
            amplitude=self.vna_amplitude,
            parameter=self.vna_parameter,
            start_frequency=self.x[0],  # First frequency
            stop_frequency=self.x[-1],  # Last frequency
            sweep_points=len(self.x),
            center_frequency=self.center_frequency,
            span_frequency=self.span_frequency,
            if_bandwidth=self.vna_if_bandwidth,
            data_array_format=self.vna_data_array_format,
            average_points=self.vna_average_points,
            smoothing=self.vna_smoothing,
            averaging_enabled=self.vna_averaging 
        )

        self.suggested_delay = vna.config.calculate_delay()
        
        # Set voltage source to idle if needed
        if 'volt_source' in self.handlers:
            self._volt_source_idle()
        
        # Update information string
        self.info = f"""
        PUNCHOUT MEASUREMENT
        ----------------------------------
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Attenuation: {self.attenuation} dB
        VNA Amplitude: {self.vna_amplitude} dBm
        VNA Parameter: {vna.config.parameter}
        VNA Center frequency: {vna.config.center_frequency/1e9} GHz
        VNA Span: {vna.config.span_frequency/1e6} MHz
        VNA IF bandwidth: {vna.config.if_bandwidth:.2e} Hz
        VNA Sweep points: {vna.config.sweep_points}
        VNA Data array format: {vna.config.data_array_format}
        VNA Smoothing: {vna.config.smoothing}
        Voltage source: {self.volt_source_voltage} V
        Suggested delay: {self.suggested_delay} s
        """

         # Add notes to self.info if provided
        if self.notes:
            self.info += f"Notes: {self.notes}\n"

        self.show()
    
    def _vna_idle(self):
        """Set VNA to idle mode."""
        vna = self.handlers['vna']
        vna.power_off()
        vna.clear_averaging()
    
    def _volt_source_idle(self):
        """Set voltage source to idle mode."""
        if 'volt_source' in self.handlers:
            self.handlers['volt_source'].set_idle()


    def measure_point(self, freq: float, att: float):
        """
        Measure a single frequency point.
        
        In punchout measurement, we sweep all frequencies at once, 
        so this is a placeholder that returns the corresponding magnitude from the sweep.
        
        Args:
            freq: Frequency to measure
            att: Attenuation value (unused in this implementation)
            
        Returns:
            float: Magnitude at this frequency
        """
        # Find index of this frequency in the array
        freq_idx = np.where(self.x == freq)[0][0]
        
        # Return corresponding magnitude from sweep
        return self.data[0, freq_idx] if len(self.data) > 0 else 0
    
    def run(self):
        """
        Override BaseMeasurement.run() with custom implementation.
        
        For punchout, we run a single sweep instead of point-by-point measurement.
        """
        vna = self.handlers['vna']
        
        
        try:
            # Enable voltage source if needed
            if 'volt_source' in self.handlers and self.volt_source_voltage != 0:
                volt_source = self.handlers['volt_source']
                volt_source.power_on()
                time.sleep(0.1)
                volt_source.set_voltage(self.volt_source_voltage)
            
            # Power on VNA and prepare for measurement
            vna.power_on()
            vna.clear_averaging()
            
            # Wait for settling
            start = time.time()
            while time.time() - start < self.suggested_delay:
                time.sleep(0.1)  # Small sleep
                # Check for manual interrupt
                if hasattr(self, '_interrupted') and self._interrupted:
                    self._interrupted = True
                    raise KeyboardInterrupt
            
            # Get data from VNA using Handler API
            result = vna.get_data()
            
            if result is None:
                print("Warning: VNA measurement returned no data")
                return {
                    'freqs': self.x,
                    'mags': np.zeros_like(self.x),
                    'phases': np.zeros_like(self.x),
                    'header': self.header,
                    'info': self.info,
                    'attenuation': self.attenuation
                }
            
            # Unpack results
            mags = result['mags']
            phases = result.get('phases', np.zeros_like(mags))  # Handle missing
            freqs = result.get('freqs', self.x)  # Handle missing
            
            # Store results
            self.data[0, :] = mags  # Store in first row of data matrix
            self.phases = phases
            
            # Calculate maximum point
            max_idx = np.argmax(self.data[0, :])
            self.max_freq = self.x[max_idx]
            self.max_mag = self.data[0, max_idx]
            
            # Save data and plot results
            self.save_data()
            self.plot(save=True)
            
        except KeyboardInterrupt:
            print("Measurement interrupted by user.")
        except Exception as e:
            print(f"Error during measurement: {e}")
        finally:
            # Always clean up resources
            self.cleanup()
        
        return None
    
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
        data_x = self.x
        data_y = self.data[0, :] if y_info == "magnitude" else self.phases
        
        xaxis_name = "Frequency (Hz)"
        yaxis_name = "Magnitude (dB)" if y_info == "magnitude" else "Phase (degrees)"
        
        # Calculate max values if they don't exist
        if not hasattr(self, 'max_freq') or not hasattr(self, 'max_mag'):
            if self.data.size > 0:
                max_idx = np.argmax(self.data[0, :])
                self.max_freq = self.x[max_idx]
                self.max_mag = self.data[0, max_idx]
            else:
                self.max_freq = 0
                self.max_mag = 0
        
        # Create name info
        name_info = [
            f"Att {self.attenuation} dB",
            f"Max Freq {self.max_freq:.2e} Hz"
        ]
        
        # Create plot using Plot class
        plot_obj = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name=xaxis_name,
            yaxis_name=yaxis_name,
            name_info=name_info,
            title=f"{self.MEASURE_NAME} Measurement",
            data_x=data_x,
            data_y=data_y
        )
        
        fig = plot_obj.plot(save=save)
        
        # Add maximum point marker
        if self.data.size > 0:
            fig.add_scatter(
                x=[self.max_freq],
                y=[self.max_mag],
                mode='markers+text',
                text=[f"({self.max_freq:.2e}, {self.max_mag:.2f})"],
                textposition="top center",
                name="Max Point"
            )
        
        # Store save name
        name_info_str = '_'.join([info.replace(' ', '').replace('.', '') for info in name_info])
        self.save_name = f"{self.MEASURE_NAME}_{name_info_str}"
        
        fig.show()

        return fig


    @classmethod
    def double_plot(cls, freqs1, mags1, freqs2, mags2):
        """
        Plot two sets of magnitude data against frequency.
        
        Args:
            freqs1: Frequency data for the first dataset
            mags1: Magnitude data for the first dataset
            freqs2: Frequency data for the second dataset
            mags2: Magnitude data for the second dataset
            
        Returns:
            Plotly figure object
        """
        # Find the maximum magnitude and its corresponding frequency for both datasets
        max_idx1 = np.argmax(mags1)
        max_freq1 = freqs1[max_idx1]
        max_mag1 = mags1[max_idx1]

        max_idx2 = np.argmax(mags2)
        max_freq2 = freqs2[max_idx2]
        max_mag2 = mags2[max_idx2]

        # Calculate the Delta parameter
        delta = abs(max_freq1 - max_freq2)

        # Create the plot
        fig = px.line(x=freqs1, y=mags1, labels={'x': 'Frequency (Hz)', 'y': 'Magnitude (dB)'})
        fig.add_scatter(x=freqs2, y=mags2, mode='lines', name='Dataset 2')
        fig.add_scatter(x=[max_freq1], y=[max_mag1], mode='markers+text', text=[f"({max_freq1:.2e}, {max_mag1:.2f})"],
                        textposition="top center", name="Max Point 1")
        fig.add_scatter(x=[max_freq2], y=[max_mag2], mode='markers+text', text=[f"({max_freq2:.2e}, {max_mag2:.2f})"],
                        textposition="top center", name="Max Point 2")
        fig.update_layout(
            title=f"Punch-out measurement - Max Freq: {max_freq1:.2e} Hz vs {max_freq2:.2e} Hz",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            annotations=[
            dict(
                x=max_freq1,
                y=max_mag1,
                xref="x",
                yref="y",
                text=f"Delta: {delta:.3e} Hz",
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-40
            )
            ]
        )

        # Create output directories
        output_dir_html = f"{cls.DIR_NAME}/html"
        output_dir_pdf = f"{cls.DIR_NAME}/pdf"
        os.makedirs(output_dir_html, exist_ok=True)
        os.makedirs(output_dir_pdf, exist_ok=True)

        # Save the plot to files
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        html_filename = f"{output_dir_html}/{cls.MEASURE_NAME}_double_{timestamp}_maxFreq{max_freq1:.2e}Hz_vs_{max_freq2:.2e}Hz.html"
        fig.write_html(html_filename)

        pdf_filename = f"{output_dir_pdf}/{cls.MEASURE_NAME}_double_{timestamp}_maxFreq{max_freq1:.2e}Hz_vs_{max_freq2:.2e}Hz.pdf"
        fig.write_image(pdf_filename, format="pdf")

        return fig


    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, data, header, and info.
        Override of BaseMeasurement.save_data() to maintain compatibility
        with legacy code.
        """
        # Create output directory
        output_dir_data = f"{self.DIR_NAME}/data"
        os.makedirs(output_dir_data, exist_ok=True)
        
        # Generate filename with timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        if not hasattr(self, 'save_name'):
            name_info = [f"Att{self.attenuation}dB"]
            if hasattr(self, 'max_freq'):
                name_info.append(f"MaxFreq{self.max_freq:.2e}Hz")
            self.save_name = f"{self.MEASURE_NAME}_{'_'.join(name_info)}"
        
        filename = f"{output_dir_data}/{timestamp}_{self.save_name}.npz"
        
        # Save data with compatible format for both old and new code
        data_dict = {
            'header': self.header,
            'info': self.info,
            'measurement_info': self.info,  # For backward compatibility
            'attenuation': self.attenuation,
            'x': self.x,
            'freqs': self.x,  # For backward compatibility
            'data': self.data,
            'mags': self.data[0],  # For backward compatibility
            'phases': self.phases,
            'y': np.array([self.attenuation])
        }
        
        np.savez(filename, **data_dict)
        print(f"Data saved to {filename}")
        
        return filename
    

    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot measurement data from a file with max point marker.
        
        Args:
            filepath: Path to the saved data file (.npz or .npy)
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        # Use base class implementation
        fig = super(PunchOutMeasurement, cls).plot_from_file(filepath, y_info=y_info, save=save)
        
        if fig is not None and y_info == "magnitude":
            try:
                # Load data to find max point
                data = np.load(filepath, allow_pickle=True)
                
                # Extract frequency and magnitude data
                if isinstance(data, np.lib.npyio.NpzFile):
                    # Get frequency data (with compatibility for old data files)
                    if 'x' in data:
                        freqs = data['x']
                    else:
                        freqs = data['freqs']
                    
                    # Get magnitude data (with compatibility for old data files)
                    if 'data' in data:
                        if len(data['data'].shape) > 1:
                            mags = data['data'][0]  # First row of data matrix
                        else:
                            mags = data['data']
                    else:
                        mags = data['mags']
                else:
                    # Legacy NPY item() format
                    data = data.item()
                    freqs = data.get('x', data.get('freqs'))
                    mags = data.get('data', data.get('mags'))
                    if isinstance(mags, np.ndarray) and len(mags.shape) > 1:
                        mags = mags[0]  # First row of data matrix
                
                # Find the maximum magnitude and its corresponding frequency
                max_idx = np.argmax(mags)
                max_freq = freqs[max_idx]
                max_mag = mags[max_idx]

                # Add a scatter point for the maximum magnitude
                fig.add_scatter(
                    x=[max_freq],
                    y=[max_mag],
                    mode='markers+text',
                    text=[f"({max_freq:.2e}, {max_mag:.2f})"],
                    textposition="top center",
                    name="Max Point"
                )
            except Exception as e:
                print(f"Could not add max point marker: {e}")
        
        return fig

    
    @classmethod
    def list_measurements(cls):
        """
        Lists all measurement files in the data directory.
        
        Returns:
            list: List of measurement files
        """
        return super().list_measurements()


    @classmethod
    def plot_measurement(cls, file_index, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement by index.
        
        Args:
            file_index: Index of the file to load
            y_info: Type of data to plot ("magnitude" or "phase")
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        return super(PunchOutMeasurement, cls).plot_measurement(file_index, y_info=y_info, save=save)
    
    def cleanup(self):
        """Clean up instruments and connections."""
        try:
            # Set VNA to idle
            self._vna_idle()
            
            # Set voltage source to idle if needed
            if 'volt_source' in self.handlers and self.volt_source_voltage != 0:
                self._volt_source_idle()

            for name, handler in self.handlers.items():
                if hasattr(handler, 'cleanup'):
                    handler.cleanup()
                
            print("Measurement resources cleaned up.")
        except Exception as e:
            print(f"Warning during cleanup: {e}")

                
