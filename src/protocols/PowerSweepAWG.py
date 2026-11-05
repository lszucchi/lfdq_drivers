import numpy as np
from pathlib import Path
from time import sleep
from datetime import datetime
from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

MEASURE_NAME = "PowerSweepAWG"
DIR_NAME = "powersweepawg"

class PowerSweepAWGMeasurement(BaseMeasurement):
    """
    AWG-based power sweep measurement using Handler for AWG, Alazar, RF source, and attenuator.
    Sweeps attenuation vs frequency with pulsed acquisitions.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME

    def __init__(self, header: str,
                 attenuation_range: np.ndarray,
                 start_freq: float,
                 stop_freq: float,
                 n_points: int,
                 rf_amplitude: float,
                 rf_frequency: float,
                 awg_sample_rate: float,
                 awg_ref_clock: float,
                 pre_trigger_samples: int,
                 post_trigger_samples: int,
                 records_per_buffer: int,
                 buffers_per_acquisition: int,
                 decimation_value: int,
                 volt_source_voltage: float = 0,
                 awg_handler=None,
                 alazar_handler=None,
                 rf_handler=None,
                 attenuator_handler=None):
        # build axes
        x = np.linspace(start_freq, stop_freq, n_points)
        y = attenuation_range
        # handlers
        handlers = {
            'awg': awg_handler or Handler(InstrumentType.AWG, "169.254.101.100"),
            'alazar': alazar_handler or Handler(InstrumentType.ALAZAR, "ALAZAR_BOARD::0"),
            'rf': rf_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        if volt_source_voltage:
            handlers['volt'] = Handler(InstrumentType.VOLTAGE_SOURCE, "GPIB0::2::INSTR", port=8, step_voltage=0.001)
            self.volt_source_voltage = volt_source_voltage
        # protocol params
        self.rf_amplitude = rf_amplitude
        self.rf_frequency = rf_frequency
        self.awg_sample_rate = awg_sample_rate
        self.awg_ref_clock = awg_ref_clock
        self.pre_trigger_samples = pre_trigger_samples
        self.post_trigger_samples = post_trigger_samples
        self.records_per_buffer = records_per_buffer
        self.buffers_per_acquisition = buffers_per_acquisition
        self.decimation_value = decimation_value
        super().__init__(header, x, y, handlers)

    def prepare(self):
        awg = self.handlers['awg']
        alazar = self.handlers['alazar']
        rf = self.handlers['rf']
        att = self.handlers['att']
        # set initial attenuator
        att.set_attenuation(self.y[0])
        # configure AWG for power sweep
        awg.configure_power_sweep_awg(
            awg_sample_rate=self.awg_sample_rate,
            awg_ref_clock=self.awg_ref_clock,
            rf_measurement_amplitude=self.rf_amplitude,
            rf_measurement_frequency=self.rf_frequency,
            records_per_buffer=self.records_per_buffer,
            buffers_per_acquisition=self.buffers_per_acquisition,
            amplitude_reference_alazar=-30.0,  # Default value
            trigger_level_volts=0.5,  # Default value
            trigger_range_volts=3.0,  # Default value
            waveform_head_cut=100,  # Default value
            period=14,  # Default value
            buffer_timeout_ms=5000,  # Default value
            decimation_value=self.decimation_value,
            rf_measurement_length=10e-6  # Default measurement length
        )
        # configure RF source
        rf.configure_measurement(rf_amplitude=self.rf_amplitude, rf_frequency=self.rf_frequency)
        # configure Alazar for capture
        alazar.configure_measurement(
            pre_trigger_samples=self.pre_trigger_samples,
            post_trigger_samples=self.post_trigger_samples,
            records_per_buffer=self.records_per_buffer,
            buffers_per_acquisition=self.buffers_per_acquisition,
            decimation_value=self.decimation_value
        )
        # info
        self.info = (
            f"{MEASURE_NAME} Measurement\n"
            f"Attenuation: {self.y[0]} to {self.y[-1]} dB ({len(self.y)} pts)\n"
            f"Frequency: {self.x[0]/1e9:.3f} to {self.x[-1]/1e9:.3f} GHz ({len(self.x)} pts)\n"
            f"RF Amp: {self.rf_amplitude} dBm, RF Freq: {self.rf_frequency/1e9:.3f} GHz\n"
            f"AWG SR: {self.awg_sample_rate/1e9:.1f} GS/s, RefClk: {self.awg_ref_clock/1e6:.1f} MHz\n"
            f"Alazar pre: {self.pre_trigger_samples}, post: {self.post_trigger_samples}\n"
            f"RecBuf: {self.records_per_buffer}, BufAcq: {self.buffers_per_acquisition}\n"
            f"Decim: {self.decimation_value}"
        )

    def measure_point(self, freq: float, att_val: float):
        alazar = self.handlers['alazar']
        rf = self.handlers['rf']
        att = self.handlers['att']
        # set attenuation and RF frequency
        att.set_attenuation(att_val)
        rf.set_frequency(freq)
        # capture data
        I, Q = alazar.capture_alazar_data()
        # return magnitude
        return 20 * np.log10(np.sqrt(I**2 + Q**2))

    def _set_y(self, att_val: float):
        # Set attenuation value for each row in the measurement
        self.handlers['att'].set_attenuation(att_val)

    def plot(self, save=True):
        # 3D heatmap
        fig = Plot(
            measure_name=MEASURE_NAME,
            measurement_info=self.info,
            dir_name=DIR_NAME,
            xaxis_name="Frequency (Hz)",
            yaxis_name="Attenuation (dB)",
            name_info=[f"Att: {self.y[0]}~{self.y[-1]} dB"],
            title=f"{MEASURE_NAME} Measurement",
            data_x=self.x,
            data_y=self.y,
            data_z=self.data,
            zaxis_name="Magnitude (dB)"
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
        ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        fname = f"{ts}_{self.MEASURE_NAME.lower()}.npz"
        path = out_dir / fname
        np.savez(path,
                 header=self.header,
                 info=self.info,
                 x=self.x,
                 y=self.y,
                 data=self.data,
                 freqs=self.x,
                 attenuation=self.y,
                 mags=self.data,
                 measure_name=self.MEASURE_NAME)
        print(f"Data saved to {path}")
        return path

    def cleanup(self):
        """Clean up resources after measurement."""
        try:
            # Stop RF output
            if hasattr(self.handlers['rf']._driver, 'stop_rf'):
                self.handlers['rf']._safe_driver_call('stop_rf')
            
            # Stop AWG
            if hasattr(self.handlers['awg']._driver, 'stop'):
                self.handlers['awg']._safe_driver_call('stop')
            
            # Set all instruments to idle state
            for handler in self.handlers.values():
                handler.set_idle()
                
            print("Measurement resources cleaned up.")
        except Exception as e:
            print(f"Error during cleanup: {e}")

    @classmethod
    def plot_measurement(cls, idx: int, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement by index using BaseMeasurement's implementation.
        
        Args:
            idx: Index of the measurement file from list_measurements()
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        return super(PowerSweepAWGMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
        
    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot a saved measurement from a file using BaseMeasurement's implementation.
        
        Args:
            filepath: Path to the saved data file
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        return super(PowerSweepAWGMeasurement, cls).plot_from_file(filepath, y_info=y_info, save=save)
