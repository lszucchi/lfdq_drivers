import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Union
from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot
from datetime import datetime
from pathlib import Path

MEASURE_NAME = "TwoToneAWG"
DIR_NAME = "twotoneawg"

class TwoToneAWGMeasurement(BaseMeasurement):
    """
    Two-tone AWG-based measurement class.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME

    def __init__(self, header: str,
                 attenuation: float,
                 rf_measurement_amplitude: float,
                 rf_measurement_frequency: float,
                 rf_excitation_amplitude: float,
                 excitation_start: float,
                 excitation_stop: float,
                 excitation_step: float,
                 awg_sample_rate: float,
                 awg_ref_clock: float,
                 records_per_buffer: int,
                 buffers_per_acquisition: int,
                 amplitude_reference_alazar: float,
                 trigger_level_volts: float,
                 trigger_range_volts: float,
                 waveform_head_cut: int,
                 period: int,
                 buffer_timeout_ms: int,
                 decimation_value: int,
                 volt_source_voltage: float = 0,
                 awg_handler: Optional[Handler] = None,
                 alazar_handler: Optional[Handler] = None,
                 rf_measurement_handler: Optional[Handler] = None,
                 rf_excitation_handler: Optional[Handler] = None,
                 attenuator_handler: Optional[Handler] = None):
        # x axis: excitation frequencies, y axis: single attenuation value
        x = np.arange(excitation_start, excitation_stop, excitation_step)
        y = np.array([attenuation])
        # instrument handlers
        handlers = {
            'awg': awg_handler or Handler(InstrumentType.AWG, "169.254.101.100"),
            'alazar': alazar_handler or Handler(InstrumentType.ALAZAR, "ALAZAR_BOARD::0"),
            'rfm': rf_measurement_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.103::inst0::INSTR", source_number=1),
            'rfe': rf_excitation_handler or Handler(InstrumentType.RF_SOURCE, "TCPIP0::169.254.101.104::inst0::INSTR", source_number=2),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR, "TCPIP0::169.254.101.101::inst0::INSTR")
        }
        # store other params
        self.rf_measurement_amplitude = rf_measurement_amplitude
        self.rf_measurement_frequency = rf_measurement_frequency
        self.rf_excitation_amplitude = rf_excitation_amplitude
        self.awg_sample_rate = awg_sample_rate
        self.awg_ref_clock = awg_ref_clock
        self.records_per_buffer = records_per_buffer
        self.buffers_per_acquisition = buffers_per_acquisition
        self.amplitude_reference_alazar = amplitude_reference_alazar
        self.trigger_level_volts = trigger_level_volts
        self.trigger_range_volts = trigger_range_volts
        self.waveform_head_cut = waveform_head_cut
        self.period = period
        self.buffer_timeout_ms = buffer_timeout_ms
        self.decimation_value = decimation_value
        self.volt_source_voltage = volt_source_voltage
        super().__init__(header, x, y, handlers)

    def prepare(self):
        # Configure AWG
        awg = self.handlers['awg']
        awg.configure_measurement(awg_amplitude=1.0, awg_offset=0.0, ref_clock_frequency=self.awg_ref_clock)
        awg.configure_awg_advanced(sample_rate=self.awg_sample_rate, ref_clock=self.awg_ref_clock, channels_config={})
        # Configure RF sources
        rfm = self.handlers['rfm']
        rfm.configure_measurement(rf_amplitude=self.rf_measurement_amplitude, rf_frequency=self.rf_measurement_frequency)
        rfe = self.handlers['rfe']
        rfe.configure_measurement(rf_amplitude=self.rf_excitation_amplitude, rf_frequency=self.x[0])
        # Configure Alazar
        alazar = self.handlers['alazar']
        alazar.configure_alazar_specific = getattr(alazar, 'configure_alazar', None)
        # Info
        self.info = f"TWOTONE AWG: {self.x[0]/1e9:.3f}-{self.x[-1]/1e9:.3f}GHz, Att={self.y[0]}dB"

    def measure_point(self, freq: float, att: float):
        # set attenuation, excitation freq, start AWG pulse
        self.handlers['att'].set_attenuation(att)
        rfe = self.handlers['rfe']
        rfe.set_frequency(freq)
        # capture data
        I, Q = self.handlers['alazar'].capture_alazar_data()
        return np.sqrt(I**2 + Q**2)

    def _set_y(self, att: float):
        self.handlers['att'].set_attenuation(att)

    def plot(self, save=True):
        """
        Plot the two-tone AWG measurement result as a line since y-axis has single attenuation.
        """
        fig = Plot(
            measure_name=self.MEASURE_NAME,
            measurement_info=self.info,
            dir_name=self.DIR_NAME,
            xaxis_name="Frequency (Hz)",
            yaxis_name="Magnitude",
            name_info=[f"Att: {self.y[0]} dB"],
            title=f"{self.MEASURE_NAME} Measurement",
            data_x=self.x,
            data_y=self.data.flatten()
        ).plot(save=save)
        fig.show()
        return fig
        
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
        return super(TwoToneAWGMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
        
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
        return super(TwoToneAWGMeasurement, cls).plot_from_file(filepath, y_info=y_info, save=save)
        
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
