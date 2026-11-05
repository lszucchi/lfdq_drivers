import numpy as np
from pathlib import Path
from time import sleep
from datetime import datetime
from IPython.display import clear_output

from protocols.BaseMeasurement import BaseMeasurement
from protocols.Handler import Handler, InstrumentType
from protocols.Plot import Plot

MEASURE_NAME = "FluxSweep"
DIR_NAME = "fluxsweep"

class FluxSweepMeasurement(BaseMeasurement):
    """
    Flux sweep measurement: sweep VNA over frequency while stepping voltage.
    Inherits common run/save/plot behavior from BaseMeasurement.
    """
    MEASURE_NAME = MEASURE_NAME
    DIR_NAME = DIR_NAME
    def __init__(self, header: str,
                 voltage_init: float,
                 voltage_final: float,
                 voltage_step: float,
                 attenuation: float,
                 cavity_frequency: float,
                 vna_span: float,
                 vna_amplitude: float,
                 vna_parameter: str,
                 vna_if_bandwidth: float,
                 vna_n_points: int,
                 vna_data_array_format: str,
                 vna_average_points: int,
                 vna_smoothing: int = 0,
                 voltage_source_handler=None,
                 attenuator_handler=None,
                 vna_handler=None):
        # X axis: frequency sweep around cavity
        x = np.linspace(cavity_frequency - vna_span/2,
                        cavity_frequency + vna_span/2,
                        vna_n_points)
        # Y axis: voltage steps
        y = np.arange(voltage_init, voltage_final + voltage_step, voltage_step)
        # instrument handlers
        handlers = {
            'vna': vna_handler or Handler(InstrumentType.VNA,
                                          "TCPIP0::169.254.101.105::inst0::INSTR"),
            'att': attenuator_handler or Handler(InstrumentType.ATTENUATOR,
                                                 "TCPIP0::169.254.101.101::inst0::INSTR"),
            'volt': voltage_source_handler or Handler(InstrumentType.VOLTAGE_SOURCE,
                                                      "GPIB0::2::INSTR", port=8, step_voltage=0.001)
        }
        # store protocol parameters
        self.vna_amplitude = vna_amplitude
        self.vna_parameter = vna_parameter
        self.vna_if_bandwidth = vna_if_bandwidth
        self.vna_data_array_format = vna_data_array_format
        self.vna_average_points = vna_average_points
        self.vna_smoothing = vna_smoothing
        self.attenuation = attenuation
        super().__init__(header, x, y, handlers)

    def prepare(self):
        vna = self.handlers['vna']
        att = self.handlers['att']
        volt = self.handlers['volt']
        # Attenuator and voltage source idle
        att.set_attenuation(self.attenuation)
        volt.set_idle()
        # VNA idle
        vna.power_off()
        vna.clear_averaging()
        # Prepare flux sweep
        self.delay = vna.prepare_flux_sweep(
            self.vna_amplitude,
            self.vna_parameter,
            self.x[0], self.x[-1], len(self.x),
            self.vna_if_bandwidth,
            self.vna_data_array_format,
            self.vna_average_points,
            self.vna_smoothing,
            self.y
        )
        # build info string
        self.info = (
            f"FLUX SWEEP\n"
            f"Voltage: {self.y[0]:.3f} to {self.y[-1]:.3f} V ({len(self.y)} pts)\n"
            f"Attenuation: {self.attenuation} dB\n"
            f"Frequency: {self.x[0]/1e9:.3f} to {self.x[-1]/1e9:.3f} GHz ({len(self.x)} pts)\n"
            f"VNA Amp: {self.vna_amplitude} dBm, Param: {self.vna_parameter}\n"
            f"IFBW: {self.vna_if_bandwidth} Hz, DataFmt: {self.vna_data_array_format}\n"
            f"AvgPts: {self.vna_average_points}, Smoothing: {self.vna_smoothing}\n"
            f"Delay per row: {self.delay} s"
        )

    def measure_point(self, freq: float, volt_val: float):
        vna = self.handlers['vna']
        volt = self.handlers['volt']
        # set voltage and attenuation
        volt.power_on()
        sleep(0.1)
        volt.set_voltage(volt_val)
        # set VNA parameters
        vna.set_power(self.vna_amplitude)
        vna.set_frequency(freq)
        # single-point measurement via sweep point
        result = vna.run_flux_sweep_point(self.delay)
        mags, phases, freqs = result
        # pick closest frequency index
        idx = (np.abs(freqs - freq)).argmin()
        return mags[idx]

    def _set_y(self, volt_val: float):
        # set voltage at start of each row
        volt = self.handlers['volt']
        volt.power_on()
        sleep(0.1)
        volt.set_voltage(volt_val)

    def plot(self, save=True):
        # override to include MEASURE_NAME and DIR_NAME
        fig = Plot(
            measure_name=MEASURE_NAME,
            measurement_info=self.info,
            dir_name=DIR_NAME,
            xaxis_name="Frequency (Hz)",
            yaxis_name="Voltage (V)",
            name_info=[f"Volt: {self.y[0]:.3f}-{self.y[-1]:.3f} V"],
            title=f"{MEASURE_NAME} Measurement",
            data_x=self.x,
            data_y=self.y,
            data_z=self.data,
            zaxis_name="Magnitude"
        ).plot_3d(save=save)
        fig.show()
        return fig

    def save_data(self):
        """
        Save measurement data with standard format.
        """
        measure_name = getattr(self, 'MEASURE_NAME', self.__class__.__name__)
        dir_name = getattr(self, 'DIR_NAME', self.DIR_NAME)
        out_dir = Path(dir_name) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{measure_name.lower()}.npz"
        path = out_dir / fname
        
        # Save with standard keys for BaseMeasurement compatibility
        np.savez(path,
                 header=self.header,
                 info=self.info,
                 x=self.x,  # Standard key
                 freqs=self.x,  # Legacy key
                 y=self.y,  # Standard key
                 volts=self.y,  # Legacy key
                 data=self.data,  # Standard key
                 mags=self.data,  # Legacy key
                 measure_name=measure_name)
        
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
        return super(FluxSweepMeasurement, cls).plot_measurement(idx, y_info=y_info, save=save)
        
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
        return super(FluxSweepMeasurement, cls).plot_from_file(filepath, y_info=y_info, save=save)

    def cleanup(self):
        """
        Perform cleanup of all instruments after measurement.
        """
        # Return voltage source to idle state
        if 'volt' in self.handlers:
            self.handlers['volt'].set_idle()
        
        # General cleanup for all handlers
        for name, handler in self.handlers.items():
            if hasattr(handler, 'cleanup'):
                handler.cleanup()
