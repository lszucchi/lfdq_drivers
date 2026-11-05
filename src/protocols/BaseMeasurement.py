import abc
import os
import time
from pathlib import Path
from datetime import datetime
from IPython.display import clear_output
import numpy as np
from typing import Dict, List, Optional, Any  # Add this import
from protocols.Plot import Plot
import inspect

class BaseMeasurement(abc.ABC):
    """
    Abstract base class for all measurement protocols.
    Provides array generation, loop scaffolding, progress reporting,
    data saving, and common file listing/plotting mechanisms.
    """
    def __init__(self, header: str, x_array: np.ndarray, y_array: np.ndarray, handlers: dict):
        self.header = header
        # Measurement axes
        self.x = x_array
        self.y = y_array
        # Data storage: 2D array for magnitude or single value
        self.data = np.zeros((len(self.y), len(self.x)), dtype=float)
        self.handlers = handlers
        # Directory name for saving
        self.DIR_NAME = self.__class__.__name__.lower()
        # Info string
        self.info = ""
        # Prepare instruments
        self.prepare()

    @abc.abstractmethod
    def prepare(self):
        """
        Instrument-specific preparation.
        Should set self.info and any timing parameters.
        """
        pass

    @abc.abstractmethod
    def measure_point(self, x_val, y_val):
        """
        Instrument-specific point measurement.
        Should return a scalar or tuple, depending on protocol.
        """
        pass

    def run(self):
        """
        Generic run loop over y and x axes, storing results in self.data.
        Provides real-time partial plotting during measurement.
        """
        # display measurement info
        self.show()
        total = len(self.y) * len(self.x)
        count = 0
        for j, yv in enumerate(self.y):
            clear_output(wait=True)
            self._set_y(yv)
            for i, xv in enumerate(self.x):
                clear_output(wait=True)
                result = self.measure_point(xv, yv)
                self.data[j, i] = result
                count += 1
                self._show_progress(count, total)
                
                # Update plot in real-time with partial=True every 5 points or at the end of each row
                # This provides visual feedback without slowing down the measurement too much
                if count % 1 == 0 or i == len(self.x) - 1:
                    self.plot(save=False, partial=True)  # Update plot in real-time
                
                if hasattr(self, '_interrupted') and self._interrupted:
                    self._interrupted = True
                    raise KeyboardInterrupt
            
            # Plot after each completed row
            clear_output(wait=True)
            self._show_progress(count, total)
            self.plot(save=False, partial=True)
            
        # save and plot final results
        self.save_data()
        self.show()
        self.plot(save=True, partial=False)  # Final complete plot
        self.show()
        self.plot(save=True)

    def _set_y(self, y_val):
        # Override if instruments require a change along y axis (e.g. attenuation)
        pass

    def _show_progress(self, count, total):
        pct = count / total * 100
        bar_len = 40
        filled = int(round(bar_len * pct / 100))
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"Progress: [{bar}] {pct:.1f}% ({count}/{total})")

    def save_data(self):
        """
        Save raw data to numpy .npz with x, y, data, header, and info.
        """
        measure_name = getattr(self, 'MEASURE_NAME', self.__class__.__name__)
        dir_name = getattr(self, 'DIR_NAME', self.DIR_NAME)
        out_dir = Path(dir_name) / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{measure_name.lower()}.npz"
        path = out_dir / fname
        np.savez(path,
                 header=self.header,
                 info=self.info,
                 x=self.x,
                 y=self.y,
                 data=self.data)
        print(f"Data saved to {path}")
        return path

    @classmethod
    def list_measurements(cls):
        """
        List saved measurement files for this protocol.
        """
        data_dir = Path(cls.__name__.lower()) / "data"
        if not data_dir.exists():
            print("No measurements found.")
            return []
        files = sorted(data_dir.iterdir())
        for i, f in enumerate(files):
            print(f"[{i}] {f.name}")
        return files

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
        files = cls.list_measurements()
        if not files or idx < 0 or idx >= len(files):
            raise IndexError("Invalid measurement index")
        
        return cls.plot_from_file(str(files[idx]), y_info=y_info, save=save)

    @classmethod
    def plot_from_file(cls, filepath, y_info="magnitude", save=False):
        """
        Load and plot measurement data from a file without requiring a custom plot function.
        
        Args:
            filepath: Path to the saved data file (.npz or .npy)
            y_info: Type of data to plot ("magnitude" or other data type if available)
            save: Whether to save the generated plot
            
        Returns:
            Plotly figure object
        """
        try:
            # Try loading as .npz (preferred format)
            if filepath.endswith('.npz'):
                data = np.load(filepath)
                # Extract data using standard keys
                x = data['x'] if 'x' in data else data['freqs'] if 'freqs' in data else None
                y = data['y'] if 'y' in data else None
                measurement_data = data['data'] if 'data' in data else data['mags'] if 'mags' in data else None
                info = str(data.get('info', data.get('measurement_info', 'No info available')))
                
                # Handle single dimension data
                if measurement_data is not None and len(measurement_data.shape) == 1:
                    measurement_data = np.reshape(measurement_data, (1, -1))
                
                # If we have y values (e.g., attenuation, voltage, etc.)
                if y is not None and measurement_data is not None and x is not None:
                    # 3D or 2D plot depending on y dimension
                    if len(y) > 1:
                        # 3D plot
                        name_info = [f"{cls.__name__} measurement"]
                        measure_name = data.get('measure_name', cls.__name__)
                        dir_name = getattr(cls, 'DIR_NAME', cls.__name__.lower())
                        fig = Plot(
                            measure_name=measure_name,
                            measurement_info=info,
                            dir_name=dir_name,
                            xaxis_name="X axis",
                            yaxis_name="Y axis",
                            name_info=name_info,
                            title=f"{measure_name} Measurement",
                            data_x=x,
                            data_y=y,
                            data_z=measurement_data,
                            zaxis_name=y_info.capitalize()
                        ).plot_3d(save=save)
                    else:
                        # 2D plot (single row)
                        name_info = [f"{cls.__name__} measurement"]
                        measure_name = data.get('measure_name', cls.__name__)
                        dir_name = getattr(cls, 'DIR_NAME', cls.__name__.lower())
                        fig = Plot(
                            measure_name=measure_name,
                            measurement_info=info,
                            dir_name=dir_name,
                            xaxis_name="X axis",
                            yaxis_name=y_info.capitalize(),
                            name_info=name_info,
                            title=f"{measure_name} Measurement",
                            data_x=x,
                            data_y=measurement_data.flatten()
                        ).plot(save=save)
                    
                    fig.show()
                    return fig
                
            # Try loading as .npy (legacy format)
            elif filepath.endswith('.npy'):
                data = np.load(filepath, allow_pickle=True).item()
                # Extract data from dictionary
                x = data.get('freqs', data.get('x', None))
                measurement_data = data.get('mags', data.get('data', None))
                info = str(data.get('measurement_info', data.get('info', 'No info available')))
                
                if x is not None and measurement_data is not None:
                    # 2D plot for legacy data
                    name_info = [f"{cls.__name__} measurement"]
                    measure_name = data.get('measure_name', cls.__name__)
                    dir_name = getattr(cls, 'DIR_NAME', cls.__name__.lower())
                    fig = Plot(
                        measure_name=measure_name,
                        measurement_info=info,
                        dir_name=dir_name,
                        xaxis_name="X axis",
                        yaxis_name=y_info.capitalize(),
                        name_info=name_info,
                        title=f"{measure_name} Measurement",
                        data_x=x,
                        data_y=measurement_data
                    ).plot(save=save)
                    
                    fig.show()
                    return fig
            
            raise ValueError(f"Unsupported file format or missing required data in {filepath}")
            
        except Exception as e:
            print(f"Error loading/plotting data: {e}")
            return None

    def show(self):
        """
        Display the measurement info string.
        """
        print(self.info)

    def plot(self, save=True, partial=False):
        """
        Plot the measurement data using Plot helper.
        For single-row data, a 2D line plot is used; otherwise a 3D heatmap.
        
        Args:
            save: Whether to save the plot to file
            partial: Whether this is a partial plot during measurement (True) or final plot (False)
            
        Returns:
            Plotly figure object
        """
        measure_name = getattr(self, 'MEASURE_NAME', self.__class__.__name__)
        dir_name = getattr(self, 'DIR_NAME', self.DIR_NAME)
        # prepare name info for saving
        name_info = [self.info]
        title_suffix = " (Partial)" if partial else ""
        
        # determine plot dimensions
        if self.data.shape[0] <= 1:
            # For 2D plots with partial data, find the last non-zero element
            if partial:
                # Find the last measured point (non-zero value)
                non_zero_indices = np.nonzero(self.data[0])[0]
                if len(non_zero_indices) > 0:
                    last_idx = non_zero_indices[-1]
                    # Plot only up to the last measured point
                    data_x = self.x[:last_idx+1]
                    data_y = self.data[0, :last_idx+1]
                else:
                    # No data points yet
                    data_x = []
                    data_y = []
            else:
                # Full plot
                data_x = self.x
                data_y = self.data.flatten()
            
            fig = Plot(
                measure_name=measure_name,
                measurement_info=self.info,
                dir_name=dir_name,
                xaxis_name="X (axis)",
                yaxis_name="Y (axis)",
                name_info=name_info,
                title=measure_name + title_suffix,
                data_x=data_x,
                data_y=data_y
            ).plot(save=save)
        else:
            # 3D plot - for partial data, highlight unmeasured regions
            if partial:
                # Create a mask to identify measured values
                # Assumption: unmeasured values are exactly 0
                mask = ~np.isclose(self.data, 0)
                
                # Create a copy of the plot data for partial plotting
                partial_data = np.copy(self.data)
                
                # Set a very low value for unmeasured points (for better visualization)
                min_val = np.min(partial_data[mask]) if np.any(mask) else -120
                partial_data[~mask] = min_val
                
                plot_z_data = partial_data
            else:
                plot_z_data = self.data
            
            fig = Plot(
                measure_name=measure_name,
                measurement_info=self.info,
                dir_name=dir_name,
                xaxis_name="X (axis)",
                yaxis_name="Y (axis)",
                name_info=name_info,
                title=measure_name + title_suffix,
                data_x=self.x,
                data_y=self.y,
                data_z=plot_z_data
            ).plot_3d(save=save)
        
        fig.show()
        return fig

    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata for this measurement.
        
        Returns:
            dict: Metadata dictionary with measurement parameters
        """
        meta_info = {}
        for handler in self.handlers.values():
            if hasattr(handler.config, 'list_parameters'):
                # nest the handler.config.list_parameters() (dictionary) in the meta_info dictionary
                meta_info[handler.instrument_type] = handler.config.list_parameters()
            else:
                # If no config, just store the handler name
                meta_info[handler.instrument_type] = str(handler)
    
    @classmethod
    def template_default(cls):
        """
        Generate a default template for this measurement.
        
        Returns:
            dict: Default template with common parameters
        """
        sig = inspect.signature(cls.__init__)
        config = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.default is not inspect.Parameter.empty:
                config[name] = param.default
            else:
                config[name] = None
        return config