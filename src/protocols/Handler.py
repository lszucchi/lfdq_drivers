"""
Unified Handler class for instrument management using factory pattern.

This module provides a standardized interface for all measurement instruments,
implementing automatic connection management, state tracking, and unified methods
while maintaining backward compatibility with existing drivers.

Author: Measurement System
Date: June 2025
"""

import logging
import warnings
import time
import numpy as np
import threading
from typing import Optional, Union, Dict, Any, List, Tuple
from enum import Enum
from contextlib import contextmanager
import gc

# Import existing drivers
from instruments.E5063A_driver import E5063A_driver
from instruments.E8257D_driver import E8257D_driver
from instruments.Agilent11713C_driver import Agilent11713C_driver
from instruments.SIM928_driver import SIM928_driver
from instruments.SIM970_driver import SIM970_driver
from instruments.SMB100B_driver import SMB100B_driver
# Import new drivers
from instruments.ATS9872_driver import ATS9872_driver
from instruments.M8195A_driver import M8195A_driver
from instruments.DG645_driver import DG645_driver

from measurements.DataChannelManager import DataChannelManager 
from measurements.PulseSequence import PulseSequence
from measurements.waveforms.SquarePulse import SquarePulse
from measurements.waveforms.SquareSideBandPulse import SquareSideBandPulse
from measurements.waveforms.ZeroPulse import ZeroPulse
from measurements.waveforms.GaussianPulse import GaussianPulse
from measurements.waveforms.CosinePulse import CosinePulse
from measurements.waveforms.DragPulse import DragPulse


class InstrumentType(Enum):
    """Enumeration of supported instrument types."""
    VNA = "VNA"
    RF_SOURCE = "RF_SOURCE"
    ATTENUATOR = "ATTENUATOR"
    VOLTAGE_SOURCE = "VOLTAGE_SOURCE"
    VOLTAGE_METER = "VOLTAGE_METER"
    ALAZAR = "ALAZAR"
    AWG = "AWG"
    DELAY_GENERATOR = "DELAY_GENERATOR"
    DATA_CHANNEL = "DATA_CHANNEL"
    IF_SIGNAL_GENERATOR = "IF_SIGNAL_GENERATOR"


class InstrumentState(Enum):
    """Enumeration of instrument states."""
    DISCONNECTED = "DISCONNECTED"
    IDLE = "IDLE"
    MEASURING = "MEASURING"
    ERROR = "ERROR"


class MeasurementConfig:
    """Configuration container for measurement parameters."""
    def __init__(self, inst, inst_type):
        self.inst = inst
        self.inst_type = inst_type

        # VNA Configuration
        if inst_type == InstrumentType.VNA:
            self.amplitude = inst.get_amplitude() 
            self.parameter = inst.parameter 
            self.center_frequency = inst.center_frequency
            self.span_frequency = inst.span_frequency
            self.start_frequency = inst.start_frequency
            self.stop_frequency = inst.stop_frequency
            self.if_bandwidth = inst.if_bandwidth_frequency
            self.sweep_points = inst.sweep_points
            self.average_points = inst.average_points
            self.averaging_enabled = inst.averaging
            self.data_array_format = inst.data_array_format
            self.smoothing = inst.smooth
            self.autoscale: Optional[int] = None

        # RF Source Configuration
        if inst_type == InstrumentType.RF_SOURCE:
            self.rf_frequency = inst.get_frequency()
            self.rf_amplitude = inst.get_amplitude()
            self.rf_mod = inst.is_mod_on()
            self.pulse_trigger_source = inst.get_pulse_trigger_source()
            self.rf_phase = inst.get_phase()
        
        # Attenuator Configuration
        if inst_type == InstrumentType.ATTENUATOR:
            self.attenuation = inst.get_attenuation()
        
        # Voltage Source Configuration
        if inst_type == InstrumentType.VOLTAGE_SOURCE:
            self.voltage = inst.get_voltage()
            self.port = inst.port
            self.step_voltage = inst.step_voltage
            self.step_time = inst.step_time
        
        # Two-Tone Configuration
        self.cavity_frequency: Optional[float] = None
        self.rf_amplitudes: Optional[List[float]] = None
        
        # Alazar Configuration
        if inst_type == InstrumentType.ALAZAR:
            # ATS9872_driver doesn't have getter methods, so we set default values
            # These will be updated through configure_measurement
            self.if_frequency = 240e6  # Default IF frequency
            self.rf_measurement_length = 5e-6  
            self.pre_trigger_samples = 256
            self.records_per_buffer = 500
            self.buffers_per_acquisition = 100
            self.amplitude_reference_alazar = -8
            self.input_load = 50
            self.trigger_level_volts = 0.7
            self.trigger_range_volts = 1
            self.waveform_head_cut = 1500
            self.period = int(1/self.if_frequency * 1e9)
            self.buffer_timeout_ms = 5000
            self.decimation_value = 1
            self.ttl_trigger = True
            self.save_data = False
            self.sampling_rate = 1e9/self.decimation_value
            self.post_trigger_samples = int(self.rf_measurement_length * self.sampling_rate/256)*256
                    
        # AWG Configuration
        if inst_type == InstrumentType.AWG:
            # Safe access to M8195A_driver methods
            # Only call methods that exist in the driver
            if hasattr(inst, 'getSampleRate'):
                self.sample_rate = inst.getSampleRate()
            else:
                self.sample_rate = 64e9  # Default sample rate
            
            if hasattr(inst, 'setRefInClockExternal'):
                self.ref_in_clock_external = True
                inst.setRefInClockExternal()

            # Store the clock frequency directly (this is set during initialization)
            self.ref_clock_frequency = inst.clockFrequency if hasattr(inst, 'clockFrequency') else 10e6
            
            # Safe access to other methods
            if hasattr(inst, 'getMemoryDivision'):
                self.memory_division = inst.getMemoryDivision()
            else:
                self.memory_division = 2  # Default value
                
            # Set default values for these as they likely don't exist as methods
            self.voltage1 = inst.getVoltage(1) if hasattr(inst, 'getVoltage') else 1
            self.voltage2 = inst.getVoltage(2) if hasattr(inst, 'getVoltage') else 1
            self.voltage3 = inst.getVoltage(3) if hasattr(inst, 'getVoltage') else 0.5
            self.voltage4 = inst.getVoltage(4) if hasattr(inst, 'getVoltage') else 0.5

            self.voltageOffset1 = inst.getVoltageOffset(1) if hasattr(inst, 'getVoltageOffset') else 1
            self.voltageOffset2 = inst.getVoltageOffset(2) if hasattr(inst, 'getVoltageOffset') else 1
            self.voltageOffset3 = inst.getVoltageOffset(3) if hasattr(inst, 'getVoltageOffset') else 0.5
            self.voltageOffset4 = inst.getVoltageOffset(4) if hasattr(inst, 'getVoltageOffset') else 0.5

            self.channel_setting = {1: True, 2: True, 3: True, 4: True}  # Default enable all
            self.channel_voltage = {1: self.voltage1, 2: self.voltage2,
                                    3: self.voltage3, 4: self.voltage4}
            self.channel_voltage_offset = {1: self.voltageOffset1, 2: self.voltageOffset2,
                                            3: self.voltageOffset3, 4: self.voltageOffset4}
        
        # Delay Generator Configuration
        if inst_type == InstrumentType.DELAY_GENERATOR:
            self.trigger_rate = inst.get_trigger_rate()
            self.trigger_source = inst.get_trigger_source()
            self.delay_channels = inst.get_delay_channels()

        if inst_type == InstrumentType.DATA_CHANNEL:
            # self.channels = inst.awgChannels  # Get AWG channels from DataChannelManager
            self.channels = inst.awgChannels
            self.channel_data = inst._channelData  # Get channel data from DataChannelManager
            self.channel_config = {'Q': (2, 1), 'I': (1, 1), 'm': (1, 2)} # name: (channel, marker)
            self.rf_measurement_length = 5e-6  # Default RF measurement length
            self.total_experiment_duration = 100e-6 # Default total experiment duration
            self.pulse_sequence = None

        if inst_type == InstrumentType.IF_SIGNAL_GENERATOR:
            self.rf_on = inst.is_rf_on()
            self.rf_frequency = inst.get_frequency()
            self.rf_level = inst.get_level()
            self.rf_mod_src = inst.get_mod_src()
            self.rf_mod_on = inst.is_mod_on()


    def calculate_delay(self) -> float:
        """Calculate suggested measurement delay based on parameters."""
        # Check if parameters are defined
        if self.if_bandwidth is None or self.if_bandwidth <= 0:
            # Default if bandwidth not specified
            if_bandwidth = 1000.0
        else:
            if_bandwidth = self.if_bandwidth
            
        if self.sweep_points is None or self.sweep_points <= 0:
            # Default points if not specified
            sweep_points = 201
        else:
            sweep_points = self.sweep_points
            
        if self.average_points is None or self.average_points <= 0:
            # Default average if not specified
            avg_points = 1
        else:
            avg_points = self.average_points
            
        # Use more conservative calculation with safety factor
        base_delay = 2.3 / if_bandwidth * sweep_points * avg_points
        return max(1, np.ceil(base_delay))
    
    def list_params(self) -> Dict[str, Any]:
        """List configuration parameters as a dictionary for an specific instrument type."""
        params = {}
        if self.inst_type == InstrumentType.VNA:
            params = {
                'instrument': 'E5063A VNA', 
                'amplitude': self.amplitude,
                'parameter': self.parameter,
                'center_frequency': self.center_frequency,
                'span_frequency': self.span_frequency,
                'start_frequency': self.start_frequency,
                'stop_frequency': self.stop_frequency,
                'if_bandwidth': self.if_bandwidth,
                'sweep_points': self.sweep_points,
                'average_points': self.average_points,
                'averaging_enabled': self.averaging_enabled,
                'data_array_format': self.data_array_format,
                'smoothing': self.smoothing,
                'autoscale': self.autoscale
            }
        elif self.inst_type == InstrumentType.RF_SOURCE:
            params = {
                'instrument': 'E8257D RF Source',
                'rf_frequency': self.rf_frequency,
                'rf_amplitude': self.rf_amplitude,
                'rf_mod': self.rf_mod,
                'pulse_trigger_source': self.pulse_trigger_source,
                'rf_phase': self.rf_phase
            }
        elif self.inst_type == InstrumentType.ATTENUATOR:
            params = {
                'instrument': 'Agilent11713C Attenuator',
                'attenuation': self.attenuation
            }
        elif self.inst_type == InstrumentType.VOLTAGE_SOURCE:
            params = {
                'instrument': 'SIM928 Voltage Source',
                'voltage': self.voltage,
                'port': self.port,
                'step_voltage': self.step_voltage,
                'step_time': self.step_time
            }
        elif self.inst_type == InstrumentType.ALAZAR:
            params = {
                'instrument': 'ATS9872 Alazar',
                'if_frequency': self.if_frequency,
                'pre_trigger_samples': self.pre_trigger_samples,
                'post_trigger_samples': self.post_trigger_samples,
                'records_per_buffer': self.records_per_buffer,
                'buffers_per_acquisition': self.buffers_per_acquisition,
                'amplitude_reference_alazar': self.amplitude_reference_alazar,
                'input_load': self.input_load,
                'trigger_level_volts': self.trigger_level_volts,
                'trigger_range_volts': self.trigger_range_volts,
                'waveform_head_cut': self.waveform_head_cut,
                'period': self.period,
                'buffer_timeout_ms': self.buffer_timeout_ms,
                'decimation_value': self.decimation_value,
                'ttl_trigger': self.ttl_trigger,
                'save_data': self.save_data
            }
        elif self.inst_type == InstrumentType.AWG:
            params = {
                'instrument': 'M8195A AWG',
                'sample_rate': self.sample_rate,
                'ref_in_clock_external': self.ref_in_clock_external,
                'ref_clock_frequency': self.ref_clock_frequency,
                'memory_division': self.memory_division,
                'channel_setting': self.channel_setting,
                'channel_voltage': self.channel_voltage,
                'channel_voltage_offset': self.channel_voltage_offset
            }
        elif self.inst_type == InstrumentType.DELAY_GENERATOR:
            params = {
                'instrument': 'DG645 Delay Generator',
                'trigger_rate': self.trigger_rate,
                'trigger_source': self.trigger_source,
                'delay_channels': self.delay_channels
            }
        elif self.inst_type == InstrumentType.DATA_CHANNEL:
            params = {
                'instrument': 'DataChannelManager',
                'channels': self.channels,
                'channel_data': self.channel_data,
                'channel_config': self.channel_config,
                'rf_measurement_length': self.rf_measurement_length,
                'total_experiment_duration': self.total_experiment_duration,
                'pulse_sequence': self.pulse_sequence
            }
        elif self.inst_type == InstrumentType.IF_SIGNAL_GENERATOR:
            params = {
                'instrument': 'SMB100B IF Signal Generator',
                'rf_on': self.rf_on,
                'rf_frequency': self.rf_frequency,
                'rf_level': self.rf_level,
                'rf_mod_src': self.rf_mod_src,
                'rf_mod_on': self.rf_mod_on
            }
        else:
            raise ValueError(f"Unsupported instrument type: {self.inst_type}")
        
        return params
            

class HandlerError(Exception):
    """Base exception class for Handler errors."""
    pass


class ConnectionError(HandlerError):
    """Exception raised for connection-related errors."""
    pass


class StateError(HandlerError):
    """Exception raised for invalid state operations."""
    pass


class ParameterError(HandlerError):
    """Exception raised for invalid parameters."""
    pass


class Handler:
    """
    Unified instrument handler using factory pattern.
    
    This class provides a standardized interface for all instrument types while
    maintaining backward compatibility with existing drivers. Each Handler instance
    manages exactly one physi cal instrument.
    
    Attributes:
        instrument_type (InstrumentType): Type of the managed instrument
        address (str): Instrument connection address
        state (InstrumentState): Current instrument state
        config (MeasurementConfig): Measurement configuration
    """
    
    # Class-level mapping of instrument types to driver classes
    _DRIVER_MAP = {
        InstrumentType.VNA: E5063A_driver,
        InstrumentType.RF_SOURCE: E8257D_driver,
        InstrumentType.ATTENUATOR: Agilent11713C_driver,
        InstrumentType.VOLTAGE_SOURCE: SIM928_driver,
        InstrumentType.VOLTAGE_METER: SIM970_driver,
        InstrumentType.ALAZAR: ATS9872_driver,
        InstrumentType.AWG: M8195A_driver,
        InstrumentType.DELAY_GENERATOR: DG645_driver,
        InstrumentType.DATA_CHANNEL: DataChannelManager,
        InstrumentType.IF_SIGNAL_GENERATOR: SMB100B_driver
    }
    
    def __init__(self, instrument_type: Union[str, InstrumentType], address: str, 
                 source_number: Optional[int] = None, port: Optional[int] = None,
                 step_voltage: float = 0.001, step_time: float = 0.05,
                 auto_connect: bool = True, **kwargs):
        """
        Initialize Handler instance.
        
        Args:
            instrument_type: Type of instrument to manage
            address: Connection address (e.g., "TCPIP0::169.254.101.105::inst0::INSTR")
            source_number: RF source number (1 or 2) for RF_SOURCE type
            port: Port number for SIM devices
            step_voltage: Step voltage for voltage sources
            step_time: Step time for voltage ramping
            auto_connect: Whether to automatically connect on instantiation
            **kwargs: Additional arguments passed to the underlying driver
        
        Raises:
            ValueError: If instrument_type is not supported
            ConnectionError: If auto_connect is True and connection fails
        """
        try:
            # Convert string to enum if needed
            if isinstance(instrument_type, str):
                try:
                    instrument_type = InstrumentType(instrument_type)
                except ValueError:
                    raise ValueError(f"Unsupported instrument type: {instrument_type}")
            
            self.instrument_type = instrument_type
            self.address = address
            self.source_number = source_number
            self.port = port
            self.step_voltage = step_voltage
            self.step_time = step_time
            self.init_flag = False
            
            # Initialize logging
            self._logger = logging.getLogger(f"Handler.{instrument_type.value}")
            
            # State management
            self.state = InstrumentState.DISCONNECTED
            self._state_lock = threading.Lock()
            
            # Initialize driver instance
            self._driver: Optional[Any] = None
            self._create_driver(**kwargs)

            # Initialize measurement configuration
            self.config = MeasurementConfig(self._driver, instrument_type)
            
            # Auto-connect if requested
            if auto_connect and instrument_type != InstrumentType.DATA_CHANNEL:
                self.connect()
            elif instrument_type == InstrumentType.DATA_CHANNEL or instrument_type == InstrumentType.AWG:
                self.state = InstrumentState.IDLE
                self.configure_measurement()
                self._logger.info(f"DataChannelManager initialized in IDLE state")
                
        except Exception as e:
            # Ensure proper cleanup if initialization fails
            self.state = InstrumentState.ERROR
            if hasattr(self, '_logger'):
                self._logger.error(f"Handler initialization failed: {e}")
            raise
    
    def _create_driver(self, **kwargs):
        """Create the appropriate driver instance based on instrument type."""
        driver_class = self._DRIVER_MAP.get(self.instrument_type)
        if not driver_class:
            raise ValueError(f"No driver available for {self.instrument_type}")
        
        try:
            if self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
                # SIM928 requires port and step parameters
                self._driver = driver_class(
                    self.address, 
                    self.port or 8, 
                    step_voltage=self.step_voltage,
                    step_time=self.step_time
                )
            elif self.instrument_type == InstrumentType.VOLTAGE_METER:
                # SIM970 requires port parameter
                self._driver = driver_class(self.address, self.port or 5)
            elif self.instrument_type == InstrumentType.ALAZAR:
                # Alazar doesn't need address, it's directly instantiated
                self._driver = driver_class()
            elif self.instrument_type == InstrumentType.AWG:
                # M8195A needs address
                self._driver = driver_class(self.address, **kwargs)
            elif self.instrument_type == InstrumentType.DELAY_GENERATOR:
                # DG645 may need special initialization parameters
                self._driver = driver_class(self.address, **kwargs)
            elif self.instrument_type == InstrumentType.DATA_CHANNEL:
                self._driver = driver_class(**kwargs)
            elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
                # SMB100B requires address
                self._driver = driver_class(self.address, **kwargs)
            else:
                # Standard driver initialization
                self._driver = driver_class(self.address)
                
        except Exception as e:
            self._logger.error(f"Failed to create driver: {e}")
            raise ConnectionError(f"Failed to create driver for {self.instrument_type}: {e}")
    
    def connect(self) -> bool:
        """Connect to the instrument."""
        with self._state_lock:
            try:
                if self._driver and hasattr(self._driver, 'open'):
                    self._driver.open()
                self.state = InstrumentState.IDLE
                self._logger.info(f"Connected to {self.instrument_type} at {self.address}")
                return True
            except Exception as e:
                self.state = InstrumentState.ERROR
                self._logger.error(f"Connection failed: {e}")
                raise ConnectionError(f"Failed to connect to {self.instrument_type}: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the instrument."""
        with self._state_lock:
            try:
                if self.is_connected():
                    self.set_idle()  # Ensure instrument is in safe state
                    if self._driver and hasattr(self._driver, 'close'):
                        self._driver.close()
                self.state = InstrumentState.DISCONNECTED
                self._logger.info(f"Disconnected from {self.instrument_type}")
            except Exception as e:
                self._logger.error(f"Error during disconnect: {e}")
                self.state = InstrumentState.ERROR
    
    def is_connected(self) -> bool:
        """Check if instrument is connected."""
        return self.state not in [InstrumentState.DISCONNECTED, InstrumentState.ERROR]
    
    def get_state(self) -> InstrumentState:
        """Get current instrument state."""
        return self.state
    
    def _validate_connection(self) -> None:
        """Validate that instrument is connected before operations."""
        if not self.is_connected():
            raise StateError(f"Instrument {self.instrument_type} is not connected")
    
    def _validate_state(self, required_states: List[InstrumentState]) -> None:
        """Validate that instrument is in required state for operation."""
        if self.state not in required_states:
            raise StateError(f"Invalid state {self.state} for operation. Required: {required_states}")
    
    def _safe_driver_call(self, method_name: str, *args, is_property=False, **kwargs):
        """
        Safely call a driver method or property with proper error handling.
        
        Args:
            method_name: Name of the method or property to call
            *args: Arguments to pass to the method
            is_property: Whether the method_name is a property
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            Result of the method call or property access
        """
        if not self._driver:
            self._logger.warning(f"Driver not initialized for {method_name}")
            return None
        
        if not hasattr(self._driver, method_name):
            self._logger.warning(f"Method/property {method_name} not available for {self.instrument_type}")
            return None
        
        try:
            if is_property:
                # Handle property get/set
                if args:
                    # Setting property
                    setattr(self._driver, method_name, args[0])
                    return None
                else:
                    # Getting property
                    return getattr(self._driver, method_name)
            else:
                # Normal method call
                method = getattr(self._driver, method_name)
                return method(*args, **kwargs)
        except Exception as e:
            self._logger.warning(f"Error calling {method_name}: {e}")
            return None
    
    def _safe_driver_attr(self, attr_name: str, value=None):
        """Safely access or set a driver attribute."""
        if not self._driver:
            raise StateError("Driver not initialized")
        
        if not hasattr(self._driver, attr_name):
            raise ParameterError(f"Attribute {attr_name} not available for {self.instrument_type}")
        
        if value is not None:
            setattr(self._driver, attr_name, value)
        else:
            return getattr(self._driver, attr_name)

    # ========== STANDARDIZED CONFIGURATION METHODS ==========
    
    def configure_measurement(self, **config_params) -> None:
        """
        Configure measurement parameters.
        
        Args:
            **config_params: Configuration parameters to set
                - amplitude: Power/amplitude value
                - frequency: Center frequency
                - span: Frequency span
                - start_freq/stop_freq: Frequency range
                - if_bandwidth: IF bandwidth
                - sweep_points: Number of sweep points
                - average_points: Averaging points
                - parameter: Measurement parameter (S21, etc.)
                - data_format: Data array format
                - smoothing: Smoothing factor
                - if_frequency: IF frequency for Alazar
        """
        self._validate_connection()
        
        # Update configuration
        for param, value in config_params.items():
            if hasattr(self.config, param):
                setattr(self.config, param, value)
            else:
                self._logger.warning(f"Unknown configuration parameter: {param}")
        
        # Apply configuration based on instrument type
        try:
            if self.instrument_type == InstrumentType.VNA:
                self._configure_vna()
            elif self.instrument_type == InstrumentType.RF_SOURCE:
                self._configure_rf_source()
            elif self.instrument_type == InstrumentType.ATTENUATOR:
                self._configure_attenuator()
            elif self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
                self._configure_voltage_source()
            elif self.instrument_type == InstrumentType.ALAZAR:
                self._configure_alazar()
            elif self.instrument_type == InstrumentType.AWG:
                self._configure_awg()
            elif self.instrument_type == InstrumentType.DELAY_GENERATOR:
                self._configure_delay_generator()
            elif self.instrument_type == InstrumentType.DATA_CHANNEL:
                self._configure_data_channel()
            elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
                self._configure_if_signal_generator()
            else:
                raise ValueError(f"Unsupported instrument type: {self.instrument_type}")
                
        except Exception as e:
            self._logger.error(f"Failed to configure measurement: {e}")
            raise ParameterError(f"Configuration failed: {e}")
    
    def get_data(self, **kwargs):
        """
        Args:
            **kwargs:
                - channel: Channel number for voltage meter
                - average: Number of averages for voltage meter
                - sleeptime: Sleep time between measurements for voltage meter
        """


        self._validate_connection()

        if self.instrument_type == InstrumentType.VNA:
            return {'mags': self._safe_driver_call('get_data')[0],
                    'phases': self._safe_driver_call('get_data')[1],
                    'freqs': self._safe_driver_call('get_data')[2]}
        
        if self.instrument_type == InstrumentType.VOLTAGE_METER:

            data = self._safe_driver_call('get_voltage',
                                   channel=kwargs.get('channel', 1),
                                   avg=kwargs.get('average', 1),
                                   sleeptime=kwargs.get('sleeptime', 0.05))
            
            self._safe_driver_call('clean_buffer')
            self._logger.info(f"Voltage data retrieved from channel {kwargs.get('channel', 1)}")
            
            return data
        
        if self.instrument_type == InstrumentType.AWG:
            # For AWG, we assume data is already loaded in memory
            if hasattr(self._driver, 'getDataFromSocketBinary'):
                data = self._safe_driver_call('getDataFromSocketBinary')
                if data is not None:
                    self._logger.info("AWG data retrieved from socket")
                    return data
                else:
                    self._logger.warning("No data available in AWG memory")
                    return None

        if self.instrument_type == InstrumentType.ALAZAR:
            """
            Capture data from Alazar device.
            Returns the data as I, Q, and time arrays.
            """
            # Default parameters for Alazar capture
            params = self.config.list_params() # Get current configuration parameters as dict
            for param in ['pre_trigger_samples', 'post_trigger_samples', 'records_per_buffer',
                          'buffers_per_acquisition', 'amplitude_reference_alazar', 'input_load',
                            'trigger_level_volts', 'trigger_range_volts', 'waveform_head_cut',
                            'period', 'buffer_timeout_ms', 'decimation_value', 'ttl_trigger', 'save_data']:
                if param not in params:
                    raise ParameterError(f"Missing required parameter: {param}")
            # Capture data using the driver
            data = self._safe_driver_call('capture',
                                          preTriggerSamples=params['pre_trigger_samples'],
                                          postTriggerSamples=params['post_trigger_samples'],
                                          recordsPerBuffer=params['records_per_buffer'],
                                          buffersPerAcquisition=params['buffers_per_acquisition'],
                                          powerReference_dBm=params['amplitude_reference_alazar'],
                                          inputLoad=params['input_load'],
                                          triggerLevel_volts=params['trigger_level_volts'],
                                          triggerRange_volts=params['trigger_range_volts'],
                                          waveformHeadCut=params['waveform_head_cut'],
                                          period=params['period'],
                                          buffer_timeout_ms=params['buffer_timeout_ms'],
                                          decimation_value=params['decimation_value'],
                                          TTL=params['ttl_trigger'],
                                          save=params['save_data'])
            if data is not None:
                self._logger.info("Alazar data captured successfully")

            return {'I': data[0],
                    'Q': data[1]}

    def get_error(self) -> None:
        """Get the last error message from the instrument."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.RF_SOURCE:
            error = self._safe_driver_call('get_error')
            if error:
                self._logger.error(f"RF Source error: {error}")
        elif self.instrument_type == InstrumentType.AWG:
            error = self._safe_driver_call('getError')
            if error:
                self._logger.error(f"AWG error: {error}")
        elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
            error = self._safe_driver_call('get_error')
            if error:
                self._logger.error(f"IF Signal Generator error: {error}")
        else:
            self._logger.warning("get_error not implemented for this instrument type")
        
    def clear_error(self) -> None:
        """Clear the last error message from the instrument."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.RF_SOURCE:
            self._safe_driver_call('clear_error')
            self._logger.info("RF Source error cleared")
        elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
            self._safe_driver_call('clear_errors')
            self._logger.info("IF Signal Generator errors cleared")
        else:
            self._logger.warning("clear_error not implemented for this instrument type")
        

    def _configure_vna(self):
        """Configure VNA-specific parameters."""
        
        # Apply settings to driver
        self._safe_driver_call('set_amplitude', self.config.amplitude)
        self._safe_driver_attr('parameter', self.config.parameter)
        self._safe_driver_attr('span_frequency', self.config.span_frequency)
        self._safe_driver_attr('start_frequency', self.config.start_frequency)
        self._safe_driver_attr('stop_frequency', self.config.stop_frequency)
        self._safe_driver_attr('center_frequency', self.config.center_frequency)
        self._safe_driver_attr('if_bandwidth_frequency', self.config.if_bandwidth)
        self._safe_driver_attr('sweep_points', self.config.sweep_points)
        self._safe_driver_attr('average_points', self.config.average_points)
        self._safe_driver_attr('averaging', self.config.averaging_enabled)
        self._safe_driver_attr('data_array_format', self.config.data_array_format)
        self._safe_driver_attr('smooth', self.config.smoothing)
        if self.config.autoscale is not None:
            self._safe_driver_call('autoscale', self.config.autoscale)

    def _configure_rf_source(self):
        """Configure RF source-specific parameters."""
        self._safe_driver_call('set_amplitude', self.config.rf_amplitude)
        self._safe_driver_call('set_frequency', self.config.rf_frequency)
        self._safe_driver_call('start_mod' if self.config.rf_mod else 'stop_mod')
        self._safe_driver_call('set_pulse_trigger_source', self.config.pulse_trigger_source)
        self._safe_driver_call('set_phase', self.config.rf_phase if hasattr(self.config, 'rf_phase') else 0)
    
    def _configure_attenuator(self):
        """Configure attenuator-specific parameters."""
        self._safe_driver_call('set_attenuation', self.config.attenuation)
    
    def _configure_voltage_source(self):
        """Configure voltage source-specific parameters."""
        self._safe_driver_call('ramp_voltage', self.config.voltage)
    
    def _configure_alazar(self):
        """Configure Alazar-specific parameters."""
        # The ATS9872_driver has a simple interface with only a capture method
        # It doesn't have configuration methods, so we just store the parameters
        # for use during capture operations
        
        # No need to call any methods, just store the configuration values
        # These will be passed directly to the capture method when needed
        self._logger.info("Alazar configuration parameters stored for use during capture")
        
        # Note: Not actually configuring the hardware here, as ATS9872_driver 
        # doesn't provide configuration methods - all parameters are passed
        # directly to the capture method when it's called
    
    def _configure_awg(self):
        """Configure AWG-specific parameters."""
        # The M8195A_driver has specific methods different from generic ones
        # We'll use the safe_driver_call to ensure we only call methods that exist
        if not self.init_flag:
            if hasattr(self._driver, 'setRefInClockExternal'):
                self._safe_driver_call('setRefInClockExternal')
                self.init_flag = True

        if hasattr(self._driver, 'freeMemory'):
            self._safe_driver_call('freeMemory')
                
        # Set sample rate if it's been changed and method exists
        if hasattr(self._driver, 'setSampleRate') and hasattr(self.config, 'sample_rate'):
            self._safe_driver_call('setSampleRate', self.config.sample_rate)
        
        # Set memory division if method exists
        if hasattr(self._driver, 'setMemoryDivision') and hasattr(self.config, 'memory_division'):
            # Convert string value to int if needed
            division_value = self.config.memory_division
            if isinstance(division_value, int):
                self._safe_driver_call('setMemoryDivision', division_value)
        
        # Configure channels if defined in config
        if hasattr(self.config, 'channel_setting'):
            for channel, is_enabled in self.config.channel_setting.items():
                if is_enabled and hasattr(self._driver, 'enableChanneloutput'):
                    self._safe_driver_call('enableChanneloutput', channel)
                elif not is_enabled and hasattr(self._driver, 'disableChanneloutput'):
                    self._safe_driver_call('disableChanneloutput', channel)
                
        # Set voltage if defined
        if hasattr(self.config, 'channel_voltage') and hasattr(self._driver, 'setVoltage'):
            for channel, voltage in self.config.channel_voltage.items():
                self._safe_driver_call('setVoltage', channel, voltage)

        # Set voltage offset if defined
        if hasattr(self.config, 'channel_voltage_offset') and hasattr(self._driver, 'setVoltageOffset'):
            for channel, voltage_offset in self.config.channel_voltage_offset.items():
                self._safe_driver_call('setVoltageOffset', channel, voltage_offset)
        
        if hasattr(self._driver, 'clockFrequency'):
            # Set reference clock frequency if method exists
            self._safe_driver_call('setRefInClockFrequency', self.config.ref_clock_frequency)

        self._logger.info("AWG configuration complete")

    def _configure_delay_generator(self):
        """Configure Delay Generator-specific parameters."""
        if hasattr(self._driver, 'set_trigger_rate'):
            self._safe_driver_call('set_trigger_rate', self.config.trigger_rate)
        
        if hasattr(self._driver, 'set_trigger_source'):
            self._safe_driver_call('set_trigger_source', self.config.trigger_source)
        
        if hasattr(self._driver, 'set_delay_channels'):
            self._safe_driver_call('set_delay_channels', self.config.delay_channels)
        
        self._logger.info("Delay Generator configuration complete")
    
    def _configure_data_channel(self):
        """Configure Data Channel-specific parameters."""
        self._safe_driver_call('clearAwgChannel')
        for name, (channel, marker) in self.config.channel_config.items():
            print(f"Configuring channel {channel} with name {name} and marker {marker}")
            self._safe_driver_call('labelAwgChannel', channel=channel, 
                                    channelName=name, markerValue=marker)

        if hasattr(self._driver, 'prepareChannelData'):
            if self.init_flag:
                total_experiment_duration = self.config.total_experiment_duration
                for channel in self.config.pulse_sequence.channels.keys():
                    total_experiment_duration = max(total_experiment_duration,
                                                    self.config.pulse_sequence.get_totallength(channel))
                self._safe_driver_call('prepareChannelData',
                                    sequence=self.config.pulse_sequence,
                                    totalExperimentDuration=total_experiment_duration)
            
        
        if hasattr(self._driver, 'loadChannelDataToAwg'):
            if self.init_flag:
                self._safe_driver_call('allocAwgMemory')
                for channel_name in self.config.pulse_sequence.channels.keys():
                    print(f"Loading channel data for {channel_name}")
                    self._safe_driver_call('loadChannelDataToAwg', channel_name)
                self._safe_driver_call('setInstrumentsMarker', 1)
            
            self.config.channel_data = self._driver._channelData  # Update channel data from DataChannelManager
        
        if hasattr(self._driver, 'get_error'):
            error = self._safe_driver_call('get_error')
            if error:
                self.state = InstrumentState.ERROR
                raise ConnectionError(f"Failed to configure the Data Channel: {e}") 
        
        self.init_flag = True                         
        self._logger.info("Data Channel configuration complete")

    def _configure_if_signal_generator(self):
        """Configure IF Signal Generator-specific parameters."""
        if hasattr(self._driver, 'rf_on'):
            self._safe_driver_call('is_rf_on')
        
        if hasattr(self._driver, 'set_frequency'):
            self._safe_driver_call('set_frequency', self.config.rf_frequency)
        
        if hasattr(self._driver, 'set_level'):
            self._safe_driver_call('set_level', self.config.rf_level)
        
        if hasattr(self._driver, 'set_mod_source'):
            self._safe_driver_call('set_mod_source', self.config.rf_mod)
        
        if hasattr(self._driver, 'mod_on') and self.config.rf_mod_on:
            self._safe_driver_call('mod_on')
        elif hasattr(self._driver, 'mod_off') and not self.config.rf_mod_on:
            self._safe_driver_call('mod_off')
        
        self._logger.info("IF Signal Generator configuration complete")



    def setup_awg_pulse_sequence(self, data_manager, pulse_sequence, 
                            total_experiment_duration: float, 
                                pulse_channel: str = 'm'):
        """
        Setup AWG pulse sequence using DataChannelManager.
        
        Args:
            data_manager: DataChannelManager instance
            pulse_sequence: PulseSequence instance
            total_experiment_duration: Total experiment duration in seconds
            pulse_channel: Channel name for pulse ('m' for marker)
        """
        if self.instrument_type != InstrumentType.AWG:
            raise ParameterError("AWG pulse sequence setup only available for AWG")
        
        self._validate_connection()
        
        try:
            # Clear and setup AWG channels
            if hasattr(data_manager, 'clearAwgChannel'):
                data_manager.clearAwgChannel()
            
            # Label AWG channels (standard configuration for I/Q + marker)
            if hasattr(data_manager, 'labelAwgChannel'):
                data_manager.labelAwgChannel(channel=2, channelName='Q', markerValue=1)
                data_manager.labelAwgChannel(channel=1, channelName='I', markerValue=1)
                data_manager.labelAwgChannel(channel=1, channelName=pulse_channel, markerValue=2)
            
            # Prepare channel data
            if hasattr(data_manager, 'prepareChannelData'):
                data_manager.prepareChannelData(
                    sequence=pulse_sequence,
                    totalExperimentDuration=total_experiment_duration
                )
            
            # Allocate and load memory
            if hasattr(self._driver, 'freeMemory'):
                self._safe_driver_call('freeMemory')
            
            if hasattr(data_manager, 'allocAwgMemory'):
                data_manager.allocAwgMemory()
            
            if hasattr(data_manager, 'loadChannelDataToAwg'):
                data_manager.loadChannelDataToAwg(pulse_channel)
            
            if hasattr(data_manager, 'setInstrumentsMarker'):
                data_manager.setInstrumentsMarker(1)
            
            self._logger.info("AWG pulse sequence setup completed")
            
        except Exception as e:
            self._logger.error(f"AWG pulse sequence setup failed: {e}")
            raise ParameterError(f"AWG pulse sequence setup failed: {e}")

    
    def set_idle(self) -> None:
        """
        Set instrument to idle state.
        
        This method safely puts the instrument into an idle state,
        powering down outputs and stopping measurements.
        """
        try:
            # Set instrument to idle based on type
            if self.instrument_type == InstrumentType.VNA:
                if hasattr(self._driver, 'power'):
                    self._safe_driver_attr('power', 0)
                if hasattr(self._driver, 'clear_averaging'):
                    self._safe_driver_call('clear_averaging')
            
            elif self.instrument_type == InstrumentType.RF_SOURCE:
                if hasattr(self._driver, 'stop_rf'):
                    self._safe_driver_call('stop_rf')
                if hasattr(self._driver, 'stop_mod'):
                    self._safe_driver_call('stop_mod')
            
            elif self.instrument_type == InstrumentType.AWG:
                if hasattr(self._driver, 'stop'):
                    self._safe_driver_call('stop')
            
            elif self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
                if hasattr(self._driver, 'ramp_voltage'):
                    self._safe_driver_call('ramp_voltage', 0.0)
                if hasattr(self._driver, 'turn_off'):
                    self._safe_driver_call('turn_off')

            elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
                # if hasattr(self._driver, 'rf_off'):
                #     self._safe_driver_call('rf_off')
                # if hasattr(self._driver, 'mod_off'):
                #     self._safe_driver_call('mod_off')
                pass

            # Update state
            self.state = InstrumentState.IDLE
            self._logger.info(f"{self.instrument_type} set to idle state")
            
        except Exception as e:
            self._logger.warning(f"Failed to set {self.instrument_type} to idle: {e}")
            # Don't raise exception during cleanup

    def configure_power_sweep_awg(self, 
                                 awg_sample_rate: float,
                                 awg_ref_clock: float,
                                 rf_measurement_amplitude: float,
                                 rf_measurement_frequency: float,
                                 records_per_buffer: int,
                                 buffers_per_acquisition: int,
                                 amplitude_reference_alazar: float,
                                 trigger_level_volts: float,
                                 trigger_range_volts: float,
                                 waveform_head_cut: int,
                                 period: int,
                                 buffer_timeout_ms: int,
                                 decimation_value: int,
                                 rf_measurement_length: float):
        """
        Configure all parameters for power sweep AWG measurement.
        
        Args:
            awg_sample_rate: AWG sampling rate in Hz
            awg_ref_clock: Reference clock frequency in Hz
            rf_measurement_amplitude: RF measurement amplitude in dBm
            rf_measurement_frequency: RF measurement frequency in Hz
            records_per_buffer: Alazar records per buffer
            buffers_per_acquisition: Alazar buffers per acquisition
            amplitude_reference_alazar: Alazar amplitude reference in dBm
            trigger_level_volts: Trigger level in Volts
            trigger_range_volts: Trigger range in Volts
            waveform_head_cut: Waveform head cut points
            period: Period for processing
            buffer_timeout_ms: Buffer timeout in milliseconds
            decimation_value: Decimation value
            rf_measurement_length: RF measurement pulse length in seconds
        """
        self._validate_connection()
        
        # Configure based on instrument type
        if self.instrument_type == InstrumentType.AWG:
            # Configure AWG-specific parameters
            channels_config = {
                1: {"voltage": 1.0, "offset": 0.0},  # I channel
                2: {"voltage": 1.0, "offset": 0.0},  # Q channel
                3: {"voltage": 1.0, "offset": 0.5},  # Marker 1
                4: {"voltage": 1.0, "offset": 0.5}   # Marker 2
            }
            self.configure_awg_advanced(awg_sample_rate, awg_ref_clock, channels_config)
            
        elif self.instrument_type == InstrumentType.RF_SOURCE:
            # Configure RF source for AWG operation
            self.configure_measurement(
                rf_amplitude=rf_measurement_amplitude,
                rf_frequency=rf_measurement_frequency
            )
            
        elif self.instrument_type == InstrumentType.ALAZAR:
            # Calculate sampling parameters
            default_sampling_rate = 1e9
            sampling_rate = default_sampling_rate / decimation_value
            post_trigger_samples = int(rf_measurement_length * sampling_rate / 256) * 256
            
            # Configure Alazar parameters
            self.configure_measurement(
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
                ttl_trigger=True,
                save_data=False
            )
    
    
    
    def power_on(self):
        """Turn on the instrument power/output."""
        if self.instrument_type == InstrumentType.VNA:
            self._safe_driver_attr('power', 1)
        elif self.instrument_type == InstrumentType.RF_SOURCE:
            self._safe_driver_call('start_rf')
        elif self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
            self._safe_driver_call('turn_on')
        elif self.instrument_type == InstrumentType.AWG:
            self._safe_driver_call('start')
        elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
            self._safe_driver_call('rf_on')
        else:
            self._logger.warning(f"power_on not implemented for {self.instrument_type}")



    def power_off(self):
        """Turn off instrument power/output."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.VNA:
            self._safe_driver_attr('power', 0)
        elif self.instrument_type == InstrumentType.RF_SOURCE:
            self._safe_driver_call('stop_rf')
            self._safe_driver_call('stop_mod')
        elif self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
            self._safe_driver_call('turn_off')
        elif self.instrument_type == InstrumentType.AWG:
            self._safe_driver_call('stop')
        elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
            self._safe_driver_call('rf_off')
        else:
            self._logger.warning(f"power_off not implemented for {self.instrument_type}")
    
    def clear_averaging(self):
        """Clear averaging for VNA instruments."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.VNA:
            self._safe_driver_call('clear_averaging')
        else:
            self._logger.warning(f"clear_averaging not available for {self.instrument_type}")
    
    def set_frequency(self, frequency: float):
        """Set frequency for RF instruments."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.RF_SOURCE:
            self.configure_measurement(rf_frequency=frequency)
        elif self.instrument_type == InstrumentType.VNA:
            self.configure_measurement(center_frequency=frequency)
        elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
            self._safe_driver_call('set_frequency', frequency)
        else:
            self._logger.warning(f"set_frequency not available for {self.instrument_type}")
    
    def set_power(self, power: float):
        """Set power/amplitude for instruments."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.RF_SOURCE:
            self.configure_measurement(rf_amplitude=power)
        elif self.instrument_type == InstrumentType.VNA:
            self.configure_measurement(amplitude=power)
        else:
            self._logger.warning(f"set_power not available for {self.instrument_type}")
    
    def set_voltage(self, voltage: float):
        """Set voltage for voltage source instruments."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
            self.configure_measurement(voltage=voltage)
        else:
            self._logger.warning(f"set_voltage not available for {self.instrument_type}")
    
    def set_attenuation(self, attenuation: float):
        """Set attenuation for attenuator instruments."""
        self._validate_connection()
        
        if self.instrument_type == InstrumentType.ATTENUATOR:
            self.configure_measurement(attenuation=attenuation)
        else:
            self._logger.warning(f"set_attenuation not available for {self.instrument_type}")
    
    def cleanup(self):
        """
        Perform comprehensive cleanup of instrument resources.
        
        This method ensures all instruments are safely set to idle state,
        outputs are disabled, and connections are closed properly.
        It should be called at the end of measurements to ensure proper
        resource management.
        """
        try:
            # First set the instrument to idle state
            self.set_idle()
            gc.collect()  # Run garbage collection to free resources
            
            # Specific cleanup actions by instrument type
            if self.instrument_type == InstrumentType.VNA:
                # Turn off RF output
                if hasattr(self._driver, 'power'):
                    self._safe_driver_attr('power', 0)
            
            elif self.instrument_type == InstrumentType.RF_SOURCE:
                # Stop RF output and modulation
                if hasattr(self._driver, 'stop_rf'):
                    self._safe_driver_call('stop_rf')
                if hasattr(self._driver, 'stop_mod'):
                    self._safe_driver_call('stop_mod')
            
            elif self.instrument_type == InstrumentType.AWG:
                # Stop AWG output
                if hasattr(self._driver, 'stop'):
                    self._safe_driver_call('stop')
                # Free memory if available
                if hasattr(self._driver, 'freeMemory'):
                    self._safe_driver_call('freeMemory')
            
            elif self.instrument_type == InstrumentType.VOLTAGE_SOURCE:
                # Set voltage to zero and turn off output
                if hasattr(self._driver, 'set_voltage'):
                    self._safe_driver_call('set_voltage', 0.0)
                if hasattr(self._driver, 'power_off'):
                    self._safe_driver_call('power_off')
            
            elif self.instrument_type == InstrumentType.ALAZAR:
                pass 
            
            elif self.instrument_type == InstrumentType.ATTENUATOR:
                # Set attenuation to default (0 dB)
                if hasattr(self._driver, 'set_attenuation'):
                    self._safe_driver_call('set_attenuation', 0.0)
            
            elif self.instrument_type == InstrumentType.IF_SIGNAL_GENERATOR:
                # Stop RF output
                # if hasattr(self._driver, 'rf_off'):
                #     self._safe_driver_call('rf_off')
                # if hasattr(self._driver, 'mod_ff'):
                #     self._safe_driver_call('mod_off')
                pass
                  
            if hasattr(self._driver, 'close'):
                self._safe_driver_call('close')

            self.state = InstrumentState.IDLE
            self._logger.info(f"Cleanup completed for {self.instrument_type}")
            
        except Exception as e:
            self._logger.warning(f"Cleanup failed for {self.instrument_type}: {e}")
            # Don't raise exception during cleanup
    
    def create_data_channel_manager(self):
        """
        Create a DataChannelManager instance for this AWG handler.
        
        Returns:
            DataChannelManager instance or None if not available
        """
        if self.instrument_type != InstrumentType.AWG:
            self._logger.warning("DataChannelManager only available for AWG instruments")
            return None
            
        self._validate_connection()
        
        try:
            # Try to import DataChannelManager
            from measurements.DataChannelManager import DataChannelManager
            
            # Create and return the manager
            data_manager = DataChannelManager(self._driver)
            self._logger.info("Created DataChannelManager for AWG")
            return data_manager
            
        except ImportError:
            self._logger.warning("DataChannelManager not available in environment")
            return None
        except Exception as e:
            self._logger.error(f"Failed to create DataChannelManager: {e}")
            return None