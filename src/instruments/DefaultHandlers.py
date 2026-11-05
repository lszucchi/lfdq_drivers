"""
Default Handlers for Measurement System

This module provides handlers for measurement instruments with automatic
connection verification. Only instruments that are explicitly imported by the user
will have handlers created, and only successfully connected instruments will be available.

Key features:
1. Lazy loading - Handlers are only created when they are imported and accessed
2. Connection verification - Only connected instruments will have non-None handlers
3. Automatic address handling - Addresses are defined within this module

Usage:
    # Import only the handlers you need - no other connections will be attempted
    from defaultHandlers import VNA, RF_SOURCE_1, ATT
    
    # Example usage
    if VNA is not None:
        VNA.configure_measurement(amplitude=-10)
        
    # Clean up when done
    disconnect_all()
"""

from protocols.Handler import Handler, InstrumentType, HandlerError
import sys, types

# Instrument addresses - imported directly into this module for simplicity
VNA_ADDRESS = "TCPIP0::169.254.101.105::inst0::INSTR"
ATT_ADDRESS = "TCPIP0::169.254.101.101::inst0::INSTR"
RF_SOURCE_1_ADDRESS = "TCPIP0::169.254.101.103::inst0::INSTR"
RF_SOURCE_2_ADDRESS = "TCPIP0::169.254.101.104::inst0::INSTR"
VOLTAGE_SOURCE_ADDRESS = "GPIB0::2::INSTR"
VOLTAGE_METER_ADDRESS = "GPIB0::2::INSTR" 
AWG_ADDRESS = '169.254.101.100'
IF_SIGNAL_GENERATOR_ADDRESS = 'TCPIP::169.254.2.20::5025::SOCKET'

# Dictionary to hold all handlers - will only contain handlers that have been explicitly requested
# This allows us to track which handlers have been initialized
handlers = {}

def create_handler(instrument_type, address, source_number=None, **kwargs):
    """
    Attempt to create and connect a handler for a specific instrument.
    
    Args:
        instrument_type: Type of instrument (e.g., "VNA", "RF_SOURCE")
        address: Connection address string
        source_number: Source number for RF sources
        **kwargs: Additional arguments for handler initialization
        
    Returns:
        Handler if successful, None if connection fails
    """
    try:
        # Create handler with auto_connect=True
        if instrument_type != "DATA_CHANNEL":
            handler = Handler(instrument_type, address, source_number=source_number, **kwargs)
        else:
            # DATA_CHANNEL does not require an address or source number
            handler = Handler(instrument_type, None, source_number=None, **kwargs)
        
        # If we reach here, connection was successful
        print(f"Successfully connected to {instrument_type}")
        return handler
    except HandlerError as e:
        print(f"Warning: Could not connect to {instrument_type} at {address}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error connecting to {instrument_type} at {address}: {str(e)}")
        return None

# Define handlers as properties to instantiate them on-demand
class LazyHandlers:
    @property
    def VNA(self):
        if 'VNA' not in handlers:
            handlers['VNA'] = create_handler("VNA", VNA_ADDRESS)
        return handlers.get('VNA')
    
    @property
    def ATT(self):
        if 'ATT' not in handlers:
            handlers['ATT'] = create_handler("ATTENUATOR", ATT_ADDRESS)
        return handlers.get('ATT')
    
    @property
    def RF_SOURCE_1(self):
        if 'RF_SOURCE_1' not in handlers:
            handlers['RF_SOURCE_1'] = create_handler("RF_SOURCE", RF_SOURCE_1_ADDRESS, source_number=1)
        return handlers.get('RF_SOURCE_1')
    
    @property
    def RF_SOURCE_2(self):
        if 'RF_SOURCE_2' not in handlers:
            handlers['RF_SOURCE_2'] = create_handler("RF_SOURCE", RF_SOURCE_2_ADDRESS, source_number=2)
        return handlers.get('RF_SOURCE_2')
    
    @property
    def VOLTAGE_SOURCE(self):
        if 'VOLTAGE_SOURCE' not in handlers:
            handlers['VOLTAGE_SOURCE'] = create_handler("VOLTAGE_SOURCE", VOLTAGE_SOURCE_ADDRESS)
        return handlers.get('VOLTAGE_SOURCE')
    
    @property
    def VOLTAGE_METER(self):
        if 'VOLTAGE_METER' not in handlers:
            handlers['VOLTAGE_METER'] = create_handler("VOLTAGE_METER", VOLTAGE_METER_ADDRESS)
        return handlers.get('VOLTAGE_METER')
    
    @property
    def AWG(self):
        if 'AWG' not in handlers:
            handlers['AWG'] = create_handler("AWG", AWG_ADDRESS)
            handlers['DATA_CHANNEL'] = Handler("DATA_CHANNEL", awg = handlers['AWG']._driver, address = None)
        return handlers.get('AWG')
    
    @property
    def DATA_CHANNEL(self):
        # Ensure AWG is initialized first since DATA_CHANNEL depends on it
        if 'DATA_CHANNEL' not in handlers:
            if 'AWG' not in handlers:
                _ = self.AWG  # This will initialize both AWG and DATA_CHANNEL
            # If DATA_CHANNEL still doesn't exist (AWG might have failed)
            if 'DATA_CHANNEL' not in handlers and 'AWG' in handlers and handlers['AWG'] is not None:
                handlers['DATA_CHANNEL'] = Handler("DATA_CHANNEL", awg=handlers['AWG']._driver, address=None)
            print("Successfully created DATA_CHANNEL handler")
        return handlers.get('DATA_CHANNEL')
    
    @property
    def ALAZAR(self):
        if 'ALAZAR' not in handlers:
            handlers['ALAZAR'] = create_handler("ALAZAR", None)
        return handlers.get('ALAZAR')
    
    @property
    def IF_SIGNAL_GENERATOR(self):
        if 'IF_SIGNAL_GENERATOR' not in handlers:
            handlers['IF_SIGNAL_GENERATOR'] = create_handler("IF_SIGNAL_GENERATOR", IF_SIGNAL_GENERATOR_ADDRESS)
        return handlers.get('IF_SIGNAL_GENERATOR')
        

# Create a single instance of LazyHandlers to expose as module attributes
_lazy_handlers = LazyHandlers()

# Instead of directly accessing the properties, we'll use a descriptor approach
# This ensures that the actual handler is only created when imported and accessed
class LazyDescriptor:
    def __init__(self, handler_name):
        self.handler_name = handler_name
        
    def __get__(self, instance, owner=None):
        # Get the property from the LazyHandlers instance
        return getattr(_lazy_handlers, self.handler_name)

# Export the handlers as module-level lazy descriptors
VNA = LazyDescriptor('VNA')
ATT = LazyDescriptor('ATT')
RF_SOURCE_1 = LazyDescriptor('RF_SOURCE_1')
RF_SOURCE_2 = LazyDescriptor('RF_SOURCE_2')
VOLTAGE_SOURCE = LazyDescriptor('VOLTAGE_SOURCE')
VOLTAGE_METER = LazyDescriptor('VOLTAGE_METER')
AWG = LazyDescriptor('AWG')
ALAZAR = LazyDescriptor('ALAZAR')
DATA_CHANNEL = LazyDescriptor('DATA_CHANNEL')
IF_SIGNAL_GENERATOR = LazyDescriptor('IF_SIGNAL_GENERATOR')

def disconnect_all():
    """
    Disconnect all connected handlers.
    """
    for name, handler in handlers.items():
        if handler is not None:
            try:
                print(f"Disconnecting {name}...")
                handler.cleanup()
                handler.disconnect()
            except Exception as e:
                print(f"Error disconnecting {name}: {str(e)}")

def reconnect_all():
    """
    Attempt to reconnect all disconnected handlers.
    """
    for name, handler in handlers.items():
        if handler is not None and not handler.is_connected():
            try:
                print(f"Reconnecting {name}...")
                handler.connect()
                print(f"Successfully reconnected to {name}")
            except Exception as e:
                print(f"Error reconnecting to {name}: {str(e)}")

def get_connected_handlers():
    """
    Return a dictionary of all connected handlers.
    
    Note: This only returns handlers that have already been initialized.
    It does not attempt to connect to any new instruments.
    
    Returns:
        Dictionary with handler names as keys and handler objects as values
    """
    return {name: handler for name, handler in handlers.items() if handler is not None and handler.is_connected()}


class _ModuleWrapper(types.ModuleType):
    def __getattr__(self, name):
        if hasattr(_lazy_handlers, name):
            return getattr(_lazy_handlers, name)
        elif name in ('get_connected_handlers', 'disconnect_all', 'reconnect_all'):
            return globals()[name]
        raise AttributeError(f"No such handler: {name}")

    def __dir__(self):
        return dir(_lazy_handlers) + ['get_connected_handlers', 'disconnect_all', 'reconnect_all']

    

# Replace the module with the wrapper
sys.modules[__name__] = _ModuleWrapper(__name__)
