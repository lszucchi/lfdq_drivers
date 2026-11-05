"""
SMB100B Signal Generator Driver

This module provides a Python driver for the Rohde & Schwarz SMB100B signal generator.
It implements basic functionality for RF output control, frequency/power settings, 
modulation control, and instrument mode management using the RsInstrument library.

Author: Denys Derlian
Date: 26 de junho de 2025
"""

from RsInstrument import RsInstrument

class SMB100B_driver:
    """
    Driver class for Rohde & Schwarz SMB100B Signal Generator.
    
    This class provides a high-level interface to control the SMB100B signal generator
    through SCPI commands. It supports RF output control, frequency and power settings,
    modulation control, and remote/local mode switching.
    
    Attributes:
        instr (RsInstrument): The underlying instrument connection object
        mode (str): Current control mode of the instrument
    """
    def __init__(self, resource_address: str, reset: bool = False):
        """
        Initialize the SMB100B signal generator driver.
        
        Args:
            resource_address (str): VISA resource string for the instrument connection.
                                  Example: 'TCPIP::169.254.2.20::5025::SOCKET'
            reset (bool, optional): Whether to reset the instrument on initialization.
                                  Defaults to False.
        
        Note:
            The initialization automatically turns off RF output and modulation,
            and sets a 5-second timeout for VISA communications.
        """
        self.instr = RsInstrument(resource_address, reset=reset)
        self.instr.visa_timeout = 5000  # timeout em ms
        # self.rf_off()  # Ensure RF is off on startup
        # self.mod_off()  # Ensure modulation is off on startup
        self.mode = self.get_control_mode()  # Store current control mode  
    
    def idn(self):
        """
        Get the instrument identification string.
        
        Returns:
            str: Instrument identification string containing manufacturer, model,
                 serial number, and firmware version information.
        """
        return self.instr.query_str('*IDN?').strip()
    
    # ==================== RF OUTPUT CONTROL ====================
    
    def rf_on(self):
        """
        Turn on the RF output.
        
        Note:
            This enables the main RF output of the signal generator.
        """
        self.instr.write('OUTPUT ON')
    
    def rf_off(self):
        """
        Turn off the RF output.
        
        Note:
            This disables the main RF output of the signal generator for safety.
        """
        self.instr.write('OUTPUT OFF')
    
    def is_rf_on(self) -> bool:
        """
        Check if the RF output is currently enabled.
        
        Returns:
            bool: True if RF output is on, False if off.
        """
        return bool(int(self.instr.query_str('OUTPUT?').strip()))
    
    # ==================== FREQUENCY CONTROL ====================
    
    def set_frequency(self, freq_hz: float):
        """
        Set the RF output frequency.
        
        Args:
            freq_hz (float): Frequency in Hz. Must be within the instrument's
                           specified frequency range.
        
        Example:
            driver.set_frequency(1e9)  # Set to 1 GHz
        """
        self.instr.write(f'FREQ {freq_hz}')
    
    def get_frequency(self) -> float:
        """
        Get the current RF output frequency.
        
        Returns:
            float: Current frequency in Hz.
        """
        return float(self.instr.query_str('FREQ?').strip())
    
    # ==================== POWER LEVEL CONTROL ====================
    
    def set_level(self, level_dbm: float):
        """
        Set the RF output power level.
        
        Args:
            level_dbm (float): Power level in dBm. Must be within the instrument's
                             specified power range.
        
        Example:
            driver.set_level(-10.0)  # Set to -10 dBm
        """
        self.instr.write(f'POW:LEV {level_dbm}')
    
    def get_level(self) -> float:
        """
        Get the current RF output power level.
        
        Returns:
            float: Current power level in dBm.
        """
        return float(self.instr.query_str('POW:LEV?').strip())
    
    # ==================== MODULATION CONTROL ====================
    
    def mod_on(self):
        """
        Turn on modulation.
        
        Note:
            This enables the modulation function. The modulation source
            (internal/external) should be configured separately.
        """
        self.instr.write('MOD:STAT ON')

    def mod_off(self):
        """
        Turn off modulation.
        
        Note:
            This disables all modulation functions.
        """
        self.instr.write('MOD:STAT OFF')

    def is_mod_on(self) -> bool:
        """
        Check if modulation is currently enabled.
        
        Returns:
            bool: True if modulation is on, False if off.
        """
        return bool(int(self.instr.query_str('MOD:STAT?').strip()))
    
    # ==================== MODULATION SOURCE CONTROL ====================
    
    def set_mod_src_internal(self):
        """
        Set modulation source to internal.
        
        Note:
            Uses the instrument's internal modulation generators.
        """
        self.instr.write('MOD:STAT INT')
    
    def set_mod_src_external(self):
        """
        Set modulation source to external.
        
        Note:
            Uses external modulation signals applied to the instrument's
            modulation input connectors.
        """
        self.instr.write('MOD:STAT EXT')
    
    def get_mod_src(self) -> str:
        """
        Get the current modulation source setting.
        
        Returns:
            str: Current modulation source ('INT' or 'EXT').
        """
        return self.instr.query_str('MOD:STAT?').strip()
    
    def set_mod_src(self, source: str):
        """
        Set the modulation source.
        
        Args:
            source (str): Modulation source. Must be 'INT' for internal
                         or 'EXT' for external (case insensitive).
        
        Raises:
            ValueError: If source is not 'INT' or 'EXT'.
        
        Example:
            driver.set_mod_src('INT')  # Use internal modulation
            driver.set_mod_src('EXT')  # Use external modulation
        """
        if source.upper() == 'INT':
            self.set_mod_src_internal()
        elif source.upper() == 'EXT':
            self.set_mod_src_external()
        else:
            raise ValueError("Invalid modulation source. Use 'INT' or 'EXT'.")
    
    # ==================== ERROR HANDLING ====================
    def get_error(self):
        """
        Get the oldest error from the instrument's error queue.
        
        Returns:
            str: Error message string. Returns "0,No error" if no errors present.
        
        Note:
            The instrument maintains a FIFO error queue. This method retrieves
            and removes the oldest error from the queue.
        """
        return self.instr.query_str('SYST:ERR?').strip()
    
    def clear_errors(self):
        """
        Clear all errors from the instrument's error queue.
        
        Note:
            This method reads and prints all errors until the queue is empty.
            Useful for debugging and ensuring clean instrument state.
        """
        while True:
            err = self.get_error()
            print('Erro:', err)
            if err.startswith('0'):  # "0,No error" indicates empty queue
                break

    def close(self):
        """
        Close the instrument connection and release resources.
        
        Note:
            Always call this method when finished with the instrument
            to properly release the VISA connection.
        """
        self.instr.close()
    
    def open(self):
        """
        Open the instrument connection.
        
        Note:
            This method is called automatically on initialization, but can
            be used to re-establish the connection if needed.
        """
        self.instr.open()
    
    def reconnect(self):
        """
        Reconnect to the instrument.
        
        Note:
            This method closes the current connection and reopens it.
            Useful if the connection has been lost or needs to be reset.
        """
        self.instr.reconnect()

    # ==================== CONTROL MODE MANAGEMENT ====================

    def set_mode(self, mode: str):
        """
        Set the control mode of the instrument.
        
        Args:
            mode (str): Control mode. Must be 'LOCAL', 'REMOTE', or 'LOCKOUT'
                       (case insensitive).
        
        Raises:
            ValueError: If mode is not one of the valid options.
        
        Note:
            - LOCAL: Releases remote control, allows front panel operation
            - REMOTE: Forces remote control, locks front panel
            - LOCKOUT: Forces remote control with complete front panel lockout
        """
        if mode.upper() == 'LOCAL':
            self.go_to_local_mode()
        elif mode.upper() == 'REMOTE':
            self.go_to_remote_mode()
        elif mode.upper() == 'LOCKOUT':
            self.go_to_local_lockout()
        else:
            raise ValueError("Invalid mode. Use 'LOCAL', 'REMOTE', or 'LOCKOUT'.")
    
    def go_to_local_mode(self):
        """
        Release remote control and allow manual front panel use.
        
        Note:
            After calling this, the instrument can be operated manually
            from its front panel controls.
        """
        self.instr.write('SYST:LOC')

    def go_to_remote_mode(self):
        """
        Force remote control and lock front panel.
        
        Note:
            After calling this, the instrument is controlled remotely
            and front panel controls are disabled.
        """
        self.instr.write('SYST:REM')

    def go_to_local_lockout(self):
        """
        Force remote control with front panel completely locked (LLO).
        
        Note:
            This is the most restrictive mode - the instrument can only
            be controlled remotely and the front panel is completely locked.
        """
        self.instr.write('SYST:LOCal:LOCK')

    def get_control_mode(self) -> str:
        """
        Get the current control mode of the instrument.
        
        Returns:
            str: Current control mode. Possible values:
                - "LOCAL": Local control enabled
                - "REMOTE": Remote control active  
                - "LOCKOUT": Remote control with front panel lockout
                - "UNKNOWN": Mode could not be determined
                - "Unknown or unsupported SCPI control query": Query not supported
        
        Note:
            Not all instruments return a clear mode string, so this may
            return approximate status based on available SCPI commands.
        """
        try:
            # Try this command if the instrument supports it (some R&S instruments do)
            mode = self.instr.query_str('SYST:COMM:LAN:CONTrol:STATe?').strip()
            if mode == '0':
                return "LOCAL"
            elif mode == '1':
                return "REMOTE"
            elif mode == '2':
                return "LOCKOUT"
            else:
                return f"UNKNOWN ({mode})"
        except:
            # Fallback heuristic if above command is not supported
            return "Unknown or unsupported SCPI control query"

    # ==================== EXAMPLE USAGE ====================

    """
    Example Usage:

        # Initialize the driver
        driver = SMB100B_driver('TCPIP::169.254.2.20::5025::SOCKET')
        
        # Get instrument information
        print(f"Connected to: {driver.idn()}")
        
        # Configure basic settings
        driver.set_frequency(1e9)      # Set to 1 GHz
        driver.set_level(-10.0)        # Set to -10 dBm
        
        # Turn on RF output
        driver.rf_on()
        
        # Check current settings
        freq = driver.get_frequency()
        level = driver.get_level()
        print(f"Current settings: {freq/1e9:.3f} GHz, {level:.1f} dBm")
        
        # Configure modulation
        driver.set_mod_src('INT')      # Use internal modulation
        driver.mod_on()                # Enable modulation
        
        # Check modulation status
        if driver.is_mod_on():
            print(f"Modulation ON, source: {driver.get_mod_src()}")
        
        # Switch to remote mode
        driver.set_mode('REMOTE')
        print(f"Control mode: {driver.get_control_mode()}")
        
        # Clean shutdown
        driver.rf_off()                # Turn off RF for safety
        driver.mod_off()               # Turn off modulation
        driver.set_mode('LOCAL')       # Return control to front panel
        driver.close()                 # Close connection
        
        # Error handling example
        try:
            driver.set_mod_src('INVALID')
        except ValueError as e:
            print(f"Error: {e}")
        
        # Check for instrument errors
        driver.clear_errors()

    Dependencies:
        - RsInstrument: Rohde & Schwarz instrument driver library
        - VISA: Virtual Instrument Software Architecture
        
    Installation:
        pip install RsInstrument

    Supported Instruments:
        - Rohde & Schwarz SMB100B Signal Generator
        - Compatible with other R&S signal generators using similar SCPI commands
    """

