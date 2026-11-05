from instruments.VisaInstrument import *
from time import sleep
import numpy as np
import pandas as pd


class E5063A_driver(VisaInstrument):
    def __init__(self,address):
        super().__init__(address)
        self._average_points = self.average_points
        self._center_freq = self.center_frequency
        self._start_freq = self.start_frequency
        self._stop_freq = self.stop_frequency
        self._span_freq = self.span_frequency
        self._sweep_points = self.sweep_points
        self._if_bandwidth = self.if_bandwidth_frequency
        self._data_format = self.data_format

    def autoscale(self,trace_idx):
        '''This command executes the auto scale function. The Auto Scale function automatically adjusts the value of the reference division line and the scale per division to display the trace appropriately.'''
        self.write(':DISPlay:WINDow1:TRACe'+str(trace_idx)+':Y:SCALe:AUTO')

    def scale(self,trace_idx,value):
        self.write(':DISP:WIND:TRAC'+str(trace_idx)+':Y:PDIV '+str(value))

    def set_amplitude(self,amp):
        '''Set amplitude power in dBm'''

        if amp > 0 or amp < -20:
            raise ValueError("amp must be below or equal to 0 dBm and above or equal to -20 dBm")

        self.write(':SOUR1:POWER {}'.format(amp))

    def get_amplitude(self):
        return float(self.query(':SOUR1:POWER?').split('\n')[0])

    @property
    def sweep_time(self):
        '''This command gets the sweep time.

            Parameter: Value
            Description: Sweep time
            Data Type: Numeric type (Real)
            Range: 0 ~ 2M
            Preset Value: 24.7228364697 ms
            Unit: s
            Resolution: -
        '''
        return float(self.query(":SENSe1:SWEep:TIME:DATA?").split('\n')[0])

    @property
    def smooth(self):
        '''This command turns ON/OFF the smoothing, for the active trace of the selected channel Ch.

            Parameter: Selection Option
            Description: ON/OFF of the smoothing
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: OFF
        '''
        return self.query(":CALCulate1:SELected:SMOothing:STATe?").split("\n")[0]

    @smooth.setter
    def smooth(self,state):
        '''This command turns ON/OFF the smoothing, for the active trace of the selected channel Ch.

            Parameter: Selection Option
            Description: ON/OFF of the smoothing
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: OFF
        '''
        self.write(":CALCulate1:SELected:SMOothing:STATe {}".format(state))


    @property
    def power(self):
        '''This command turns ON/OFF of the stimulus signal output. Measurement cannot be made until the stimulus signal output is turned ON.

            Parameter: Selection Option
            Description: Status of the signal output
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: ON
        '''
        return self.query('OUTPut:STATe?').split('\n')[0]
    @power.setter
    def power(self,state):
        '''This command turns ON/OFF of the stimulus signal output. Measurement cannot be made until the stimulus signal output is turned ON.

            Parameter: Selection Option
            Description: Status of the signal output
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: ON
        '''
        self.write('OUTPut:STATe {}'.format(state))


    @property
    def port(self):
        '''This command gets the output port used for absolute.

            Parameter: Value
            Description: Source output port
            Data Type: Numeric type (Integer)
            Range: 1 to 2
            Preset Value: 1
            Note: You need to set the measurement parameter for absolute measurements with the   :CALC:PAR:DEF command.
        '''
        return int(self.query(":CALCulate1:PARameter1:SPORt?").strip('\n'))
    
    @port.setter
    def port(self,p):
        '''This command sets the output port used for absolute.

            Parameter: Value
            Description: Source output port
            Data Type: Numeric type (Integer)
            Range: 1 to 2
            Preset Value: 1
            Note: You need to set the measurement parameter for absolute measurements with the   :CALC:PAR:DEF command.
        '''
        self.write(":CALCulate1:PARameter1:SPORt {}".format(p))

    @property
    def trace_number(self):
         '''This command gets the number of traces. ==
            Parameter: Selection Option
            Description: Number of traces
            Data Type: Numeric type (Integer)
            Range: 1 to 4
            Preset Value: 1
        '''
         return int(self.query(":CALCulate1:PARameter:COUNt?"))
    
    @trace_number.setter
    def trace_number(self, trace_no):
         '''This command gets the number of traces. ==
            Parameter: Selection Option
            Description: Number of traces
            Data Type: Numeric type (Integer)
            Range: 1 to 4
            Preset Value: 1
        '''
         self.write(":CALCulate1:PARameter:COUNt %d" % trace_no)

    @property
    def parameter(self, trace_idx=1):
        '''This command gets the measurement parameter. :CALC:PAR:SPOR specifies the source output port for the absolute measurements.

            Parameter: Selection Option
            Description: Measurement parameter
            Data Type: Character string type (String)
            Range: Select either one of the following:
                    "S<XY>"
                    Where:
                    x=1 to 2
                    Y=1 to 2
            Preset Value: "S11"
        '''
        return self.query(":CALCulate1:PARameter{}:DEFine?".format(trace_idx)).strip('\n')

    @parameter.setter
    def parameter(self, par, trace_idx=1):
        '''This command gets the measurement parameter. :CALC:PAR:SPOR specifies the source output port for the absolute measurements.

            Parameter: Selection Option
            Description: Measurement parameter
            Data Type: Character string type (String)
            Range: Select either one of the following:
                    "S<XY>"
                    Where:
                    x=1 to 2
                    Y=1 to 2
            Preset Value: "S11"
        '''

        self.write(":CALCulate1:PARameter1:DEFine %s" % par)

    def set_trace_spar(self, trace_idx, spar):  
            self.write(":CALCulate1:PARameter%d:DEFine %s" %(trace_idx, spar))

    def set_active_trace(self, trace_idx=1):
        '''This command sets the active trace

            Parameter: Trace index
            Data Type: Numeric type (Integer)
            Range: 1 to 4 (lim by trace_number)
        '''

        self.write(":CALC1:PAR%d:SEL" % trace_idx)
    

    def data_array(self):
        '''This command gets the formatted data array.

            The array data element varies in the data format. For more information, see Internal Data Processing.

            If valid data is not calculated because of the invalid measurement, "1.#QNB" is read out.

            Parameter: Value
            Description: Formatted data array

                        Where n is an integer between 1 and NOP (number of measurement points):

                            <numeric n×2-1>:  Real part of data (complex number) at the n-th measurement point.

                            <numeric n×2>: Imaginary part of data (complex number) at the n-th measurement point. Always 0 when the data format is not the Smith chart format or the polar format

                        The number of data is {NOP×2}
            Data Type: Variant type Array (Range)
            Note: If there is no array data of NOP×2 when setting a formatted data array, an error occurs when executed.
        '''
        try:
            data = self.query(":CALCulate1:SELected:DATA:FDATa?")
            return data
        except Exception as e:
            print(f"Error getting data array: {e}")
            return ""

    def frequency_array(self):
        '''This command returns the frequency stimulus data.

        Parameter: Value
        Description: Indicates the array data (frequency) of NOP (number of measurement points). Where n is an integer between 1 and NOP.

                        Data(n-1): Frequency at the n-th measurement point

                    The index of the array starts from 0.
        Data Type: Variant type (Variant)'''


        try:
            data = self.query("SENSe:FREQuency:DATA?")
            return data
        except Exception as e:
            print(f"Error getting frequency array: {e}")
            return ""

    @property
    def data_array_format(self):
        '''This command gets the data array format.

            Parameter: Selection Option
            Description: Data format
            Data Type: Character string type (String)
            Range: "MLOGarithmic": Specifies the log magnitude format.
                    "PHASe": Specifies the phase format.
                    "GDELay": Specifies the group delay format.
                    "SLINear": Specifies the Smith chart format (Lin/Phase).
                    "SLOGarithmic": Specifies the Smith chart format (Log/Phase).
                    "SCOMplex": Specifies the Smith chart format (Re/Im).
                    "SMITh": Specifies the Smith chart format (R+jX).
                    "SADMittance": Specifies the Smith chart format (G+jB).
                    "PLINear": Specifies the polar format (Lin/Phase).
                    "PLOGarithmic": Specifies the polar format (Log/Phase).
                    "POLar": Specifies the polar format (Re/Im).
                    "MLINear": Specifies the linear magnitude format.
                    "SWR": Specifies the SWR format.
                    "REAL": Specifies the real format.
                    "IMAGinary": Specifies the imaginary format.
                    "UPHase": Specifies the expanded phase format.
                    "PPHase": Specifies the positive phase format.
            Preset Value: "MLOGarithmic"
        '''
        return self.query(":CALCulate1:SELected:FORMat?").strip('\n')
    
    @data_array_format.setter
    def data_array_format(self,dformat):
        '''This command sets the data array format.

            Parameter: Selection Option
            Description: Data format
            Data Type: Character string type (String)
            Range: "MLOGarithmic": Specifies the log magnitude format.
                    "PHASe": Specifies the phase format.
                    "GDELay": Specifies the group delay format.
                    "SLINear": Specifies the Smith chart format (Lin/Phase).
                    "SLOGarithmic": Specifies the Smith chart format (Log/Phase).
                    "SCOMplex": Specifies the Smith chart format (Re/Im).
                    "SMITh": Specifies the Smith chart format (R+jX).
                    "SADMittance": Specifies the Smith chart format (G+jB).
                    "PLINear": Specifies the polar format (Lin/Phase).
                    "PLOGarithmic": Specifies the polar format (Log/Phase).
                    "POLar": Specifies the polar format (Re/Im).
                    "MLINear": Specifies the linear magnitude format.
                    "SWR": Specifies the SWR format.
                    "REAL": Specifies the real format.
                    "IMAGinary": Specifies the imaginary format.
                    "UPHase": Specifies the expanded phase format.
                    "PPHase": Specifies the positive phase format.
            Preset Value: "MLOGarithmic"
        '''
        self.write(":CALCulate1:SELected:FORMat {}".format(dformat))


    @property
    def data_format(self):
        '''This command can be used to set/get the format data using the following SCPI commands:
            :CALC:DATA:FDAT
            :CALC:DATA:FMEM
            :CALC:DATA:SDAT
            :CALC:DATA:SMEM
            :CALC:DATA:XAX
            :CALC:FUNC:DATA
            :CALC:LIM:DATA
            :CALC:LIM:REP
            :CALC:LIM:REP:ALL
            :CALC:BLIM:REP
            :CALC:PLIM:DATA
            :CALC:PLIM:REP
            :CALC:RLIM:DATA
            :CALC:PLIM:REP
            :SENS:CORR:COEF
            :SENS:FREQ:DATA
            :SENS:SEGM:DATA

            Parameter: Selection Option
            Description: Data transfer format
            Data Type: Character string type (String)
            Range: ASCii: ASCII transfer format
                   REAL: IEEE 64-bit floating point binary transfer format
                   REAL32: IEEE 32-bit floating point binary transfer format
            Preset Value: ASCii
        '''
        dformat = self.query(":FORMat:DATA?").split('\n')[0]
        self._data_format = dformat
        return self._data_format
    
    @data_format.setter
    def data_format(self,dtype):
        '''This command can be used to sett the format data using the following SCPI commands:
            :CALC:DATA:FDAT
            :CALC:DATA:FMEM
            :CALC:DATA:SDAT
            :CALC:DATA:SMEM
            :CALC:DATA:XAX
            :CALC:FUNC:DATA
            :CALC:LIM:DATA
            :CALC:LIM:REP
            :CALC:LIM:REP:ALL
            :CALC:BLIM:REP
            :CALC:PLIM:DATA
            :CALC:PLIM:REP
            :CALC:RLIM:DATA
            :CALC:PLIM:REP
            :SENS:CORR:COEF
            :SENS:FREQ:DATA
            :SENS:SEGM:DATA

            Parameter: Selection Option
            Description: Data transfer format
            Data Type: Character string type (String)
            Range: ASCii: ASCII transfer format
                   REAL: IEEE 64-bit floating point binary transfer format
                   REAL32: IEEE 32-bit floating point binary transfer format
            Preset Value: ASCii
        '''
        self.write(":FORMat:DATA {}".format(dtype))


    def clear_averaging(self):
        '''This command resets the data count to 0, it is used for averaging. Measurement data before the execution of this object is not used for averaging.'''
        self.write(":SENSe1:AVERage:CLEar")


    @property
    def averaging(self):
        '''This command gets the averaging function.

            Parameter: Selection Option
            Description: ON/OFF of the averaging function
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: OFF
        '''
        return self.query(":SENSe1:AVERage:STATe?").split('\n')[0]

    @averaging.setter
    def averaging(self,state):
        '''This command sets the averaging function.

            Parameter: Selection Option
            Description: ON/OFF of the averaging function
            Data Type: Boolean type (Boolean)
            Range: ON|OFF|1|0
            Preset Value: OFF
        '''

        self.write(":SENSe1:AVERage:STATe {}".format(state))


    @property
    def if_bandwidth_frequency(self):
        '''This command gets the IF bandwidth.

            Parameter: Value
            Description: IF bandwidth
            Data Type: Numeric type (Real)
            Range: 10|15|20|30|40|50|70|100|150|200|300|400|500|700|1k|1.5k|2k|3k|4k|5k|7k|10k|15k|20k|30k|40k|50k|70k|100k|150k|200k|300k
            Preset Value: 70k
            Unit:Hz
        '''

        if_band = int(float(self.query(":SENSe1:BANDwidth:RESolution?").split("\n")[0]))
        self._if_bandwidth = if_band
        return self._if_bandwidth

    @if_bandwidth_frequency.setter
    def if_bandwidth_frequency(self,freq):
        '''This command sets the IF bandwidth.

            Parameter: Value
            Description: IF bandwidth
            Data Type: Numeric type (Real)
            Range: 10|15|20|30|40|50|70|100|150|200|300|400|500|700|1k|1.5k|2k|3k|4k|5k|7k|10k|15k|20k|30k|40k|50k|70k|100k|150k|200k|300k
            Preset Value: 70k
            Unit:Hz
        '''

        self.write(":SENSe1:BANDwidth:RESolution {}".format(freq))

    @property
    def center_frequency(self):
        '''This command gets the center frequency.

            Parameter: Value
            Description: Center Frequency
            Data Type: Numeric type (Real)
            Range: 100kHz to maximum frequency
            Preset Value: Depending on frequency option
            Unit: Hz
            Resolution: 500m
        '''

        freq = int(float(self.query(":SENSe1:FREQuency:CENTer?").split("\n")[0]))
        self._center_freq = freq
        return self._center_freq
    @center_frequency.setter
    def center_frequency(self,freq):
        '''This command sets the center frequency.

            Parameter: Value
            Description: Center Frequency
            Data Type: Numeric type (Real)
            Range: 100kHz to maximum frequency
            Preset Value: Depending on frequency option
            Unit: Hz
            Resolution: 500m
        '''

        self.write(":SENSe1:FREQuency:CENTer {}".format(freq))

    @property
    def start_frequency(self):
        '''This command sets the start frequency.

            Parameter: Value
            Description: Start frequency
            Data Type: Numeric type (Real)
            Range: 100k to maximum frequency
            Preset Value: 100k
            Unit: Hz
            Resolution: 1
        '''

        freq = int(float(self.query(":SENSe1:FREQuency:start?").split("\n")[0]))
        self._start_freq = freq
        return self._start_freq
    @start_frequency.setter
    def start_frequency(self,freq):
        '''This command gets the start frequency.

            Parameter: Value
            Description: Start frequency
            Data Type: Numeric type (Real)
            Range: 100k to maximum frequency
            Preset Value: 100k
            Unit: Hz
            Resolution: 1
        '''

        self.write(":SENSe1:FREQuency:start {}".format(freq))

    @property
    def stop_frequency(self):
        '''This command gets the stop frequency.

            Parameter: Value
            Description: Stop value
            Data Type: Numeric type (Real)
            Range: 100k ~ Maximum frequency
            Preset Value: Depending on frequency option
            Unit: Hz
            Resolution: 1
        '''

        freq = int(float(self.query(":SENSe1:FREQuency:stop?").split("\n")[0]))
        self._stop_freq = freq
        return self._stop_freq
    @stop_frequency.setter
    def stop_frequency(self,freq):
        '''This command sets the stop frequency.

        Parameter: Value
        Description: Stop value
        Data Type: Numeric type (Real)
        Range: 100k ~ Maximum frequency
        Preset Value: Depending on frequency option
        Unit: Hz
        Resolution: 1
        '''
        self.write(":SENSe1:FREQuency:stop {}".format(freq))

    @property
    def span_frequency(self):
        '''This command gets the frequency span.

            Parameter:Value
            Description: Span frequency
            Data Type: Numeric type (Real)
            Range: 0 ~ Maximum Frequency -100k
            Preset Value: Depending on frequency option
            Unit: Hz
            Resolution: 1
        '''
        freq = int(float(self.query(":SENSe1:FREQuency:span?").split("\n")[0]))
        self._span_freq = freq
        return self._span_freq
    @span_frequency.setter
    def span_frequency(self,freq):
        '''This command sets the frequency span.

            Parameter:Value
            Description: Span frequency
            Data Type: Numeric type (Real)
            Range: 0 ~ Maximum Frequency -100k
            Preset Value: Depending on frequency option
            Unit: Hz
            Resolution: 1
        '''
        self.write(":SENSe1:FREQuency:span {}".format(freq))

    @property
    def sweep_points(self):
        '''This command gets the number of measurement points.

            Parameter: Value
            Description: Number of measurement points
            Data Type: Numeric type (Integer)
            Range: 2 ~ 10001
            Preset Value: 201
            Resolution: 1
        '''

        points = int(self.query(":SENSe1:SWEep:POINts?").split("\n")[0])
        self._sweep_points = points
        return self._sweep_points
    @sweep_points.setter
    def sweep_points(self,points):
        '''This command sets the number of measurement points.

            Parameter: Value
            Description: Number of measurement points
            Data Type: Numeric type (Integer)
            Range: 2 ~ 10001
            Preset Value: 201
            Resolution: 1
        '''

        self.write(":SENSe1:SWEep:POINts {}".format(points))

    @property
    def average_points(self):
        '''This command gets the averaging factor.

            Parameter: Value
            Description: Averaging factor
            Data Type: Numeric type (Integer)
            Range: 1 ~ 999
            Preset Value: 16
            Unit: -
            Resolution: -
        '''

        avg = int(self.query(":SENSe1:AVERage:COUNt?").split("\n")[0])
        self._average_points = avg
        return self._average_points
    
    @average_points.setter
    def average_points(self,avg):
        '''This command sets the averaging factor.

            Parameter: Value
            Description: Averaging factor
            Data Type: Numeric type (Integer)
            Range: 1 ~ 999
            Preset Value: 16
            Unit: -
            Resolution: -
        '''
        self.write(":SENSe1:AVERage:COUNt {}".format(avg))

    @property 
    def sweep_time_auto(self):
        return int(self.query(":SENSe1:SWEep:TIME:AUTO?").strip("\n"))
    
    @sweep_time_auto.setter
    def sweep_time_auto(self, param):
        """
        Get and set automatic sweep time, boolean {0|1}
        """
        
        self.write(":SENSe1:SWEep:TIME:AUTO %s" % param).strip("\n")

    @property
    def sweep_time(self):
        return float(self.query(":SENSe1:SWEep:TIME?").strip("\n"))
    
    @sweep_time.setter
    def sweep_time(self, stime):
        """
        Get and set sweep time for channel 1, in seconds
        """
        self.sweep_time_auto="ON"
        self.write(":SENSe1:SWEep:TIME %.2f" % stime)

    def get_data(self, select=None):
        """
        Get data from the VNA in the current format for the selected trace
        Range: 1 to 4 (limited by trace_number) or "all"
        
        Returns:
            tuple: Tuple containing (magnitudes_or_real, phases_or_imag, frequencies)
        """

        if select=="all":
            data=self.get_data(1)
            data=np.vstack((data[-1], data[:2]))
            for i in range (2, self.trace_number+1):
                data=np.vstack((data, self.get_data(i)[:-1]))

            self.set_active_trace(1)
            return np.transpose(data)
        
        if isinstance(select, int):
            self.set_active_trace(select)

        try:
            # Get the formatted data array and frequencies
            prev_format=self.data_array_format
            self.data_array_format="PLOG"

            data_str = self.data_array()
            freq_str = self.frequency_array()
            
            self.data_array_format=prev_format
            if not data_str or not freq_str:
                print("Warning: Empty data or frequency array")
                return None
            
            # Parse the data
            data_values = np.array([float(x) for x in data_str.split(',')])
            freq_values = np.array([float(x) for x in freq_str.split(',')])
            
            # Data is in pairs (real/imag or mag/phase)
            if len(data_values) % 2 == 0:
                # Reshape for complex data (real/imag or mag/phase pairs)
                data_pairs = data_values.reshape(-1, 2)
                
                # Return the pair components and frequencies
                return data_pairs[:, 0], data_pairs[:, 1], freq_values
            else:
                print(f"Warning: Unexpected data length: {len(data_values)}")
                return None
                
        except Exception as e:
            print(f"Error in get_data: {e}")
            return None


    @property
    def trigger_source(self):
        return self.query(":TRIG:SOUR?")
    
    @trigger_source.setter
    def trigger_source(self, source="BUS"):
        self.write(":TRIGger:SOURce {}".format(source))

    def get_syserr(self):
        return self.query(":SYST:ERR?").strip("\n")
    
    def clear(self):
        self.write("*CLS")

    def trigger(self):
        if self.trigger_source == "BUS\n":
            self.write(":TRIGger:SINGle")
            return None
        
        raise RuntimeError("Trigger source must be BUS")
    
    def OPC(self):
        # Wait until operation complete and return 1
        return self.query("*OPC?")
    
    def get_OPC(self, mask=65535):
        # Get Operation Status Condition Register
        return int(self.query(":STAT:OPER:COND?")) & mask
    
    
    def simple_config(self, start, stop, points, if_bandwidth, avg_points=128, power=-5, unit="GHz", data_format="PLOG"): 
        # Inserir os valores em Hz, tempo em segundos, IF bandwidth em Hz, power entre 0 e -20 dBm

        self.power = "OFF"
        self.clear_averaging()
        
        self.average_points = avg_points
        self.averaging = bool(avg_points)

        self.set_amplitude(power)

        self.data_array_format=data_format
        
        try:
            unit=10**(3*['H', 'k', 'M', 'G'].index(unit[0]))
        except:
            raise ValueError("Invalid unit. Must be Hz, kHz, MHZ or GHz")

        self.start_frequency = start*unit
        self.stop_frequency= stop*unit
        self.sweep_points = points
        self.if_bandwidth_frequency = if_bandwidth

        return None

    def measure_get(self, sparameter, save_path=None, float_format="%+.7e", duration=5):
        if sparameter not in ["S11", "S21", "S12", "S22"]:
            raise ValueError("Invalid S-Parameter")

        self.parameter=sparameter
        self.OPC()
        
        self.power = "ON"

        if self.get_OPC(mask=32):
            self.trigger()
            self.OPC()
        else:
            sleep(duration)
        
        self.power = "OFF"
        
        df=pd.DataFrame(np.transpose(self.get_data()), columns=["amp", "phase", "freq"])

        if save_path:
            try:
                df.to_csv(save_path, float_format=float_format)
            except Exception as e:
                print(f"Error saving: {e}")

        return df
    
    def measure_s2p(self, port_number=2, save_path=None, float_format="%+.7e", duration=1, symmetric=True):
        prev_trigger=self.trigger_source 
        self.trigger_source="BUS"

        if port_number==2:
            self.trace_number=2

            self.set_trace_spar(1, "S11")
            self.set_trace_spar(2, "S21")

            if not symmetric:
                self.set_trace_spar(3, "S12")
                self.set_trace_spar(4, "S22")
        else:
            self.trace_number=1
            self.set_trace_spar(1, "S11")
    
        self.power = "ON"
        self.OPC()

        if self.get_OPC(mask=32):
            self.trigger()
            self.OPC()
        else:
            sleep(duration)

        self.power = "OFF"

        data=self.get_data("all")
        data=np.hstack((data, np.zeros((len(data), 4))))

        header=None
        if len(data[0]) == 3:
            header=["freq", "S11_amp", "S11_phase"]
            ext=".s1p"
        if len(data[0]) == 9:
            header=["freq", "S11_amp", "S11_phase", "S21_amp", "S21_phase", "S12_amp", "S12_phase", "S22_amp", "S22_phase"]
            ext=".s2p"

        df=pd.DataFrame(data=data, columns=header)

        if save_path:
            if save_path[-4]=='.':
                save_path=save_path[:-4]
            
            save_path=save_path+ext

            with open(save_path, 'w') as myfile:
                myfile.write("# HZ S DB R 50\n")

            df.to_csv(save_path, sep='\t', index=False, header=False, float_format=float_format, mode='a')

        self.trigger_source=prev_trigger

        return df
            
