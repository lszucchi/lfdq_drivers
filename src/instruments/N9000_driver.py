import pyvisa


rm = pyvisa.ResourceManager()

import numpy as np
import pandas as pd

class N9000_driver():
    '''Driver for the VNA N9000'''
    
    def __init__(self,address, timeout=5000):
        self._address = address
        self._inst = rm.open_resource(self._address, timeout=timeout)
        self.idn = self._inst.query("*IDN?")

    def IDN(self):
        return self._inst.query("*IDN?")
    
    def close(self):
        self._inst.close()

    @property
    def f_center(self):
        return float(self._inst.query(":SENS:FREQ:CENT?"))

    @f_center.setter
    def f_center(self, value):
        self._inst.write(f":SENS:FREQ:CENT {value}")

    @property
    def f_span(self):
        return float(self._inst.query(":SENS:FREQ:SPAN?"))

    @f_span.setter
    def f_span(self, value):
        self._inst.write(f":SENS:FREQ:SPAN {value}")

    @property
    def f_start(self):
        return float(self._inst.query(":SENS:FREQ:STAR?"))

    @f_start.setter
    def f_start(self, value):
        self._inst.write(f":SENS:FREQ:STAR {value}")

    @property
    def f_stop(self):
        return float(self._inst.query(":SENS:FREQ:STOP?"))

    @f_stop.setter
    def f_stop(self, value):
        self._inst.write(f":SENS:FREQ:STOP {value}")

    @property
    def f_res(self):
        return float(self._inst.query(":SENS:FREQ:RES?"))

    @f_res.setter
    def f_res(self, value):
        self._inst.write(f":SENS:BAND:RES {value}")

    @property  
    def average_state(self):
        return self._inst.query(":AVER:STAT?")

    @average_state.setter
    def average_state(self, value):
        if value not in ["ON", "OFF"]:
            raise ValueError("Invalid average state. Must be one of: ON, OFF")
        self._inst.write(f":AVER:STAT {value}")

    @property
    def average_count(self):
        return int(self._inst.query(":AVER:COUN?"))

    @average_count.setter
    def average_count(self, value):
        self._inst.write(f":AVER:COUN {int(value)}")

    @property
    def average_type(self):
        return self._inst.query(":AVER:TYPE?")

    @average_type.setter
    def average_type(self, value):
        if value not in ["LOG", "RMS", "SCALAR"]:
            raise ValueError("Invalid average type. Must be one of: LOG, RMS, SCALAR")
        self._inst.write(f":AVER:TYPE {value}")

    @property
    def auto_average(self):
        return self._inst.query(":AVER:AUTO?")

    @auto_average.setter
    def auto_average(self, value):
        if value not in ["ON", "OFF"]:
            raise ValueError("Invalid auto average value. Must be one of: ON, OFF")
        self._inst.write(f":AVER:AUTO {value}")

    def clear_average(self):
        self._inst.write(":AVER:CLE")

    @property
    def trace_type(self):
        return self._inst.query(":TRAC1:TYPE?")

    @trace_type.setter
    def trace_type(self, value):    
        if value not in ["WRIT", "AVER", "MAXH", "MINH"]:
            raise ValueError("Invalid trace type. Must be one of: WRIT, AVER, MAXH, MINH")
        self._inst.write(f":TRAC1:TYPE {value}")

    @property
    def sweep_time(self):
        return float(self._inst.query(":SWE:TIME?"))

    @sweep_time.setter
    def sweep_time(self, value):
        self._inst.write(f":SWE:TIME {value}")

    @property
    def auto_sweep_time(self):
        return self._inst.query(":SWE:AUTO?")

    @auto_sweep_time.setter
    def auto_sweep_time(self, value):
        if value not in ["ON", "OFF"]:
            raise ValueError("Invalid auto sweep time value. Must be one of: ON, OFF")
        self._inst.write(f":SWE:AUTO {value}")

    @property
    def continuous(self):
        return self._inst.query(":INIT:CONT?")

    @continuous.setter
    def continuous(self, value):
        if value not in ["ON", "OFF"]:
            raise ValueError("Invalid continuous value. Must be one of: ON, OFF")
        self._inst.write(f":INIT:CONT {value}")

    def restart(self):
        self._inst.write("*RST")

    def meas_restart(self):
        self._inst.write(":INIT:REST")

    def start(self):
        self._inst.write(":INIT")

    @property
    def sweep_type(self):
        return self._inst.query(":SWE:TYPE?")

    @sweep_type.setter
    def sweep_type(self, value):
        if value not in ["FFT", "SWE"]:
            raise ValueError("Invalid sweep type. Must be one of: FFT, SWE")
        self._inst.write(f":SWE:TYPE {value}") 

    @property 
    def auto_sweep_type(self):
        return self._inst.query(":SWE:TYPE:AUTO?") 

    @auto_sweep_type.setter
    def auto_sweep_type(self, value):
        if value not in ["ON", "OFF"]:
            raise ValueError("Invalid auto sweep type value. Must be one of: ON, OFF")
        self._inst.write(f":SWE:TYPE:AUTO {value}") 

    @property
    def sweep_points(self):
        return int(self._inst.query(":SWE:POIN?"))

    @sweep_points.setter
    def sweep_points(self, value):
        self._inst.write(f":SWE:POIN {int(value)}")

    @property
    def trigger_source(self):
        return self._inst.query(":TRIG:SOUR?")

    @trigger_source.setter
    def trigger_source(self, value):
        if value not in ["IMM", "EXT1", "EXT2", "BUS"]:
            raise ValueError("Invalid trigger source. Must be one of: IMM, EXT1, EXT2, BUS")
        self._inst.write(f"TRIG:SOUR {value}")

    @property
    def trigger_level(self):
        return float(self._inst.query(":LIST:TRIG:LEV?"))

    @trigger_level.setter
    def trigger_level(self, value):
        self._inst.write(f":LIST:TRIG:LEV {value}")

    @property
    def trigger_delay(self):
        return self._inst.query(":LIST:TRIG:DEL?")

    @trigger_delay.setter
    def trigger_delay(self, value):
        self._inst.write(f":LIST:TRIG:DEL {np.clip(value, 0, 0.5)}")

    def opc(self):
        return self._inst.query("*OPC?")

    def enable_opc(self):
        self._inst.write("*OPC")
        self._inst.write("*ESE 1")

    def trigger(self):
        self._inst.write("*TRG")

    def data_ready(self):
            return int(self._inst.query("*ESR?"))

    def start(self):
        self._inst.write(":INIT")

    def get_trace_data(self):
        self._inst.write(":FORM:DATA ASC")
        data = self._inst.query("TRAC? TRACE1")
        return np.array([float(i) for i in data.split(",")])

    def get_freq_data(self):
        return np.arange(self.f_start, self.f_stop, (self.f_stop-self.f_start)/self.sweep_points)+(self.f_stop-self.f_start)/(2*self.sweep_points)

    def get_data(self):
        f_data=np.arange(self.f_start, self.f_stop, (self.f_stop-self.f_start)/self.sweep_points)+(self.f_stop-self.f_start)/(2*self.sweep_points)
        data=np.hstack((f_data.reshape(-1,1), self.get_trace_data().reshape(-1,1)))
        df=pd.DataFrame(data, columns=["f", "dBm"])
        return df