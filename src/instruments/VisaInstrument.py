import pyvisa as visa

rm = visa.ResourceManager()

class VisaInstrument:
    def __init__(self,resource_address):
        self._inst = rm.open_resource(resource_address)
        self._idn = self._inst.query("*IDN?")[:-1]
        self._inst.timeout = 30000

    def write(self,command):
        if self._inst._session is None:
            self.open()
        return self._inst.write(command)

    def write_raw(self,commad):
        if self._inst._session is None:
            self.open()
        return self._inst.write_raw(commad)

    def query(self,command):
        if self._inst._session is None:
            self.open()
        return self._inst.query(command)

    def read_raw(self):
        if self._inst._session is None:
            self.open()
        return self._inst.read_raw()

    def open(self, timeout=30000):
        return self._inst.open(open_timeout=timeout)

    def close(self):
        return self._inst.close()