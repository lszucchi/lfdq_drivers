import socket
from time import sleep


class Model_430:

    def __init__(self,DeviceIP,timeout=10):
        #DeviceIP="169.254.101.115"
        self.DVM=socket.socket()
        self.DVM.settimeout(timeout)
        try:
            self.DVM.connect((DeviceIP, 7180))
            sleep(1e-2)
            print(self.DVM.recv(1024))
        except socket.timeout:
            print('Socket Timed Out')



    def query(self,question):
        self.DVM.send(question.encode())
        sleep(1e-2)
        try:
            response=self.DVM.recv(1024)
            return response
        except socket.timeout:
            print('Socket Timed Out') 
            return None
    
    def write(self,command):
        self.DVM.send(command.encode())

    def field(self):
        return float(self.query('CURR:SUPP?\n').decode().strip('\r\n'))*0.0912

    def ramp_to_and_monitor(self,value):
        self.field_target=value
        self.write('RAMP\n')
        sleep(1e-2)
        state=self.query('STATE?\n').decode().strip('\r\n')
        while state=='1':
            field_now=self.field()
            print(f'\r{field_now} T'+' '*100,end='\r',flush=True)
            sleep(1)
            state=self.query('STATE?\n').decode().strip('\r\n')
        final_field= float(self.query('CURR:SUPP?\n').decode().strip('\r\n'))*0.0912
        print('Target Reached {}'.format(final_field))
        self.pause()

    def ramp_to(self,value):
        self.field_target=value
        self.write('RAMP\n')
      

    def ramp_to_zero(self):
        self.write('ZERO\n')
        state=self.query('STATE?\n').decode().strip('\r\n')
        while True:
            field_now=self.field()
            print(f'\r{field_now} T'+' '*100,end='\r',flush=True)
            sleep(1)
            if (abs(field_now)<5e-6):
                print('Field at Zero')
                break


    def pause(self):
        self.write('PAUSE\n')

    @property
    def ramp_rate(self):
        result = self.query('RAMP:RATE:FIELD:1?\n').decode().strip('\r\n').split(',')[0]
        return result
    
    @property
    def field_target(self):
        result= self.query('FIELD:TARG?\n').decode().strip('\r\n')
        return result 
    
    @ramp_rate.setter
    def ramp_rate(self,value):
        self.write('CONF:RAMP:RATE:FIELD 1,{},8\n'.format(value))
    
    @field_target.setter
    def field_target(self,value):
        self.write('CONF:FIELD:TARG {}\n'.format(value))