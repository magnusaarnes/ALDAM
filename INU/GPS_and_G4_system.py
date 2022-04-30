import SIM7600X.python.TCP.TCP as fourG
import SIM7600X.python.GPS.GPS as GPS
import serial
import RPi.GPIO as GPIO
import time


class SIM7600X_GPS_and_4G():
    def __init__(self,power_but:int,port:int,APN:str,serial:serial.Serial=serial.Serial('/dev/ttyUSB2',115200),ip:str='127.0.0.1'):
        self.power_button =power_but
        self.port = port
        self.APN = APN
        self.serial = serial
        self.ip = ip

    def SIM7600X_power_on(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.power_key,GPIO.OUT)
        time.sleep(0.1)
        GPIO.output(self.power_key,GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.power_key,GPIO.LOW)
        time.sleep(10)
        self.serial.flushInput()
        return 1
        
    def SIM7600X_get_gps_position(self):
        rec_null = True
        answer = 0
        rec_buff = ''
        self.__send_cmd('AT+CGPS=1,1','OK',1)
        time.sleep(2)
        while rec_null:
            answer = self.__send_cmd('AT+CGPSINFO','+CGPSINFO: ',1)
            if 1 == answer:
                answer = 0
                if ',,,,,,' in rec_buff:
                    rec_null = False
                    time.sleep(1)
            else:
                rec_buff = ''
                self.__send_cmd('AT+CGPS=0','OK',1)
                return False
            time.sleep(1.5)

    def SIM7600X_power_down(self):
        GPIO.output(self.power_key,GPIO.HIGH)
        time.sleep(3)
        GPIO.output(self.power_key,GPIO.LOW)
        time.sleep(18)
        return 1

    def SIM7600X_send_cmd(self,command,back,timeout):
        return self.__send_cmd(self,command,back,timeout)
  
    def __send_cmd(self,command,back,timeout):
        rec_buff = ''
        serial.write((command+'\r\n').encode())
        time.sleep(timeout)
        if self.serial.inWaiting():
            time.sleep(0.1 )
            rec_buff = serial.read(self.serial.inWaiting())
        if rec_buff != '':
            if back not in rec_buff.decode():
                return 0
            else:
                return 1
        else:
            return -1

gps = SIM7600X_GPS_and_4G()
gps.SIM7600X_power_on()
n = range(0,25)
i = 0
for i in n:
    i = i+1
    print(gps.SIM7600X_get_gps_position())
gps.SIM7600X_power_down()