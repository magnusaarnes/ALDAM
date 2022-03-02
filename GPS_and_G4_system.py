import serial
import RPi.GPIO as GPIO
import time
import json
gps_update_time = 1.5 #s
msg_wait_time = 1 #s

class SIM7600X_GPS_and_4G():
    """_summary_
    Controll and opperates the  SIM7600X hatt, can send TCP pacages and recive GPS data.
    """    
    def __init__(self,power_but:int,port:int,APN:str,serial:serial.Serial=serial.Serial('/dev/ttyUSB2',115200),ip:str='127.0.0.1'):
        self.power_button =power_but
        self.port = port
        self.APN = APN
        self.serial = serial
        self.ip = ip
        self.last_recived_serial_msg = ''
        self.last_sendt_serial_msg = ''
        self.timeout = 0.5

    def SIM7600X_power_on(self):
        """
        powers  on the hat, flushes the serial port.
        Returns:
            int: when done
        """        
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
        
    def SIM7600X_get_gps_pos(self):
        """
        returns the GPS possition

        Returns:
            str: gps position
        """        
        rec_null = True
        answer = 0
        rec_buff = ''
        self.__send_cmd('AT+CGPS=1,1','OK',1)
        time.sleep(msg_wait_time)
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
                return self.last_recived_serial_msg
            time.sleep(gps_update_time)

    def SIM7600X_power_down(self):
        """
        power down the SIM7600X hat

        """        
        GPIO.output(self.power_key,GPIO.HIGH)
        time.sleep(3)
        GPIO.output(self.power_key,GPIO.LOW)
        time.sleep(10)

    def SIM7600X_send_cmd(self,command,back,timeout):
        """
         send cmd data over the serial port to the SIM7600X hat

        Args:
            command (str): the command to be sent to the hat
            back (str): the expected response
            timeout (float): timeout time in seconds

        Returns:
            int: 1 if response was as expected 0 if it was not -1 if message was not sent 
        """        
        t = self.timeout
        self.timeout = timeout
        resp = self.__send_cmd(self,command,back,timeout)
        self.timeout = t
        return resp
  
    def __send_cmd(self,command,back):
        """
         send cmd data over the serial port to the SIM7600X hat

        Args:
            command (str): the command to be sent to the hat
            back (str): the expected response

        Returns:
            int: 1 if response was as expected 0 if it was not -1 if message was not sent 
        """      
        rec_buff = ''
        self.__write_serial(command)
        time.sleep(self.timeout)
        if self.serial.inWaiting():
            time.sleep(0.1)
            rec_buff = self.__read_serial()
        if rec_buff != '':
            if back not in rec_buff.decode():
                return 0
            else:
                return 1
        else:
            return -1

    def __read_serial(self):
        """reads the serial port and returns the result

        Returns:
            str: the data in the serial port
        """
        self.last_recived_serial_msg = serial.read(self.serial.inWaiting()).decode()
        return self.last_recived_serial_msg

    def __write_serial(self,command):
        """
        writes to the serial port saves the last message that was sendt.

        Args:
            command (str): the command to be written in the serial port.
        """        
        self.last_sendt_serial_msg = command+'\r\n'
        serial.write(self.last_sendt_serial_msg.encode())


    def connect_socket(self, server_ip:str, server_port:int):
        """
            sets up a TCP socket connection from the hat to a TCP server

        Args:
            server_ip (str): the ip of the server to connect to
            server_port (int): the port of the server to connect to

        Returns:
            int: 1 if connection was successful, 0 if hat was unable to connect. -1 if hat could not be reached.
        """
        self.__send_cmd('AT+CSQ','OK')
        self.__send_cmd('AT+CREG?','+CREG: 0,1')
        self.__send_cmd('AT+CPSI?','OK')
        self.__send_cmd('AT+CIPMODE=0', 'OK')
        self.timeout = 0.5
        self.__send_cmd('AT+CGREG?','+CGREG: 0,1')
        self.timeout = 1
        self.__send_cmd(','.join('AT+CGSOCKCONT=1','IP',self.APN),'OK')
        self.__send_cmd('AT+CSOCKSETPN=1', 'OK')
        self.__send_cmd('AT+CIPMODE=0', 'OK')
        self.timeout = 5
        self.__send_cmd('AT+NETOPEN', '+NETOPEN: 0')
        self.timeout = 1
        self.__send_cmd('AT+IPADDR', '+IPADDR:')
        self.timeout = 5
        msg = self.__send_cmd(','.join('AT+CIPOPEN=0','TCP',server_ip,server_port),'+CIPOPEN:')
        return msg


    def send_tcp_packet(self, packet:str):
        """sends an tcp packet as a json string

        Args:
            packet (str): the packet to be sendt

        Returns:
            int: 1 if connection was successful, 0 if hat was unable to connect. -1 if hat could not be reached.
        """        
        self.timeout = 2
        self.__send_cmd('AT+CIPSEND=0,', '>')
        self.__write_serial(packet)
        self.timeout = 1
        ans = self.__send_cmd(b'\x1a'.decode(),'OK')
        
        return ans
        