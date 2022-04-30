#!/usr/bin/python
# -*- coding:utf-8 -*-
from cmath import exp
from numpy import double, dtype, int16
import RPi.GPIO as GPIO
import numpy as np
import serial
from khalman import khalman_gps_imu
import time
class GPS_unit():
	def __init__(self,filter:khalman_gps_imu,adress:str='/dev/ttyUSB2',hz:int=115200,power_key:int=6) -> None:
		self.ser = serial.Serial(adress,hz)
		self.ser.flushInput()
		self.filter = filter
		self.power_key = power_key
		self.rec_buff = ''
		self.rec_buff2 = ''
		self.time_count = 0
		self.dt_gps = 1.5
		self.pos = np.array([None,None],dtype=double)

	def send_at(self,command,back,timeout):
		self.rec_buff = ''
		self.ser.write((command+'\r\n').encode())
		time.sleep(timeout)
		if self.ser.inWaiting():
			time.sleep(0.01)
			self.rec_buff = self.ser.read(self.ser.inWaiting())
		if self.rec_buff != '':
			if back not in self.rec_buff.decode():
				return 0
			else:
				return 1
		else:
			return 0
	def run(self):
		self.power_on()
		try: 
			while True:
				self.get_gps_position()
		except:
			self.power_down()
		
			
				
	def get_gps_position(self):
		rec_null = True
		answer = 0
		print('Start GPS session...')
		self.rec_buff = ''
		self.send_at('AT+CGPS=1,1','OK',1)
		time.sleep(self.dt_gps)
		while rec_null:
			answer = self.send_at('AT+CGPSINFO','+CGPSINFO: ',1)
			if 1 == answer:
				answer = 0
				if ',,,,,,' in self.rec_buff:
					print('GPS is not ready')
					rec_null = False
					time.sleep(self.dt_gps)
				else:
					self.update_lat_lon(self.rec_buff)
					self.filter.update_gps(self.get_gps_position)
			else:
				print('error %d'%answer)
				self.rec_buff = ''
				self.send_at('AT+CGPS=0','OK',1)
				return False
			
			time.sleep(self.dt_gps)


	def power_on(self):
		print('SIM7600X is starting:')
		GPIO.setmode(GPIO.BCM)
		GPIO.setwarnings(False)
		GPIO.setup(self.power_key,GPIO.OUT)
		time.sleep(0.1)
		GPIO.output(self.power_key,GPIO.HIGH)
		time.sleep(2)
		GPIO.output(self.power_key,GPIO.LOW)
		time.sleep(20)
		self.ser.flushInput()
		print('SIM7600X is ready')

	def update_lat_lon(self,data):
		self.pos[1] = data[1]
		self.pos[2] = data[3]

	def power_down(self):
		print('SIM7600X is loging off:')
		GPIO.output(self.power_key,GPIO.HIGH)
		time.sleep(3)
		GPIO.output(self.power_key,GPIO.LOW)
		time.sleep(18)
		print('Good bye')
if __name__ == '__main__':
	try:
		gps = GPS_unit()
		gps.power_on()
		gps.get_gps_position()
		gps.power_down()
	except:
		if gps.ser != None:
			gps.ser.close()
		gps.power_down()
		GPIO.cleanup()
	if gps.ser != None:
			gps.ser.close()
			GPIO.cleanup()	

		