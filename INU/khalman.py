from xml.etree.ElementTree import QName
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import random 
from misc_functions import *
# states:  x y z x_dot y_dot z_dot
class khalman_gps_imu():
    def __init__(self, initial_states:np.array,dt_imu:float,dt_gps:float):
        self.states = initial_states
        self.nx = len(self.states)
        self.half = int(self.nx/2)
        self.I = np.eye(self.nx)
        self.I_h = np.eye(self.half)
        empty_quad = np.zeros([self.half,self.half])
        empty_half = np.zeros([self.half,self.nx])
        self.P = np.eye(self.nx)
        self.dt_imu = dt_imu
        self.dt_gps = dt_gps
        self.A = np.array(np.copy(self.I)+np.diagflat(np.ones(self.half)/(self.dt_gps*4), self.half))
        self.H_gps = np.block([[np.copy(self.I_h),empty_quad],[empty_half]])
        self.H_imu = np.block([[empty_half],[empty_quad,np.copy(self.I_h)]])
        self.H =self.H_gps+self.H_imu
        self.ra = self.dt_imu
        self.R_imu = np.block([[empty_half],[empty_quad,self.I_h*self.dt_imu]])
        self.R_gps = np.block([[self.I_h*self.dt_imu,empty_quad],[empty_half]])
        self.R =self.R_imu+self.R_gps
        self.G = np.vstack(np.hstack([np.ones(self.half)*(self.dt_gps*1/4)**2, np.ones(self.half)*self.dt_imu/4]))
        self.Q = np.diagflat(self.G)
        self.lock = threading.Lock()

    def get_accel_imu(self, imu_data:np.array)->np.array:
        ax = imu_data[0]; ay = imu_data[1]; az = imu_data[2]
        gx = imu_data[3]; gy = imu_data[4]; gz = imu_data[5]
        mx = imu_data[6]; my = imu_data[7]; mz = imu_data[8]
        pitch = (getPitch(ax,ay,az)+gx)/2
        roll = (getRoll(ax,ay,az)+gz)/2
        yaw = (getYaw(mx,my,mz,pitch,roll)+gy)/2
        rot = get_rotmat(pitch,yaw,roll)
        real_move = np.hstack([ax,ay,az])@rot
        return real_move

    def predict(self)-> None:
        self.states = self.A@self.states
        self.P = self.A@self.P@self.A.T + self.Q
       
    def get_states(self)->np.array:
        return self.states
  
    def update_imu(self, new_imu:np.array)->None:
        self.lock.acquire()
        y = np.array([*np.zeros(self.half), *new_imu]) - self.H_imu@self.states
        S = self.H_imu@self.P@self.H_imu.T + self.R
        K = (self.P@self.H_imu.T@np.linalg.inv(S))
        self.states = self.states + K@y
        self.P = (self.I-(K@self.H_imu))@self.P
        self.lock.release()

    def update_gps(self, new_gps:np.array)->None:
        self.lock.acquire()
        print(*new_gps,*np.zeros(self.half))
        y = np.array([*new_gps,*np.zeros(self.half)]) - self.H_gps@self.states
        S = self.H_gps@self.P@self.H_gps.T + self.R
        K = (self.P@self.H_gps.T@np.linalg.inv(S))
        self.states = self.states + K@y
        self.P = (self.I-(K@self.H_gps))@self.P
        self.lock.release()
        
    def set_gps_pos(self, new_gps:np.array)->None:
        self.states= np.array(np.vstack(np.vstack(new_gps),np.vstack(self.states[self.half:self.nx])))
    
    def run(self,dt):
        try:
            while True:
                s = time.process_time_ns()
                self.lock.acquire()
                self.predict()
                self.lock.release()
                l = time.process_time_ns()-s
                time.sleep(dt-l*1e-9)
        except:
            print('filter stopping')
dt = 0.1
dt_imu = 1
dt_gps = 20
t =  1000
b = int(t/dt)
x = np.linspace(0, t,b)
y = x*np.pi
r = 100
x1 =  np.cos(y*100)*r
x2 =  np.sin(y*100)*r
x3 = -np.sin(y*100)*r
x4 =  np.cos(y*100)*r
y = np.block([x1,x2,x3,x4])
print(y[0:4])
k = khalman_gps_imu(y[0:4],dt_imu,dt_gps)
i = 0
l = int(len(y)/4)
print(len(y))
e = np.zeros([l,1])
y1 = np.zeros([l,1])
y2 = np.zeros([l,1])
y3 = np.zeros([l,1])
y4 = np.zeros([l,1])
print()
mod = 1/dt_imu
m = 1/dt_gps
k.update_gps([x1[0],x2[0]])
k.predict()
random.random()
for t in x:
    
    #k.update_imu(y[i:i+4])
    if i%(1/m) == 0: 
        k.update_gps([x1[i]+(random.random()-0.2)*10,x2[i]+(random.random()-0.8)*10])
        
    
    if i%(1/mod) == 0:
        #print(i)
        
        k.update_imu(np.array([x3[i]+(random.random()-0.4)*5,x4[i]])+(random.random()-0.6)*5)
 
    k.predict()
    z = k.get_states()
    e[i] = np.linalg.norm(y[i:i+4]-z)
    y1[i]= z[0]
    y2[i]= z[1]
    y3[i]= z[2]
    y4[i]= z[3]
    # print(k.get_accel_imu(np.array([2,0,9.81,0,0,0,100,-100,0])))
    i += 1

plt.figure()
plt.subplot(211)
plt.plot(x1,x2)
plt.plot(y1,y2,'r--')
plt.subplot(212)
plt.plot(x3,x4)
plt.plot(y3,y4,'r--')
plt.waitforbuttonpress(0)
plt.close()