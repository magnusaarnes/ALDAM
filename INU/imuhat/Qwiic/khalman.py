from xml.etree.ElementTree import QName
import numpy as np
import matplotlib.pyplot as plt
# states:  x y z x_dot y_dot z_dot
class khalman_gps_imu():
    def __init__(self, initial_states:np.array,dt:float,dt_gps:float):
        self.states = initial_states
        self.nx = len(self.states)
        self.half = int(self.nx/2)
        self.I = np.eye(self.nx)
        self.I_h = np.eye(self.half)
        empty_quad = np.zeros([self.half,self.half])
        empty_half = np.zeros([self.half,self.nx])
        self.P = np.eye(self.nx)
        self.dt = dt
        self.dt_gps = dt_gps
        self.A = np.array(np.copy(self.I)+np.diagflat(np.ones(self.half)*0.1, self.half))
        print(self.A)
        self.H_gps = np.block([[np.copy(self.I_h),empty_quad],[empty_half]])
        self.H_imu = np.block([[empty_half],[empty_quad,np.copy(self.I_h)]])

        self.H =self.H_gps+self.H_imu
        self.ra = self.dt
        self.R_imu = np.block([[empty_half],[empty_quad,self.I_h*self.dt]])
        self.R_gps = np.block([[self.I_h*self.dt,empty_quad],[empty_half]])
        self.R =self.R_imu+self.R_gps
        self.G = np.vstack(np.hstack([np.ones(self.half)*(1/2*self.dt_gps)**2, np.ones(self.half)*self.dt/2]))
        print('G\n',self.G)
        self.Q = np.diagflat(self.G)
       
        print('Q\n',self.Q)

    def get_accel_imu(self, imu_data:np.array)->np.array:
        ax = imu_data[0]
        ay = imu_data[1]
        az = imu_data[2]
        gx = imu_data[3]
        gy = imu_data[4]
        gz = imu_data[5]
        mx = imu_data[6]
        my = imu_data[7]
        mz = imu_data[8]
        R = np.sqrt(ax*ax + ay*ay + az*az)
        pitch = np.arctan(ax/R)
        roll = np.arctan(ay/R)
        magx = mx*np.cos(pitch) + my*np.sin(roll)*np.sin(pitch) + mz*np.cos(roll)*np.sin(pitch)
        magy = my*np.cos(roll) - mz*np.sin(roll)
        yaw = np.arctan2(-magy,magx)
        Rx = np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
        print(Rx)
        print(np.array([1,np.cos(yaw)]))
        Ry = np.array([[np.cos(yaw),0,-np.sin(yaw)],[0,1,0],[np.sin(yaw),0,np.cos(yaw)]])
        print(Ry)
        Rz = np.array([[np.cos(pitch),-np.sin(pitch),0],[np.sin(pitch),np.cos(pitch),0],[0, 0, 1]])
        print(Rz)
        Rot = Rx@Ry@Rz
        real_move = np.array([ax,ay,az]).T@Rot
        return real_move


    def predict(self)-> None:
        self.states = self.A@self.states
        self.P = self.A@self.P@self.A.T + self.Q

        
    def get_states(self)->np.array:
        return self.states

    
    def update_imu(self, new_imu:np.array)->None:
        y = np.array([0,0, *new_imu]) - self.H_imu@self.states
        S = self.H_imu@self.P@self.H_imu.T + self.R
        K = (self.P@self.H_imu.T@np.linalg.inv(S))
        self.states = self.states + K@y
        self.P = (self.I-(K@self.H_imu))@self.P

    def update_gps(self, new_gps:np.array)->None:
            y = np.array([*new_gps,0,0]) - self.H_gps@self.states
            S = self.H_gps@self.P@self.H_gps.T + self.R
            K = (self.P@self.H_gps.T@np.linalg.inv(S))
            self.states = self.states + K@y
            self.P = (self.I-(K@self.H_gps))@self.P
        

    def set_gps_pos(self, new_gps:np.array)->None:
        self.states= np.array(np.vstack(np.vstack(new_gps),np.vstack(self.states[self.half:self.nx])))
        

dt = 0.1
dt_gps = 2
t =  100
b = int(t/dt)
x = np.linspace(0, t,b)
y = x*np.pi
x1 =  np.cos(y*100)
x2 =  np.sin(y*100)
x3 = -np.sin(y*100)
x4 =  np.cos(y*100)
y = np.block([x1,x2,x3,x4])
print(y[0:4])
k = khalman_gps_imu(y[0:4],dt,dt_gps)
i = 0
l = int(len(y)/4)
print(len(y))
e = np.zeros([l,1])
y1 = np.zeros([l,1])
y2 = np.zeros([l,1])
y3 = np.zeros([l,1])
y4 = np.zeros([l,1])
print()
mod = 1/dt
m = 1/dt_gps
k.update_gps([x1[0],x2[0]])
k.predict()

for t in x:
    #k.update_imu(y[i:i+4])
    if i%(m) == 0:
            
        k.update_gps([x1[i],x2[i]])
    if i%mod == 0:
        
        k.update_imu(np.array([x3[i],x4[i]]))
 
    k.predict()
    z = k.get_states()
    e[i] = np.linalg.norm(y[i:i+4]-z)
    y1[i]= z[0]
    y2[i]= z[1]
    y3[i]= z[2]
    y4[i]= z[3]
    print(k.get_accel_imu(np.array([*z,1,1,1,1,1])))
    
    i += 10

plt.figure()
plt.subplot(211)
plt.plot(x1,x2)
plt.plot(y1,y2,'r--')
plt.subplot(212)
plt.plot(x3,x4)
plt.plot(y3,y4,'r--')
plt.waitforbuttonpress(0)
plt.close()