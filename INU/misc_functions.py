import numpy as np

def getPitch(ax,ay,az):
    R = np.sqrt(ax*ax + ay*ay + az*az)
    return np.arctan(ax/R)

def getRoll(ax,ay,az):
    R = np.sqrt(ax*ax + ay*ay + az*az)
    return np.arctan(ay/R)

def getYaw(mx,my,mz,pitch,roll):
    magx = mx*np.cos(pitch) + my*np.sin(roll)*np.sin(pitch) + mz*np.cos(roll)*np.sin(pitch)
    magy = my*np.cos(roll) - mz*np.sin(roll)
    return np.arctan2(-magy,magx)

def Rx( angle):
    return np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])

def Ry(angle):
    return np.array([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])

def Rz(angle):
    return np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0, 0, 1]])
    
def get_rotmat(x,y,z):
    rx = Rx(x)
    ry = Ry(y)
    rz = Rz(z)
    return rx@ry@rz

def lat_long_to_cart(lat,lon):
    re = 6378000 # earth radius at equator.
    rp = 6356000 # earth radius at poles.
    R = np.sqrt(((re^2*np.cos(lat))+rp^2*np.sin(lat))/((re^2*np.cos(lat)+rp^2*np.sin(lat))^2))
    x = R*(np.cos(lat)*np.cos(lon))
    y = R*(np.cos(lat)*np.sin(lon))
    z = R*np.sin(lat)
    return np.array([x,y,z])