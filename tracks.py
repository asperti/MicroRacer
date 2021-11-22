import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import pathlib

#generate the compiled and converted files for lidar.pyx using cython in the directory .pyxbld
#auto recompile them at every edit on lidar.pyx
pyxbld_dir=pathlib.PurePath.joinpath(pathlib.Path().resolve(), '.pyxbld') 
import pyximport; pyximport.install(build_dir=pyxbld_dir,reload_support=True, language_level=3)
import lidar  



#find border positions at theta, given the midline cs
#need to move along the perpendicular of the midline direction
def borders(cs,theta,track_semi_width=.02):
    d = cs(theta,1)
    dx = d[:,0]
    dy = d[:,1]
    #print(dx, dy)
    pdx, pdy = -dy, dx
    corr = track_semi_width/np.sqrt(pdx**2+pdy**2)
    pos = cs(theta)
    x = pos[:,0]
    y = pos[:,1]
    return x+pdx*corr,y+pdy*corr,x-pdx*corr,y-pdy*corr

# adjust range in increasing order between 0 and 2*np.pi as required by CubicSpline
def adjust_range(angles,n):
    incr = 0
    last = 0
    fixed = []
    for i,theta in enumerate(angles):
        theta = theta+incr
        if theta < last:
            incr += 2*np.pi
            theta += 2*np.pi
        fixed.append(theta)
        last = theta
    return(fixed)

#takes in input a midline, that is a CubicSpline at angles theta, and
#returns border lines as similar Splines
def border_poly(cs,theta):
    n = len(theta)/2
    xin,yin,xout,yout = borders(cs,theta)
    cin = list(zip(xin,yin))
    cout = list(zip(xout,yout))
    thetain = [np.arctan2(y,x) for (x,y) in cin]
    thetain = adjust_range(thetain,n)
    thetaout = [np.arctan2(y,x) for (x,y) in cout]
    thetaout = adjust_range(thetaout,n)
    csin = CubicSpline(thetain, cin, bc_type='periodic')
    csout = CubicSpline(thetaout, cout, bc_type='periodic')
    return(csin,csout,thetain,thetaout)

#we try to avoid too sharp turns in tracks
def smooth(var):
    n = var.shape[0]
    if 2*var[0]-(var[n-1]+var[1]) > 1:
        var[0] = (1 + var[n-1]+var[1])/2
    elif 2*var[0]-(var[n-1]+var[1]) < -1:
        var[0] = (var[n-1] + var[1] -1) / 2
    for i in range(1,n-1):
        if 2*var[i]-(var[i-1]+var[i+1]) > 1:
            var[i] = (1 + var[i-1] + var[i+1]) / 2
        elif 2*var[i]-(var[i-1]+var[i+1]) < -1:
            var[i] = (var[i-1] + var[i+1]-1) / 2
    if 2*var[n-1]-(var[n-2]+var[0]) > 1:
        var[n-1] = (1 + var[n-2]+var[0])/2
    elif 2*var[n-1]-(var[n-2]+var[0]) < -1:
        var[n-1] = (var[n-2]+var[0]-1)/2
    return(var)


def create_random_track(curves=20):
    theta = 2 * np.pi * np.linspace(0, 1, curves)
    var =np.random.rand(curves)
    var = smooth(var)
    var = var*.5+.7
    var[curves-1]=var[0]
    #midline
    y = np.c_[np.cos(theta)*var, np.sin(theta)*var]
    cs = CubicSpline(theta, y, bc_type='periodic')
    theta2 = 2 * np.pi * np.linspace(0, 1, 2*curves)
    csin,csout,_,_ = border_poly(cs,theta2)
    return (cs,csin,csout)

def no_inversion(thetanew,thetaold):
    if thetaold < -np.pi*.9 and thetanew > np.pi*.9:
        thetanew = thetanew-np.pi*2
    return(thetanew < thetaold)

def complete(thetanew,thetaold):
    return(thetaold > 0 and thetanew <= 0)

#starting from borders we create a dense grid of points corresponding to legal
#positions on the track. This map is what defines the actual track.

#filling all points between (x0,y0) and (x1,y1) on the map. For each point
#in the line we fill a small region 3x3 around it.
def fill(x0,y0,x1,y1,map):
    #print(x0,y0,x1,y1)
    dx = x1-x0
    dy = y1-y0
    if abs(dx) >= abs(dy):
        if x0 < x1:
            xstep = 1
        else:
            xstep = -1
        ystep = dy/dx
        for i in range (0,dx+xstep,xstep):
            j = int(ystep*i)
            map[x0+i-1:x0+i+2,y0+j-1:y0+j+2] = 1
        #print(i,j)
    else:
        if y0 < y1:
            ystep = 1
        else:
            ystep = -1
        xstep = dx/dy
        for j in range (0,dy+ystep,ystep):
            i = int(xstep*j)
            map[x0+i-1:x0+i+2,y0+j-1:y0+j+2] = 1
    if not(map[x1,y1]==1):
        print(x0+i,y0+j)
    return(map.astype('bool'))


def create_route_map(inner,outer,discr=2000,show_map=False):
    map = np.zeros((1300,1300)).astype('bool')
    rad = 2 * np.pi / discr
    for theta in range(discr):
        #print(theta)
        xin,yin = inner(theta*rad)
        xout,yout = outer(theta*rad)
        xin = int(xin * 500)+650
        yin = int(yin * 500) + 650
        xout = int(xout * 500) + 650
        yout = int(yout * 500) + 650
        limit_check = xout>=0 and yout>=0 and xout<1300 and yout<1300
        if limit_check:
            fill(xin,yin,xout,yout,map)
        else:
            return(map,False)
    if show_map:
        plt.figure(figsize=(12, 6))
        plt.subplot(122)
        #plt.axis('off')
        plt.imshow(np.rot90(map))
        plt.subplot(121)
        axes = plt.gca()
        axes.set_xlim([-1.3, 1.3])
        axes.set_ylim([-1.3, 1.3])
        axes.set_aspect('equal')
        #plt.axis('off')
        xs = 2 * np.pi * np.linspace(0, 1, 200)
        plt.plot(inner(xs)[:, 0], inner(xs)[:, 1])
        plt.plot(outer(xs)[:, 0], outer(xs)[:, 1])
        #plt.axes.set_aspect('equal')
        plt.show()
    return(map,True)



def lidar_grid(x,y,vx,vy,map,angle=np.pi/3,pins=19):
    return lidar.lidar_grid(x,y,vx,vy,map,angle,pins)

#######################################################################################################################

class Racer:
    def __init__(self):
        self.curves = 20
        self.tstep = 0.1
        self.max_acc = 0.1
        self.max_turn = np.pi/6


    def reset(self):
        legal_map = False
        #map creation may fail in pahtological cases
        #we try until a legal map is created
        while not(legal_map):
            self.cs,self.csin,self.csout = create_random_track(self.curves)
            self.map,legal_map = create_route_map(self.csin, self.csout)
        self.cartheta = 0 # polar angle w.r.t center of the route
        self.carx,self.cary = self.cs(0)
        self.carvx,self.carvy = -self.cs(0,1)
        self.done = False
        v = np.random.uniform()*.5
        print("initial speed = ", v)
        vnorm = v/((self.carvx ** 2 + self.carvy ** 2) ** .5)
        self.carvx *= vnorm
        self.carvy *= vnorm
        assert (self.map[int(self.carx*500)+650, int(self.cary*500)+650])
        lidar_signal = lidar_grid(self.carx,self.cary,self.carvx,self.carvy,self.map)
        #dir, dist = max_lidar2(lidar_obs)
        #print("distance = {}, direction = {}".format(dist,dir))
        return (lidar_signal, v)

    def step(self,action):
        max_incr = self.max_acc*self.tstep
        acc,turn = action
        v = (self.carvx**2 + self.carvy**2)**.5
        newv = max(0,v+acc*max_incr)
        cardir = np.arctan2(self.carvy,self.carvx)
        newdir = cardir - turn*self.max_turn
        newcarvx = newv * np.cos(newdir)
        newcarvy = newv * np.sin(newdir)
        newcarx = self.carx + newcarvx*self.tstep
        newcary = self.cary + newcarvy*self.tstep
        newcartheta = np.arctan2(newcary,newcarx)
        on_route = self.map[int(newcarx*500)+650, int(newcary*500)+650]
        if on_route and no_inversion(newcartheta, self.cartheta):
            self.carx = newcarx
            self.cary = newcary
            self.carvx = newcarvx
            self.carvy = newcarvy
            reward = newv*self.tstep
            lidar_signal = lidar_grid(self.carx, self.cary, self.carvx, self.carvy, self.map)
            #dir,dist = max_lidar2(obs)
            if complete(newcartheta, self.cartheta):
                print("completed")
                self.done = True
            self.cartheta = newcartheta
            return ((lidar_signal,v),v*self.tstep,self.done)

        else:
            if not(on_route):
                print("crossing border")
            else:
                print("wrong direction")
            self.done = True
            reward = -3
            state = None
        return(state,reward,True)

def newrun(racer,actor):
    state = racer.reset()
    cs,csin,csout = racer.cs,racer.csin,racer.csout
    carx,cary = racer.carx,racer.cary
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = 2 * np.pi * np.linspace(0, 1, 200)
    ax.plot(csin(xs)[:, 0], csin(xs)[:, 1])
    ax.plot(csout(xs)[:, 0], csout(xs)[:, 1])
    ax.axes.set_aspect('equal')

    line, = plt.plot([], [], lw=2)
    xdata,ydata = [carx],[cary]

    acc = 0
    turn = 0

    def init():
        line.set_data([],[])
        return line,

    def counter():
        n = 0
        while not(racer.done):
            n += 1
            yield n

    def animate(i):
        nonlocal state
        #t1 = time.time()
        action = actor(state)
        #t2 = time.time()
        #print("time taken by action = {} sec.".format(t2-t1))
        #t1 = time.time()
        state,reward,done = racer.step(action)
        #t2 = time.time()
        #print("time taken by step = {} sec.".format(t2 - t1))
        xdata.append(racer.carx)
        ydata.append(racer.cary)
        line.set_data(xdata, ydata)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=counter, interval=5, blit=True, repeat=False)
    plt.show()

#racer = Racer()
#newrun(racer,myactor)