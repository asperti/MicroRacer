import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms
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
def border_poly(cs,theta,track_width):
    n = len(theta)/2
    xin,yin,xout,yout = borders(cs,theta,track_width/2)
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

# calculates the intersection between the line generated from two points on the track
# and the line generated from a point in the middle of the previous two and the origin
# Necessary to maintain the right theta of the mid point 
def mid_point(first, second, mid):
    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]
    x3 = mid[0]
    y3 = mid[1]
    a = (x1*y3/x3)-y1*((x2-x1)*y3/x3)/(y2-y1)
    b = 1 - ((x2-x1)*y3/x3)/(y2-y1)
    y = a/b
    x = y*x3/y3
    return x,y

# Generates a chicane selecting one random theta and inserting 2 points after and 2 points before
# Aligns these points and then offsets the points before and after 
# The external added points are used to improve the tightness of the curves
def generate_chicanes(y,theta, var, curves):
    curves_tightness_ratio = 3
    curves_proximity_ratio = 25
    offset = 0.02
    center = np.random.randint(1, curves-2)
    dist_r = abs(theta[center-1]-theta[center])/curves_tightness_ratio
    dist_o =  abs(theta[center-1]-theta[center])/curves_proximity_ratio
    v_f = var[center]
    t_f = theta[center]
    var = np.insert(var,center+1, v_f)
    var = np.insert(var,center, v_f)
    theta = np.insert(theta,center+1,t_f + dist_r) 
    theta = np.insert(theta,center, t_f - dist_r)
    center += 1
    var = np.insert(var,center+1, v_f)
    var = np.insert(var,center, v_f)
    theta = np.insert(theta,center+1,t_f + dist_o) 
    theta = np.insert(theta,center, t_f - dist_o)
    y = np.c_[np.cos(theta)*var, np.sin(theta)*var]
    center += 1
    y[center]=mid_point(y[center-3],y[center+3],y[center])
    y[center-1]=mid_point(y[center-3],y[center+3],y[center-1])
    y[center-2]=mid_point(y[center-3],y[center+3],y[center-2])
    y[center+1]=mid_point(y[center-3],y[center+3],y[center+1])
    y[center+2]=mid_point(y[center-3],y[center+3],y[center+2])
    if np.random.choice(a=[False, True]):
        y[center-1] = y[center-1]+y[center-1]*offset
        y[center+1] = y[center+1]-y[center+1]*offset      
    else:
        y[center-1] = y[center-1]-y[center-1]*offset
        y[center+1] = y[center+1]+y[center+1]*offset
    return y, theta, var
    
def create_random_track(curves=20,track_width=.04, chicanes=False):
    theta = 2 * np.pi * np.linspace(0, 1, curves)
    var = np.random.rand(curves)
    var = smooth(var)
    var = var*.5+.7
    var[curves-1]=var[0]
    #midline
    y = np.c_[np.cos(theta)*var, np.sin(theta)*var]
    if chicanes:
        y, theta, var = generate_chicanes(y, theta, var, curves)
    cs = CubicSpline(theta, y, bc_type='periodic')
    theta2 = 2 * np.pi * np.linspace(0, 1, 6*curves)
    csin,csout,_,_ = border_poly(cs,theta2,track_width)
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

# generates the points for the obstacles choosing randomly which side of the road to take
# similar way as how the outer borders of the track are generated
def obs_points(csout,theta,track_width=.04, obs_length=np.pi/150):
    obs_side = np.random.choice([track_width*.85,track_width*.15], size=len(theta))
    d = csout(theta,1)
    dx = d[:,0]
    dy = d[:,1]
    pdx, pdy = -dy, dx
    corr1 = obs_side/np.sqrt(pdx**2+pdy**2)
    pos = csout(theta)
    x = pos[:,0]
    y = pos[:,1]
    
    theta2 = theta - obs_length
    d2 = csout(theta2,1)
    dx2 = d2[:,0]
    dy2 = d2[:,1]
    pdx2, pdy2 = -dy2, dx2
    corr2 = obs_side/np.sqrt(pdx2**2+pdy2**2)
    pos2 = csout(theta2)
    x2 = pos2[:,0]
    y2 = pos2[:,1]
    
    return x+pdx*corr1,y+pdy*corr1, x2+pdx2*corr2, y2+pdy2*corr2

# Places obstacle on the map matrix in a similar way as the map is filled
# dim is half the thickness of the obstacle
def map_obs(x_i,y_i,x_o,y_o, map, track_width): 
    dim = int(track_width*500/4)
    for k in range(len(x_i)):
        x0, y0 = x_i[k], y_i[k]
        x1, y1 = x_o[k], y_o[k]
        x0 = int(x0 * 500) + 650
        y0 = int(y0 * 500) + 650
        x1 = int(x1 * 500) + 650
        y1 = int(y1 * 500) + 650
        dx = x1-x0
        dy = y1-y0
        if not (dx == 0 and dy == 0):
            if abs(dx) >= abs(dy):
                if x0 < x1:
                    xstep = 1
                else:
                    xstep = -1
                ystep = dy/dx
                for i in range (0,dx+xstep,xstep):
                    j = int(ystep*i)
                    map[x0+i-dim:x0+i+dim,y0+j-dim:y0+j+dim] = 0     
            else:
                if y0 < y1:
                    ystep = 1
                else:
                    ystep = -1
                xstep = dx/dy
                for j in range (0,dy+ystep,ystep):               
                    i = int(xstep*j)
                    map[x0+i-dim:x0+i+dim,y0+j-dim:y0+j+dim] = 0    
    return map
    

def generate_obstacles(csout,obstacles_n,map,track_width):
    obstacles_n = obstacles_n + 2
    theta  = np.random.uniform(low=(2*np.pi)/(obstacles_n*2), high=(2*np.pi)/obstacles_n, size=obstacles_n)
    for i in range(len(theta)-1):
        theta[i] = i*(2*np.pi)/obstacles_n + theta[i]              
    x_1,y_1,x_2,y_2 = obs_points(csout, theta[1:-1], track_width)
    map = map_obs(x_2,y_2,x_1,y_1, map, track_width)   
    return map, np.stack([x_1, x_2, y_1, y_2], axis=1)

def create_route_map(inner,outer,discr=2000,show_map=False):
    map = np.zeros((1300,1300)).astype('bool')
    rad = 2 * np.pi / discr
    for theta in range(discr):
        #print(theta)
        xin,yin = inner(theta*rad)
        xout,yout = outer(theta*rad)
        xin = int(xin * 500) + 650
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

# custom observation of the state
# we extract from the lidar signal the angle dir corresponding to maximal distance max_dir from track borders
# as well as the the distance at adjacent positions.

def max_lidar(observation,angle=np.pi/3,pins=19):
    arg = np.argmax(observation)
    dir = -angle / 2 + arg * (angle / (pins - 1))
    dist = observation[arg]
    if arg == 0:
        distl = dist
    else:
        distl = observation[arg-1]
    if arg == pins-1:
        distr = dist
    else:
        distr = observation[arg+1]
    return(dir,(distl,dist,distr))


def observe(racer_state):
    if racer_state == None:
        return np.array([0]) #not used; we could return None
    else:
        lidar_signal, v = racer_state
        dir, (distl,dist,distr) = max_lidar(lidar_signal)
        return np.array([dir, distl, dist, distr, v])

def lidar_grid(x,y,vx,vy,map,angle=np.pi/3,pins=19):
    return lidar.lidar_grid(x,y,vx,vy,map,angle,pins)

    
#######################################################################################################################

class Racer:
    def __init__(self, obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=True):
        self.curves = 20
        self.tstep = 0.04
        self.max_acc = 0.1
        self.max_turn = np.pi/6
        self.track_width = 0.04
        self.obstacles_number = 6
        self.obstacles = obstacles
        self.chicanes = chicanes
        self.turn_limit = turn_limit
        self.low_speed_termination = low_speed_termination

    
    def reset(self, shared_map=None):
        if shared_map == None:
            legal_map = False
            #map creation may fail in pahtological cases
            #we try until a legal map is created
            while not(legal_map):
                self.cs,self.csin,self.csout = create_random_track(self.curves,self.track_width , chicanes=self.chicanes)
                self.map,legal_map = create_route_map(self.csin, self.csout)
                if self.obstacles:
                    _ , self.obs_pos = generate_obstacles(self.csout, self.obstacles_number, self.map, self.track_width)
                else:
                    self.obs_pos = []
        else:
            cs, map, obs_pos, init_v = shared_map
            self.cs = cs
            self.map = map
            self.obs_pos = obs_pos
        self.cartheta = 0 # polar angle w.r.t center of the route
        self.carx,self.cary = self.cs(0)
        self.carvx,self.carvy = -self.cs(0,1)
        self.done = False
        self.completation = 0
        if shared_map == None:
            if self.low_speed_termination:
                v = max(np.random.uniform()*.5, 0.07)
            else: 
                v = np.random.uniform()*.5
        else:
            v = init_v
        print("initial speed = ", v)
        self.sinit_v = v
        vnorm = v/((self.carvx ** 2 + self.carvy ** 2) ** .5)
        self.carvx *= vnorm
        self.carvy *= vnorm
        assert (self.map[int(self.carx*500)+650, int(self.cary*500)+650])
        lidar_signal = lidar_grid(self.carx,self.cary,self.carvx,self.carvy,self.map)
        #print("distance = {}, direction = {}".format(dist,dir))
        return (observe([lidar_signal, v]))

   
    def step(self,action):
        max_incr = self.max_acc*self.tstep
        acc,turn = action
        v = (self.carvx**2 + self.carvy**2)**.5
        newv = max(0,v+acc*max_incr)
        max_vturn = self.max_turn
        cardir = np.arctan2(self.carvy,self.carvx) 
        if self.turn_limit:
            if newv>0:
                max_vturn = min(self.max_turn, (1.741*self.tstep)/newv) 
            newdir = cardir - turn*max_vturn
        else:
            newdir = cardir - turn*max_vturn
        self.max_turnrad = max_vturn
        newcarvx = newv * np.cos(newdir)
        newcarvy = newv * np.sin(newdir)
        newcarx = self.carx + newcarvx*self.tstep
        newcary = self.cary + newcarvy*self.tstep
        newcartheta = np.arctan2(newcary,newcarx)
        on_route = self.map[int(newcarx*500)+650, int(newcary*500)+650]
        if on_route and no_inversion(newcartheta, self.cartheta):
            if newv<0.05 and self.low_speed_termination:
                print("too slow")
                self.completation = 4
                self.done = True
                reward = -3
                state = None
                return(observe(state),reward,True)
            self.carx = newcarx
            self.cary = newcary
            self.carvx = newcarvx
            self.carvy = newcarvy
            reward = newv*self.tstep
            lidar_signal = lidar_grid(self.carx, self.cary, self.carvx, self.carvy, self.map)
            if complete(newcartheta, self.cartheta):
                print("completed")
                self.completation = 1
                self.done = True
            self.cartheta = newcartheta
            n_state = observe([lidar_signal, newv])
            return (n_state,reward,self.done)
        else:
            if not(on_route):
                print("crossing border")
                self.completation = 2
            else:
                print("wrong direction")
                self.completation = 3
            self.done = True
            reward = -3
            state = None
        return(observe(state),reward,True)
    
    

def pilot(actor, state, i):
    verbose = False
    action = actor(np.expand_dims(state, 0))
    if len(action)>1:
        action = action[0]
    if verbose:
        print("speed {}° car = {}".format(i+1,state[4]))
        print("acc {}° car = {}".format(i+1,action[0,0].numpy()))
        print("turn {}° car = {}".format(i+1,action[0,1].numpy()))
    return(action[0])
 
# requires a list of models that returns as first element the action tuple  
def newrun(actors, obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=True):
    debug_dir = False
    verbose_action = True
    podium = []
    order = []
    actor_n = len(actors)
    racers = []
    states = []
    xdatas = []
    ydatas = []
    for i, actor in enumerate(actors):
        racer = Racer(obstacles, turn_limit, chicanes, low_speed_termination)
        racers.append(racer)
        if i==0:
            states.append(racers[i].reset())
            cs,csin,csout, obs_pos = racers[i].cs,racers[i].csin,racers[i].csout, racers[i].obs_pos
        else:
            states.append(racers[i].reset((racers[0].cs, racers[0].map, racers[0].obs_pos, racers[0].sinit_v)))
        xdatas.append([racers[i].carx])
        ydatas.append([racers[i].cary])
        order.append(racer.cartheta)
    fig, axs = plt.subplots(ncols=2, dpi=180)
    #fig.canvas.manager.window.state('zoomed') #not supported on linux
    ax = axs[0]
    #plot map
    ax.imshow(np.rot90(racers[0].map), extent=[-1.3, 1.3, -1.3, 1.3], cmap='gray', vmin=-1, vmax=1)
    #plot borders
    xs = 2 * np.pi * np.linspace(0, 1, 200)
    ax.plot(csin(xs)[:, 0], csin(xs)[:, 1], color='black')
    ax.plot(csout(xs)[:, 0], csout(xs)[:, 1], color='black')
    #plot obstacles
    for i in range(len(obs_pos)):
        ax.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.6, color='crimson')
    ax.axes.set_aspect('equal')
    #racers lines
    lines = []
    plotcols = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']
    for i in range(actor_n):
        lobj = ax.plot([], [], lw=1.5, label="label", marker='s', markersize=3 , color=plotcols[i])[0]
        lines.append(lobj) 
    #ax.axis(False)
    
    #legend placement  
    axl = axs[1]
    axl.set_xlim([-1.3, 1.3])
    axl.set_ylim([-1.3, 1.3])
    axl.set_aspect('equal')
    label_params = ax.get_legend_handles_labels() 
    axl.axis(False)
    L = axl.legend(*label_params, loc="upper left",prop={'size': 6} )
    fig.tight_layout(pad=0.1)
    

    ### DEBUG DIRECTION AND TURNING RANGE LINES   
    if debug_dir:
        dir_line = ax.plot([], [], lw=1.5, color="orange")[0]
        lines.append(dir_line)
        max_dir_line1 = ax.plot([], [], lw=1.5, color="blue")[0]
        max_dir_line2 = ax.plot([], [], lw=1.5, color="blue")[0]
        lines.append(max_dir_line1)
        lines.append(max_dir_line2)
        
    def init():
        for line in lines:
            line.set_data([],[])
        return lines + [L]

    def counter():
        n = 0
        while not(all(racer.done for racer in racers)):
            n += 1
            yield n
            
    def animate(i):
        a = []
        [a.append("") for z in range(len(actors))]
        for j, actor in enumerate(actors):
            label = ''
            if not racers[j].done:
                if racers[j].cartheta < 0:
                    order[j]=racers[j].cartheta+2*np.pi
                else:
                    order[j]=racers[j].cartheta
                action = pilot(actor,states[j],j)
                if action[0]< 0:
                    label += "decelerating"
                elif action[0]> 0:
                    label += "accelerating"
                if verbose_action:
                    label += ": {}{:.2f} |".format(" " if action[0].numpy()>=0 else "",action[0].numpy())
                #if abs(action[1])>0.9:
                #    label += " sharp"
                if action[1]<0:
                    label += " left  "
                elif action[1]>0:
                    label += " right"
                if verbose_action:
                    label += ": {}{:.2f}".format(" " if action[1].numpy()>=0 else "", action[1].numpy())
                label += " | speed: {:.2f}".format(states[j][4])
                states[j],_,_= racers[j].step(action)
                if racers[j].completation == 1:
                    podium.append(j)
                a[j]=label
            xdatas[j].append(racers[j].carx)
            ydatas[j].append(racers[j].cary)      
        for lnum, line in enumerate(lines):
            if lnum < len(actors):
                #we use the order to show the cars behind on an upper level on the graph 
                line.set(xdata=xdatas[lnum], ydata=ydatas[lnum], zorder=order[lnum], markevery=[-1] )
                if not racers[lnum].done:
                    L.get_texts()[lnum].set_text(a[lnum])
                elif racers[lnum].completation == 1:
                    L.get_texts()[lnum].set_text("{}° place".format(podium.index(lnum)+1))
                elif racers[lnum].completation == 2:
                    L.get_texts()[lnum].set_text("Off road")
                elif racers[lnum].completation == 3:
                    L.get_texts()[lnum].set_text("Wrong direction")
                elif racers[lnum].completation == 4:
                    L.get_texts()[lnum].set_text("Under speed limit")

        
        ### DEBUG DIRECTION AND TURNING RANGE LINES 
        if not racers[0].done and debug_dir:
            single_s =states[0]
            dist_lid = single_s[2]
            single_s = single_s[0]+np.arctan2(racers[0].carvy,racers[0].carvx)
            lines[-3].set(xdata=[racers[0].carx,racers[0].carx+np.cos(single_s)*(dist_lid/((np.cos(single_s)**2+np.sin(single_s)**2)**.5))], ydata=[racers[0].cary,racers[0].cary+np.sin(single_s)*(dist_lid/((np.cos(single_s)**2+np.sin(single_s)**2)**.5))])
            min_angle = np.arctan2(racers[0].carvy,racers[0].carvx)-racers[0].max_turnrad
            lines[-2].set(xdata=[racers[0].carx,racers[0].carx+np.cos(min_angle)], ydata=[racers[0].cary,racers[0].cary+np.sin(min_angle)])
            max_angle = racers[0].max_turnrad+np.arctan2(racers[0].carvy,racers[0].carvx)
            lines[-1].set(xdata=[racers[0].carx,racers[0].carx+np.cos(max_angle)], ydata=[racers[0].cary,racers[0].cary+np.sin(max_angle)])
        
        return lines + [L]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=counter, interval=5, blit=True, repeat=False)
    
    plt.show()
    
    print("###SCOREBOARD###")
    for i, carn in enumerate(podium):
        print("{}° place : Car N.{}".format(i+1, carn+1))
    for i, racer in enumerate(racers):
        if racer.completation == 2:
            print("Car N.{} went off road".format(i+1))
        if racer.completation == 3:
            print("Carn N.{} went to the wrong direction".format(i+1))
        if racer.completation == 4:
            print("Carn N.{} went under speed limits".format(i+1))
    
  
# launches runs_num evaluations episodes with no plots and measures average steps and rewards   
# requires a model that returns as first element the action tuple        
def metrics_run(actor, runs_num = 100, obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=True):
    ep_reward_list = []
    ep_meanspeed_list = []
    ep_steps_list = []
    completation = 0
    racer = Racer(obstacles, turn_limit, chicanes, low_speed_termination)
    i = 0  
    for ep in range(runs_num):
        state = racer.reset()
        done = False
        steps = 0
        episodic_reward = 0
        mean_speed = 0
        while not done:    
            i+=1
            state = np.expand_dims(state, 0)
            action = actor(state)
            if len(action)>1:
                action = action[0]
            action = (action[0])
            nstate, reward, done = racer.step(action)
            steps+=1
            if not done:
                mean_speed += nstate[4]
            state = nstate
            episodic_reward += reward       
        ep_reward_list.append(episodic_reward) 
        ep_meanspeed_list.append(mean_speed/steps)
        ep_steps_list.append(steps)
        if racer.completation == 1:
            completation+=1
        print("Episode {}: Steps = {}, Ep. reward = {}. Avg. Ep. speed = {}".format(ep, steps,episodic_reward,mean_speed/steps))

    print("Completed episodes: {}/{}".format(completation,runs_num))
    print("Avg reward over {} episodes = {} / Avg steps = {} / Avg speed = {}".format(runs_num, np.mean(ep_reward_list), np.mean(ep_steps_list), np.mean(ep_meanspeed_list)))
    print("\n")
