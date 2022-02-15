import numpy as np

#cython syntax used to assign types to variables


#basic function for lidar. given a point (x0,y0) on the track and a direction dirang computes the
#distance of the border along the given direction
cpdef dist_grid(double x0,double y0,double dirang, map,float step=1./250,verbose=False):
    cdef double stepx = step*np.cos(dirang)
    cdef double stepy = step*np.sin(dirang)
    cdef double x = x0
    cdef double y = y0
    cdef int xg = int(x * 500) + 650
    cdef int yg = int(y * 500) + 650
    cdef bint check
    cdef bint up
    cdef bint down
    cdef bint left
    cdef bint right
    cdef bint center
    cdef int i = 0
    if not map[xg,yg]:
        print(x,y,xg,yg)
        print(map[xg,yg])
        assert(map[xg,yg])
    check = map[xg,yg]
    while (check):
        if i == 10:
            step = 1./100
            stepx = step*np.cos(dirang)
            stepy = step*np.sin(dirang)
        x += stepx
        y += stepy
        xg = int(x * 500) + 650
        yg = int(y * 500) + 650
        up = map[xg,yg+1]
        down = map[xg,yg-1]
        left = map[xg-1,yg]
        right = map[xg+1,yg]
        center = map[xg,yg]
        check = up and down and left and right
        i += 1
    x -= stepx
    y -= stepy
    if step == 1./100:
        #print("reducing step")
        x, y = dist_grid(x,y, dirang,map,step=1./500,verbose=False)
    if verbose:
        print("start at = {}, cross border at {}".format((x0,y0),(x,y)))
    return x,y


cpdef lidar_grid(double x,double y,double vx,double vy,map,float angle=np.pi/3, int pins=19):
    cdef double dirang = np.arctan2(vy, vx) #car direction
    obs = np.zeros(pins)
    cdef int i = 0
    cdef double a = dirang - angle/2
    cdef float astep = angle/(pins-1)
    cdef double cx
    cdef double cy
    for i in range(pins):
        cx,cy = dist_grid(x,y,a,map,verbose=False)
        obs[i] = ((cx-x)**2+(cy-y)**2)**.5
        a += astep
    return obs