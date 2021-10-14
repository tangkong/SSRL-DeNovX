# this is just a test script, to be implemented in helpers.py
# it is grossly 

import numpy as np
from matplotlib import pyplot as plt

# circle diameter and center coordinates
d = 10
C = np.array([d/2,d/2])

#box width 
w = 2
h = 2

# inscribe a box
n = np.int(np.ceil(d/w))
m = np.int(np.ceil(d/h))
#check if the above are even; if not, add 1
if n % 2 != 0:
    n = n+1
if m % 2 != 0:
    m = m+1


# generate an array for the inscribing box
# calculate start and end points for the box
#xst = C[0] - (w*n)/2
#xen = C[0] + (w*n)/2
#yst = C[1] - (h*n)/2
#yen = C[1] + (h*n)/2

xarr = np.linspace(C[0]-w*(n/2),C[0]+w*(n/2),num=50,endpoint=True)
yarr = np.linspace(C[1]-h*(m/2),C[1]+h*(m/2),num=50,endpoint=True)

# generate a grid
xx, yy = np.meshgrid(xarr,yarr)

# list to store all teh acceptable positions
xacc = []
yacc = []
# okay now step through points on the grid and see if the corners of the box
# they define fall outside the radius of the circle
for ii in xarr:
    for jj in yarr:
        bnds = [C - [ii-w/2,jj],C -[ii+w/2,jj],
                C - [ii,jj-h/2],C - [ii,jj+h/2]]
        test = [0,0,0,0]
        for ind,bnd in enumerate(bnds):
            if np.linalg.norm(bnd) > d/2:
                test[ind] = 1

        if 1 in test:
            continue
        else:
            xacc.append(ii)
            yacc.append(jj)

print(xacc)
print(yacc)
# plot things up to visualize
fig,ax = plt.subplots()
ax.plot(xx,yy,'r*')
circle1 = plt.Circle((5,5),5,fill=False)
ax.add_patch(circle1)
ax.plot(xacc,yacc)
plt.show()
