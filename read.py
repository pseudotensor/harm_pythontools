import numpy as np
import array
from scipy.interpolate import griddata
#from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import ma
import matplotlib.colors as colors

#read gdump header
gin = open( "dumps/gdump", "rb" )
header = gin.readline().split()
nx = int(header[1])
ny = int(header[2])
nz = int(header[3])
_dx1=float(header[7])
_dx2=float(header[8])
_dx3=float(header[9])
gin.close()
#read gdump
ti,tj,tk,x1,x2,x3,r,h,ph,gdet = np.loadtxt( "dumps/gdump", dtype=float, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 105), unpack = True ).view().reshape((10,nx,ny,nz), order='F')
#read image
fin = open( "dumps/fieldline0250.bin", "rb" )
header = fin.readline().split()
t = np.float64(header[0])
nx = int(header[1])
ny = int(header[2])
nz = int(header[3])
_dx1=float(header[7])
_dx2=float(header[8])
_dx3=float(header[9])
body = np.fromfile(fin,dtype=np.single,count=nx*ny*nz*11)
d=body.view().reshape((11,nx,ny,nz),order='F')
#rho, u, -hu_t, -T^t_t/U0, u^t, v1,v2,v3,B1,B2,B3
rho=d[0,:,:,:].view()
B1=d[8,:,:,:].view()
B2=d[9,:,:,:].view()
B3=d[10,:,:,:].view()
#body = array.array('f')
#body.fromfile(fin,nx*ny*nz*11)
fin.close()
daphi=gdet*B1
absflux=abs(daphi*_dx2*_dx3).sum(2).sum(1)

xraw=r*np.sin(h)
yraw=r*np.cos(h)
lrhoraw=np.log10(rho)

x=xraw[:,:,0].view().reshape(-1)
y=yraw[:,:,0].view().reshape(-1)
lrho=lrhoraw[:,:,0].view().reshape(-1)

#mirror
x=np.concatenate((-x,x))
y=np.concatenate((y,y))
lrho=np.concatenate((lrho,lrho))

extent=(-41,41,-41,41)
# define grid.
xi = np.linspace(-41.0, 41.0, 800)
yi = np.linspace(-41.0, 41.0, 400)
# grid the data.
zi = griddata((x, y), lrho, (xi[None,:], yi[:,None]), method='linear')

interior = np.sqrt((xi[None,:]**2) + (yi[:,None]**2)) < 1+np.sqrt(1-0.9**2)
#zi[interior] = np.ma.masked
zim = ma.masked_where(interior, zi)

palette=cm.jet
palette.set_bad('k', 1.0)
palette.set_over('r', 1.0)
palette.set_under('g', 1.0)

# contour the gridded data, plotting dots at the randomly spaced data points.
cset2 = plt.contour(zi,15,linewidths=0.5,colors='k', extent=extent,hold='on',origin='lower')
#for c in cset2.collections:
#    c.set_linestyle('solid')

#CS = plt.contourf(xi,yi,zi,15,cmap=palette)

#CS = plt.imshow(zim, extent=[0.01,80,-40,40], cmap = palette, norm = colors.Normalize(vmin=-1,vmax=-0.2,clip = False))
CS = plt.imshow(zim, extent=extent, cmap = palette, norm = colors.Normalize(clip = False),origin='lower')
#CS.cmap=cm.jet
#CS.set_axis_bgcolor("#bdb76b")

plt.colorbar(CS) # draw colorbar
plt.xlim(-40,40)
plt.ylim(-40,40)

plt.title('Density test')
plt.show()


# rbf = Rbf(x[0:288:8,:,0].view().reshape(-1),y[0:288:8,:,0].view().reshape(-1),rho[0:288:8,:,0].view().reshape(-1),epsilon=2)
# ZI = rbf( XI, YI )

# # plot the result
# n = plt.normalize(0.0, 40.0)
# plt.subplot(1, 1, 1)
# plt.imshow(XI, YI, ZI, cmap=cm.jet)
# #plt.scatter(x, y, 100, z, cmap=cm.jet)
# plt.title('RBF interpolation - multiquadrics')
# plt.xlim(0, 40.0)
# plt.ylim(0, 40.0)
# plt.colorbar()


# plt.figure()
# plt.plot(r[:,0,0],np.log10(absflux),'b')
# plt.legend(['Grid'])
# plt.axis([r[0,0,0],100,-5,5])
# plt.title('Grid plot')
# plt.show()


