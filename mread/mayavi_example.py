export ETS_TOOLKIT=qt4
ipython --gui=wx --pylab
#ipython --gui=wx
%run -i ~/py/mread/__init__.py
cd ~/Research/code/luke_flux_capture/athena4.2/bin
rdath3d("str-flux-cap.0230.tab")
#2D plot
p=plt.imshow(np.log10(pg[:,:,n3/2].transpose()), extent=(0,n1,0,n2), cmap = cm.jet, norm = colors.Normalize(clip = True),origin='lower',interpolation="nearest",vmin=-3,vmax=3)
cbar = plt.colorbar(p)
cbar.ax.set_ylabel(r'$\log_{10}p$',fontsize=20)
plt.xlabel(r"$x$",fontsize=20)
plt.ylabel(r"$y$",fontsize=20)
#
# MAYAVI
#
from mayavi import mlab
#3D plot
mlab.pipeline.volume(mlab.pipeline.scalar_field(rho))
mlab.outline()
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(rho),
                            plane_orientation='x_axes',
                            slice_index=16,
                        )
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(rho),
                            plane_orientation='y_axes',
                            slice_index=32,
                        )
flow = mlab.flow(v1, v2, v3, seed_scale=1,
                          seed_resolution=5,
                          integration_direction='both',
                          seedtype='plane')

flow = mlab.flow(B1c, B2c, B3c, seed_scale=1,
                          seed_resolution=5,
                          integration_direction='both',
                          seedtype='plane')
