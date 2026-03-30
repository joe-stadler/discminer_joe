from discminer.mining_control import _mining_moment_residuals
from discminer.mining_utils import (get_2d_plot_decorators,
                                    init_data_and_model,                                    
                                    get_noise_mask,
                                    load_moments,
                                    load_disc_grid,
                                    mark_planet_location,
                                    show_output)

from discminer.core import Data
from discminer.rail import Contours
from discminer.plottools import (make_up_ax,
                                 mod_major_ticks,
                                 mod_nticks_cbars,
                                 use_discminer_style)

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import copy
import json

use_discminer_style()

if __name__ == '__main__':
    parser = _mining_moment_residuals(None)
    parser.add_argument('-plt_sky', '--plot_at_sky', default=1, type=int,
                    help="Plot in arcsec or au? DEFAULTS to 1 (plot in arcsec).")
    args = parser.parse_args()

def plot_beam_arcsec(self, ax, projection=None, **kwargs_ellipse):
    if self.beam is None:
        return 0
    kwargs=dict(lw=1,fill=True,fc="gray",ec="k")
    kwargs.update(kwargs_ellipse)

    dpc = self.dpc.to('pc').value
    bmaj = self.beam.major.to(u.arcsecond).value
    bmin = self.beam.minor.to(u.arcsecond).value
    if projection=='au': #plot in units of pixels (using wcs is merely decorative)
        bmaj*=dpc
        bmin*=dpc
        
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    dx = np.abs(xlim[1]-xlim[0])
    dy = np.abs(ylim[1]-ylim[0])
    xbeam, ybeam = xlim[0]+0.1*dx, ylim[0]+0.1*dy
    import matplotlib.patches as patches
    ellipse = patches.Ellipse(
        xy=(xbeam, ybeam),
        angle=90 + self.beam.pa.value,
        width=bmaj,
        height=bmin,
        transform=ax.transData,
        **kwargs
    )
    ax.add_artist(ellipse)

#**************************
#JSON AND SOME DEFINITIONS
#**************************    
with open('parfile.json') as json_file:
    pars = json.load(json_file)

meta = pars['metadata']
best = pars['best_fit']
custom = pars['custom']
rings = custom['rings']
vsys = best['velocity']['vsys']
Rout = best['intensity']['Rout']

ctitle, clabel, clim, cfmt, cmap_mom, cmap_res, levels_im, levels_cc, unit = get_2d_plot_decorators(args.moment)

if args.moment=='velocity':
    cmap_mom = copy.copy(cmap_mom)
    #cmap_mom.set_under('1')
    #cmap_mom.set_over('1')
    
    cmap_res = copy.copy(cmap_res)
    #cmap_res.set_under('1')
    #cmap_res.set_over('1')
    
#****************
#SOME DEFINITIONS
#****************
file_data = meta['file_data']
tag = meta['tag']

dpc = meta['dpc']*u.pc
Rmax = 1.1*args.Router*Rout*u.au #Max model radius, 10% larger than disc Rout

au_to_m = u.au.to('m')

#********************
#LOAD DATA AND GRID
#********************
datacube, model = init_data_and_model(Rmin=0, Rmax=Rmax)

#Useful definitions for plots
with open('grid_extent.json') as json_file:
    grid = json.load(json_file)

xmax = grid['xsky'] 
if args.plot_at_sky:
    xlim = 1.0*np.min([xmax, Rmax.value])/dpc.value
    extent= np.array([-xmax, xmax, -xmax, xmax])/dpc.value
else: 
    xlim = 1.0*np.min([xmax, Rmax.value])
    extent= np.array([-xmax, xmax, -xmax, xmax])

#*************************
#LOAD DISC GEOMETRY
R, phi, z = load_disc_grid()
noise_mean, mask = get_noise_mask(datacube, thres=args.sigma,
                                  mask_phi={'map2d': np.degrees(phi['upper']),
                                            'lims': args.mask_phi},
                                  mask_R={'map2d': R['upper']/au_to_m,
                                          'lims': args.mask_R}
)

#*************************
#LOAD MOMENT MAPS    
moment_data, moment_model, residuals, mtags = load_moments(args, mask=mask)

#**************************
#MAKE PLOT
fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15,6))
ax_cbar0 = fig.add_axes([0.15, 0.14, 0.450, 0.04])
ax_cbar2 = fig.add_axes([0.68, 0.14, 0.212, 0.04])
    
kwargs_im = dict(cmap=cmap_mom, extent=extent, levels=levels_im)
kwargs_cc = dict(colors='k', linestyles='-', extent=extent, levels=levels_cc, linewidths=0.4)
kwargs_cbar = dict(orientation='horizontal', pad=0.03, shrink=0.95, aspect=10)

im0 = ax[0].contourf(moment_data, extend='both', **kwargs_im)
im1 = ax[1].contourf(moment_model, extend='both', **kwargs_im)
im2 = ax[2].contourf(residuals, cmap=cmap_res, origin='lower', extend='both', extent=extent, levels=np.linspace(-1.01*clim, 1.01*clim, 32))

cc0 = ax[0].contour(moment_data, **kwargs_cc)
cc1 = ax[1].contour(moment_model, **kwargs_cc)

cbar0 = plt.colorbar(im0, cax=ax_cbar0, format='%.1f', **kwargs_cbar)
cbar0.ax.tick_params(labelsize=12) 
cbar2 = plt.colorbar(im2, cax=ax_cbar2, format=cfmt, **kwargs_cbar)
cbar2.ax.tick_params(labelsize=12) 

mod_nticks_cbars([cbar0], nbins=10)
mod_nticks_cbars([cbar2], nbins=5)

if args.plot_at_sky: ax[0].set_ylabel('Offset [arcsec]', fontsize=16) #15
else: ax[0].set_ylabel('Offset [au]', fontsize=16)

#ax[0].set_title(ctitle, pad=40, fontsize=17)
#ax[1].set_title('Discminer Model', pad=40, fontsize=17)
#ax[2].set_title('Residuals', pad=40, fontsize=17)

cbar0.set_label('Deprojected Folded Residuals Map', labelpad=7, fontsize=16)
cbar2.set_label(r'Residuals %s'%unit, labelpad=7, fontsize=16)

for i,axi in enumerate(ax):
    make_up_ax(axi, xlims=(-xlim, xlim), ylims=(-xlim, xlim), labelsize=12)
    mod_major_ticks(axi, axis='both', nbins=8)
    axi.set_aspect(1)

    if i>=1:
        axi.tick_params(labelleft=False)
        
    Contours.emission_surface(axi, R, phi, extent=extent, which='both',
                              R_lev=np.linspace(0.1, 1.0, 10)*Rout*au_to_m)
    
    if meta['disc']=='wispit2':
        if args.plot_at_sky: Rp1 = 0.105 #gravity
        else: Rp1 = 0.105*dpc.value
        PAp1 = np.radians(196.) #gravity
        phi1 = PAp1 + np.pi/2
        #Wispit2b
        if args.plot_at_sky: Rp2 = 0.316
        else: Rp2 = 0.316*dpc.value #0.309*dpc.value
        PAp2 = np.radians(215) #updated
        phi2 = PAp2 + np.pi/2

        axi.scatter(Rp1*np.cos(phi1), Rp1*np.sin(phi1), marker='s', s=80, fc='white', ec='k', lw=1.5, alpha=1.,zorder=4.)
        axi.scatter(Rp2*np.cos(phi2), Rp2*np.sin(phi2), marker='o', s=80, fc='white', ec='k', lw=1.5, alpha=1.,zorder=4.)
    
    #Overlay dust rings?
    Contours.emission_surface(
        axi, R, phi, extent=extent,
        R_lev=np.array(rings)*u.au.to('m'), which='upper',
        kwargs_R={'linestyles': '-', 'linewidths': 1.2, 'colors': 'k'},
        kwargs_phi={'colors': 'none'}
    )
    
    
if args.plot_at_sky:
    plot_beam_arcsec(self=datacube, ax=ax[0])
    plot_beam_arcsec(self=datacube,ax=ax[2])
else:
    plot_beam_arcsec(self=datacube, ax=ax[0], fc='none', projection='au')
    plot_beam_arcsec(self=datacube, ax=ax[2], fc='none', projection='au')

mark_planet_location(ax[2], args, edgecolors='k', lw=2.0, s=200, coords='sky', model=model, midplane=False)

#####    
add = ''        
if args.plot_at_sky:
    add += '_arcsec'

plt.savefig('moment+residuals_%s%s.png'%(mtags['base'], add), bbox_inches='tight', dpi=200)
show_output(args)

