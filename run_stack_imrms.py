import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord

# funky plot packages
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm

import stack as st

# path to COMAP map files
mapfiles = glob.glob('/media/ArchiveSix/*_summer.h5')
mapfiles = [mapfiles[0], mapfiles[2], mapfiles[1]]

print(mapfiles)

 # path to the galaxy catalogue (I made preliminary cuts before running it through)
galcatfile = '../20220601_cutquasarcat.npz'

# set up a params class that you can just pass around
params = st.empty_table()
params.xwidth = 10 # number of x pixels to average between when getting the cutout T
params.ywidth = 10 # number of y pixels to average between when getting the cutout T
params.freqwidth = 2 # number of freq pixels to average between when getting the cutout T
params.noise_sim = False #True
params.inject_signal = False #True
params.inject_signal_cubes = ['/media/ArchiveSix/20220602_limlam_Ka10_chopped_partial_masscut/13579{}_lt1e13.npz'.format(i) for i in 'abc']
params.inject_signal_catas = ['/media/ArchiveSix/20220602_limlam_Ka10_chopped_partial_masscut/13579{}_halocat_1e11.npz'.format(i) for i in 'abc']
params.inject_signal_scale = 1e-4
params.inject_signal_M_min = (1.7e13,1.7e13,1.3e13)

# cent vals (for properly centering the cutout)
length = params.xwidth // 2
params.idxmin, params.idxmax = length-1, length+1

params.centfreq = 115.27 # rest frequency CO(1-0)
params.beamwidth = 1 # when smoothing to the synthesized beam, std of gaussian kernel
params.gauss_kernel = Gaussian2DKernel(params.beamwidth)
params.tophat_kernel = Tophat2DKernel(params.beamwidth)
params.spacestackwidth = 10 # in pixels -- if you only want single T value from each cutout, set to None
params.freqstackwidth = 20 # number of channels. "" ""

# plotting parameters
params.savepath = 'output_10x10_imrms'
params.saveplots = True
params.plotspace = True
params.plotfreq = True
params.fieldcents = [SkyCoord(25.435*u.deg, 0.0*u.deg), SkyCoord(170.0*u.deg, 52.5*u.deg),
                     SkyCoord(226.0*u.deg, 55.0*u.deg)]

params.beamscale=False
beamscale = np.array([[0.25, 0.5, 0.25],
                      [0.50, 1.0, 0.50],
                      [0.25, 0.5, 0.25]])
beamscale3d = np.tile(beamscale, (params.freqwidth, 1, 1))

params.beam = beamscale3d

comaplist, qsolist = st.setup(mapfiles, galcatfile, params)

params.return_imrms = True
stacktemp, stackrms, image, spectrum, qsoidxlist, imagerms = st.stacker(comaplist, qsolist, params)
print(stacktemp,stackrms)

stacktemps = []
stackrmses = []
for iii in np.arange(1,6):
    tmp = image[image.shape[0]//2-iii:image.shape[0]//2+iii,image.shape[1]//2-iii:image.shape[1]//2+iii].flatten()
    tmprms = imagerms[image.shape[0]//2-iii:image.shape[0]//2+iii,image.shape[1]//2-iii:image.shape[1]//2+iii].flatten()
    stacktemps.append(np.sum(tmp/tmprms**2)/np.sum(1/tmprms**2)*1e6)
    stackrmses.append(np.sum(1/tmprms**2)**-0.5*1e6)
    print(stacktemps[-1],stackrmses[-1])
plt.figure(figsize=(4.2,2.4)).set_tight_layout(True)
plt.errorbar(np.arange(1,6)**2*4,stacktemps,stackrmses)
plt.xlabel('pixels stacked')
plt.ylabel(r'$T_b$ ($\mu$K)')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.grid(ls=':',alpha=0.63)
plt.savefig(params.savepath+'/signal_vs_stacksize.png',dpi=288)

stacktemps = []
stackrmses = []
imx = np.arange(image.shape[0])-image.shape[0]//2+0.5
imy = np.arange(image.shape[1])-image.shape[1]//2+0.5
imX,imY = np.meshgrid(imx,imy)
for iii in np.arange(1,6):
    tmp = image[((np.abs(imX)<iii)*(np.abs(imY)<iii))^((np.abs(imX)<iii-1)*(np.abs(imY)<iii-1))].flatten()
    tmprms = imagerms[((np.abs(imX)<iii)*(np.abs(imY)<iii))^((np.abs(imX)<iii-1)*(np.abs(imY)<iii-1))].flatten()
    print(len(tmp))
    stacktemps.append(np.sum(tmp/tmprms**2)/np.sum(1/tmprms**2)*1e6)
    stackrmses.append(np.sum(1/tmprms**2)**-0.5*1e6)
    print(stacktemps[-1],stackrmses[-1])
plt.figure(figsize=(4.2,2.4)).set_tight_layout(True)
plt.errorbar(np.arange(1,6),stacktemps,stackrmses)
plt.xlabel('pixel radius')
plt.ylabel(r'$T_b$ ($\mu$K)')
plt.grid(ls=':',alpha=0.63)
plt.savefig(params.savepath+'/signal_vs_stacksize_in_rings.png',dpi=288)
