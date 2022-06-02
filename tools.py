from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import os
import tables


class empty_table():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)


def weightmean(vals, rmss, axis=None):
    """
    average of vals, weighted by rmss, over the passed axes
    """
    meanval = np.nansum(vals/rmss**2, axis=axis) / np.nansum(1/rmss**2, axis=axis)
    meanrms = np.sqrt(1/np.nansum(1/rmss**2, axis=axis))
    return meanval, meanrms

def freq_to_z(nuem, nuobs):
    zval = (nuem - nuobs) / nuobs
    return zval

def nuem_to_nuobs(nuem, z):
    nuobs = nuem / (1 + z)
    return nuobs

def nuobs_to_nuem(nuobs, z):
    nuem = nuobs * (1 + z)
    return nuem

def coord_to_pix(coords, comap):
    """
    given a coordinate value in degrees, return the (x,y) coordinates
    according to the map stored in comap
    """
    xval = (coords[0] - comap.ra[0]) / (comap.ra[1] - comap.ra[0])
    yval = (coords[1] - comap.dec[0]) / (comap.dec[1] - comap.dec[0])
    return (xval, yval)

def load_map(file, reshape=True, noise_sim=False, inject_signal=None, inject_signal_scale=1.):
    """
    loads in a file in the COMAP format, storing everything as arrays in the map class.
    COMAP data are stored with coordinates as the CENTER of each pixel
    """
    # *** give maps their own special class at some point?

    comap = empty_table() # creates empty class to put map info into

    with tables.open_file(file, 'r') as mapfile:
        maptemparr = np.array(mapfile.root.map_coadd)
        rmstemparr = np.array(mapfile.root.rms_coadd)
        comap.freq = np.array(mapfile.root.freq)
        comap.ra = np.array(mapfile.root.x)
        comap.dec = np.array(mapfile.root.y)

        patch_cent = np.array(mapfile.root.patch_center)
        comap.fieldcent = SkyCoord(patch_cent[0]*u.deg, patch_cent[1]*u.deg)

        # mark pixels with zero rms and mask them in the rms/map arrays (how the pipeline stores infs)
    comap.badpix = np.where(rmstemparr < 1e-10)
    maptemparr[comap.badpix] = np.nan
    rmstemparr[comap.badpix] = np.nan

    comap.map = maptemparr
    comap.rms = rmstemparr

    if reshape:
        # also reshape into 3 dimensions instead of separating sidebands
        comap.freq = np.reshape(comap.freq, 4*64)
        comap.map = np.reshape(comap.map, (4*64, len(comap.dec), len(comap.ra)))
        comap.rms = np.reshape(comap.rms, (4*64, len(comap.dec), len(comap.ra)))
    
    if noise_sim:
        comap.map = np.random.normal(size=comap.rms.shape)*comap.rms
    if inject_signal is not None:
        tmpcube = np.swapaxes(np.load(inject_signal)['map_cube'][...,::-1],0,-1)
        tmpcube-= np.mean(tmpcube)
        padcount_ra_lo = (len(comap.ra) - tmpcube.shape[-1])//2
        padcount_ra_hi = len(comap.ra) - tmpcube.shape[-1] - padcount_ra_lo
        padcount_declo = (len(comap.dec) - tmpcube.shape[1])//2
        padcount_dechi = len(comap.dec) - tmpcube.shape[1] - padcount_declo
        comap.map+= np.pad(tmpcube,((0,0),(padcount_declo,padcount_dechi),(padcount_ra_lo,padcount_ra_hi)))*inject_signal_scale

    # 1-pixel radii for each of the axes
    comap.fstep = (comap.freq[1] - comap.freq[0])/2
    comap.xstep = (comap.ra[1] - comap.ra[0])/2
    comap.ystep = (comap.dec[1] - comap.dec[0])/2

    # housekeeping for the arrays - give each axis an index array as well
    comap.x = np.arange(len(comap.ra))
    comap.y = np.arange(len(comap.dec))

    # rearrange so that the stored coordinate coordinate arrays correspond to the
    # bottom right (etc.) of the voxel (currently they're the center)
    comap.freq = comap.freq - comap.fstep / 2
    comap.ra = comap.ra - comap.xstep / 2
    comap.dec = comap.dec - comap.ystep / 2


    # limits on each axis for easy testing
    comap.flims = (np.min(comap.freq), np.max(comap.freq))
    comap.xlims = (np.min(comap.ra), np.max(comap.ra))
    comap.ylims = (np.min(comap.dec), np.max(comap.dec))

    # *** any other per-field info we need

    return comap

def setup(mapfiles, cataloguefile, params):
    maplist = []
    for mapfile_idx,mapfile in enumerate(mapfiles):
        if params.inject_signal:
            mapinst = load_map(mapfile,noise_sim=params.noise_sim,inject_signal=params.inject_signal_cubes[mapfile_idx],inject_signal_scale=params.inject_signal_scale)
        else:
            mapinst = load_map(mapfile,noise_sim=params.noise_sim)

        # calculate the appropriate redshift limits from the freq axis
        zlims = freq_to_z(params.centfreq, np.array(mapinst.flims))
        mapinst.zlims = np.sort(zlims)

        maplist.append(mapinst)

    catdict = {}

    if params.inject_signal:
        catdict['z'] = []
        catdict['ra'] = []
        catdict['dec'] = []
        for map_idx,mapinst in enumerate(maplist):
            inject_cata = np.load(params.inject_signal_catas[map_idx])
            M_cut_mask = (inject_cata['M']>params.inject_signal_M_min)
            catdict['z']+= [inject_cata['z'][M_cut_mask]]
            catdict['ra']+= [inject_cata['ra'][M_cut_mask]+mapinst.fieldcent.ra.value]
            catdict['dec']+= [inject_cata['dec'][M_cut_mask]+mapinst.fieldcent.dec.value]
        catdict['z']   = np.hstack(catdict['z']  )
        catdict['ra']  = np.hstack(catdict['ra'] )
        catdict['dec'] = np.hstack(catdict['dec'])
    else:
        with np.load(cataloguefile) as catfile:
            catdict['z'] = catfile['z']
            catdict['ra'] = catfile['ra']
            catdict['dec'] = catfile['dec']

    catlist = []
    for i in range(len(mapfiles)):
        catinst = field_cull_galaxy_cat(catdict, maplist[i])
        catlist.append(catinst)

    return maplist, catlist

def field_cull_galaxy_cat(galdict, comap, maxsep=3*u.deg):
    """
    takes the full version of the catalogue to be stacked on and cuts to all objects within some
    radius of the given field center
    """
    # *** get rid of skycoord dependence
    # allow you to carry around other arbitrary parameters? ****
    fieldcent = comap.fieldcent
    zlims = np.array(comap.zlims)

    # pull only objects in the field
    fieldcoords = SkyCoord(galdict['ra']*u.deg, galdict['dec']*u.deg)
    fieldsep = fieldcoords.separation(fieldcent)
    fieldidx = np.where(fieldsep < maxsep)[0]

    fieldz_cut = galdict['z'][fieldidx]
    fieldidx = fieldidx[np.where(np.logical_and(fieldz_cut > zlims[0], fieldz_cut < zlims[1]))[0]]

    # save to cat object
    galcat = empty_table()
    galcat.coords = fieldcoords[fieldidx]
    galcat.z = galdict['z'][fieldidx]
    galcat.idx = fieldidx

    # number objects in cat
    galcat.nobj = len(fieldidx)

    return galcat

def plot_mom0(comap, ext=0.95, lognorm=True):

    """
    unsure about the transpose thing
    """

    fig,ax = plt.subplots(1)

    moment0 = weightmean(comap.map, comap.rms, axis=(0))[0] * 1e6
    vext = (np.nanmin(moment0)*ext, np.nanmax(moment0)*ext)
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0.T,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='plasma')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, moment0.T, cmap='plasma')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c)
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return 0

def plot_chan(comap, channel, ext=0.95, lognorm=True):
    fig,ax = plt.subplots(1)
    plotmap = comap.map[channel,:,:].T * 1e6
    vext = (np.nanmin(plotmap)*ext, np.nanmax(plotmap)*ext)
    if lognorm:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap,
                          norm=SymLogNorm(linthresh=1, linscale=0.5,
                                          vmin=vext[0], vmax=vext[1]),
                          cmap='plasma')
    else:
        c = ax.pcolormesh(comap.ra, comap.dec, plotmap, cmap='plasma')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')

    cbar = fig.colorbar(c, ax=ax, extend='both')
    cbar.ax.set_ylabel(r'$T_b \ (\mu K)$')

    return 0
