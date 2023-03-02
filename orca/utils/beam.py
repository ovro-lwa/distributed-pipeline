from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import sys,os,inspect
from casacore import tables
from scipy.interpolate import griddata as gd

BEAM_FILE_PATH = os.path.abspath('/lustre/mmanders/LWA/modules/beam')

class BaseBeam(ABC):
    @abstractmethod
    def __init__(self, msfile: str):
        pass

    @abstractmethod
    def srcIQUV(self, az: float, el: float) -> \
            Tuple[float, float, float, float]:
        """
        Args:
            az: azimuth in degrees
            el: elevation in degrees
        """
        pass

class AnalyticBeam(BaseBeam):
    def __init__(self, msfile):
        pass

    def srcIQUV(self, az, el):
        return np.sin(el * 2 * np.pi/360) ** 1.6, 0, 0, 0

class WoodyBeam(BaseBeam):
    """
    For loading and returning LWA dipole beam values (derived from DW beam simulations) on the ASTM.
    Last edit: 08 August 2016
    """
    def __init__(self,msfile):
        self.CRFREQ = float(tables.table(msfile+'/SPECTRAL_WINDOW', ack=False).getcell('NAME', 0))/1.e6 # center frequency in MHz
        self.path   = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # absolute path to module
        # load 4096x4096 grid of azimuth,elevation values
        self.azelgrid = np.load(BEAM_FILE_PATH+'/azelgrid.npy')
        self.gridsize = self.azelgrid.shape[-1]
        # load 4096x4096 grid of IQUV values, for given msfile CRFREQ
        self.beamIQUVfile = BEAM_FILE_PATH+'/beamIQUV_'+str(self.CRFREQ)+'.npz'
        if os.path.exists(self.beamIQUVfile):
            self.beamIQUV = np.load(self.beamIQUVfile)
            self.Ibeam    = self.beamIQUV['I']
            self.Qbeam    = self.beamIQUV['Q']
            self.Ubeam    = self.beamIQUV['U']
            self.Vbeam    = self.beamIQUV['V']
        else:
            print >> sys.stderr, 'Beam .npz file does not exist at %s. Check your frequency. \
                                  May need to generate new beam file.' % str(self.CRFREQ)
            sys.exit()


    def srcIQUV(self,az,el):
        """Compute beam scaling factor
        Args:
            az: azimuth in degrees
            el: elevation in degrees

        Returns: [I,Q,U,V] flux factors, where for an unpolarized source [I,Q,U,V] = [1,0,0,0]

        """
        def knn_search(arr,grid):
            '''
            Find 'nearest neighbor' of array of points in multi-dimensional grid
            Source: glowingpython.blogspot.com/2012/04/k-nearest-neighbor-search.html
            '''
            gridsize = grid.shape[1]
            dists    = np.sqrt(((grid - arr[:,:gridsize])**2.).sum(axis=0))
            return np.argsort(dists)[0]
        # index where grid equals source az el values
        index = knn_search(np.array([ [az], [el] ]), self.azelgrid.reshape(2,self.gridsize*self.gridsize))
        Ifctr = self.Ibeam.reshape(self.gridsize*self.gridsize)[index]
        Qfctr = self.Qbeam.reshape(self.gridsize*self.gridsize)[index]
        Ufctr = self.Ubeam.reshape(self.gridsize*self.gridsize)[index]
        Vfctr = self.Vbeam.reshape(self.gridsize*self.gridsize)[index]
        return Ifctr,Qfctr,Ufctr,Vfctr

    def plotbeam(self):
        '''
        Show the IQUV beams at MS center frequency, and azimuth,elevation grids.
        '''
        import pylab as p
        # azel grids
        p.figure(figsize=(6,9))
        p.subplot(211)
        p.imshow(np.rot90(self.azelgrid[0,:,:]), cmap=plt.get_cmap('inferno'))
        p.title('azimuth')
        p.colorbar()
        p.subplot(212)
        p.imshow(np.rot90(self.azelgrid[1,:,:]), cmap=plt.get_cmap('inferno'))
        p.title('elevation')
        p.colorbar()
        # IQUV beams
        p.figure(figsize=(15,9))
        p.subplot(221)
        p.imshow(np.rot90(self.Ibeam),vmin=0.0,vmax=1.0,cmap=plt.get_cmap('inferno'))
        p.title('I')
        p.colorbar()
        p.subplot(222)
        p.imshow(np.rot90(self.Qbeam),vmin=-0.15,vmax=0.15,cmap=plt.get_cmap('inferno'))
        p.title('Q')
        p.colorbar()
        p.subplot(223)
        p.imshow(np.rot90(self.Ubeam),vmin=-0.15,vmax=0.15,cmap=plt.get_cmap('inferno'))
        p.title('U')
        p.colorbar()
        p.subplot(224)
        p.imshow(np.rot90(self.Vbeam),vmin=-0.010,vmax=0.010,cmap=plt.get_cmap('inferno'))
        p.title('V')
        p.colorbar()
        p.show()


class jones:
    """
    For loading and returning LWA dipole beam values (derived from DW beam simulations) on the ASTM.
    Last edit: 11 September 2020
    """
    def __init__(self,msfile):
        self.CRFREQ = float(tables.table(msfile+'/SPECTRAL_WINDOW', ack=False).getcell('NAME', 0))/1.e6 # center frequency in MHz
        self.path   = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # absolute path to module
        # load 4096x4096 grid of azimuth,elevation values
        #self.lmgrid  = np.load(BEAM_FILE_PATH+'/lmgrid.npy')
        #self.gridsize = self.lmgrid.shape[-1]
        ## load 4096x4096 grid of complex Co, Cx values
        #self.beamjonesfile = BEAM_FILE_PATH+'/beamJones.npz'
        self.beamjonesfile = BEAM_FILE_PATH+'/beamLudwig3rd.npz'
        if os.path.exists(self.beamjonesfile):
            #self.beamjones = np.load(self.beamjonesfile)
            #self.Co    = self.beamjones['Co']
            #self.Cx    = self.beamjones['Cx']
            self.beamjones = np.load(self.beamjonesfile)
            self.Co = self.beamjones['cofull']
            self.Cx = self.beamjones['cxfull']
            self.Corot90 = self.beamjones['cofull_rot90']
            self.Cxnrot90 = self.beamjones['cxfull_nrot90']
            self.l = self.beamjones['lfull']
            self.m = self.beamjones['mfull']
            self.freqs = self.beamjones['freqfull']
        else:
            print >> sys.stderr, 'Beam .npz file does not exist at %s. Check your frequency. \
                                  May need to generate new beam file.' % str(self.CRFREQ)
            sys.exit()


    def srcjones(self,l,m):
        """Compute beam scaling factor
        Args:
            (l,m) coordinates

        Returns: Jones matrix at coordinates (l,m)

        """
        #def knn_search(arr,grid):
        #    '''
        #    Find 'nearest neighbor' of array of points in multi-dimensional grid
        #    Source: glowingpython.blogspot.com/2012/04/k-nearest-neighbor-search.html
        #    '''
        #    gridsize = grid.shape[1]
        #    dists    = np.sqrt(((grid - arr[:,:gridsize])**2.).sum(axis=0))
        #    return np.argsort(dists)[0]
        ## index where grid equals source az el values
        #index = knn_search(np.array([ [l], [m] ]), self.lmgrid.reshape(2,self.gridsize*self.gridsize))
        #Jonesmat = np.array([ [self.Co.reshape(self.gridsize*self.gridsize)[index], self.Cx.reshape(self.gridsize*self.gridsize)[index]],
        #                      [-np.rot90(self.Cx).reshape(self.gridsize*self.gridsize)[index], np.rot90(self.Co).reshape(self.gridsize*self.gridsize)[index]] ])
        coval = gd( (self.l.ravel(), self.m.ravel(), self.freqs.ravel()), \
                selfs.Co.ravel(), (l, m, self.CRFREQ), method='linear')
        cxval = gd( (self.l.ravel(), self.m.ravel(), self.freqs.ravel()), \
                selfs.Cx.ravel(), (l, m, self.CRFREQ), method='linear')
        corot90val  = gd( (self.l.ravel(), self.m.ravel(), self.freqs.ravel()), \
                  self.Corot90.ravel(), (l, m, self.CRFREQ), method='linear')
        cxnrot90val = gd( (self.l.ravel(), self.m.ravel(), self.freqs.ravel()), \
                  self.Cxnrot90.ravel(), (l, m, self.CRFREQ), method='linear')
        Jonesmat = np.array([ [coval,       cxval     ], 
                          [cxnrot90val, corot90val] ])
        return Jonesmat

