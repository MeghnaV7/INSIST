import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb

import cv2
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, Distance, ICRS
from astropy.modeling import models, fitting
from astropy.visualization import LogStretch
from astropy.io import fits
from astropy.io.fits import CompImageHDU
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.modeling.models import custom_model

from matplotlib import cm
from matplotlib.ticker import LogLocator
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib import colors as col
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.cosmology import WMAP9 as cosmo
from astropy.coordinates import angular_separation

from photutils import aperture as aper
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from photutils.isophote import Ellipse

import photutils as phot
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources

from photutils.detection import IRAFStarFinder, DAOStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup, FittableImageModel

from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.psf import DAOPhotPSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter

from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs

import galsim

from scipy.stats import norm
from scipy.constants import c
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import copy

import os
from time import perf_counter
import json
import requests

sb.set_style('white')
matplotlib.rcParams['font.size']= 12
matplotlib.rcParams['figure.figsize']=(10,10)

matplotlib.rcParams['xtick.labelsize'] = 'x-large'
matplotlib.rcParams['ytick.labelsize'] = 'x-large'

import torch
import torch.nn.functional as F
from torch.autograd import Variable

data_path= '/home/hp/Desktop/Ahmedabad_Project/INSIST/src/pista/data'

def bandpass(wav, flux, inputs, plot = True, fig = None, ax = None):
  """
  Function to convolve response functions

  Parameters
  ----------
  wav : numpy.ndarray
        wavelenth in angstrom
  flux: numpy.ndarray
        flux normalized to [0,1]
  plot: bool,
        If true shows plots with input and convolved response functions

  fig : matplotlib.pyplot.figure
        User defined figure
  ax  : matplotlib.pyplot.axes
        User defined axes

  Returns
  -------

  fig, ax, data, params

  data : tuple,
        (wavelenth array, flux_array, convolved flux array)

  params: tuple,
          (effective wavelength, integrated flux, Effective Width)

  """
  lambda_   =  wav
  flux_AB   = flux

  if plot:
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1, figsize = (12,8))
    ax.plot(lambda_ ,flux_AB/flux_AB.max(),label = r'$F(\lambda)$', alpha = 0.7 )

  R_eff = 1

  for i in inputs:
    file_name = i.split(',')[0]
    n         = float(i.split(',')[1])
    f_max     = float(i.split(',')[2])

    filt_dat  = np.loadtxt(file_name)
    wav  = filt_dat[:,0]
    flux = filt_dat[:,1]

    if np.amax(flux)>1:
      flux/=f_max

    indices  = np.where( (wav>lambda_ [0]) & (wav<lambda_[-1]))
    wav_new  = wav[indices]
    flux_new = flux[indices]

    wav_new  = np.concatenate([[lambda_ [0]], [wav_new[0]- 1], wav_new,
                               [wav_new[-1]+ 1], [lambda_[-1]]])

    flux_new = np.concatenate([[0], [0], flux_new, [0], [0]])

    flux_out = np.interp(lambda_,wav_new,flux_new)

    R_eff      *= flux_out**n

    if plot:
      ax.plot(lambda_ ,flux_out/flux_out.max(),
              label=f"{file_name.split('/')[-1][:-4]}x{n}", alpha = 0.7)

  # Wavelength space
  conv_flux    = R_eff*flux_AB
  int_flux     = np.trapz(lambda_*conv_flux,lambda_)/np.trapz(lambda_*R_eff,
                                                              lambda_)
  W_eff        = np.trapz(R_eff,lambda_)/R_eff.max()
  lambda_phot  = np.trapz(lambda_**2*conv_flux,
                          lambda_)/np.trapz(lambda_*conv_flux,lambda_)

  c1 = lambda_ >= lambda_phot-W_eff/2
  c2 = lambda_ <= lambda_phot+W_eff/2

  R_sq         = np.where((c1 & c2),1,0)
  flux_ratio   = np.trapz(R_eff,lambda_)/np.trapz(R_sq,lambda_)

  if plot:
      ax.plot(lambda_ ,R_sq,
              label="Square Filter", alpha = 0.7)

  # Frequency space
  R_eff_Jy      = R_eff*lambda_**2*3.34e4
  flux_AB       = flux_AB*lambda_**2*3.34e4
  nu            = 3e18/lambda_

  conv_flux_Jy  = R_eff_Jy*flux_AB
  int_flux_Jy   = np.trapz(nu*conv_flux_Jy,nu)/np.trapz(nu*R_eff_Jy, nu)

  # Comparing to a square filter with same width

  data       =  lambda_, conv_flux, R_eff
  params     =  lambda_phot, int_flux, int_flux_Jy, W_eff, flux_ratio

  if plot:
    ax.plot(lambda_,conv_flux/conv_flux.max(),label = 'Convloved Flux',
            linewidth = 5)
    y = np.linspace(0,1)
    x = y*0 + lambda_phot
    label = r'$\lambda_{phot} = $' + f'{round(lambda_phot,3)}' + r' $\AA$'
    ax.plot(x,y,'--', color = 'black',label = label )

    ax.set_xlabel(r'$\AA$')
    ax.set_ylabel(r'Normalized Flux')
    fig.suptitle('Bandpass', fontsize = 20, y = 0.95)
    ax.legend()

  return fig, ax, data, params
  
def distance_transform(ras, decs, cen_ra, cen_dec, d1, d2):

  # Define the coordinates of the galaxy in the original catalog
  galaxy_ra = cen_ra*u.deg
  galaxy_dec = cen_dec*u.deg

  # Define the coordinates of the stars in the original catalog
  star_ra = ras*u.deg
  star_dec = decs*u.deg

  # Create a SkyCoord object for the galaxy in the original catalog
  galaxy_coord = SkyCoord(ra=galaxy_ra, dec=galaxy_dec, distance=Distance(d1), frame=ICRS())

  # Create a SkyCoord object for the stars in the original catalog
  star_coords = SkyCoord(ra=star_ra, dec=star_dec, frame=ICRS())

  # Calculate the factor by which to scale the coordinates
  scale_factor = d1/d2

  # Calculate the separation between the stars and the galaxy
  separation = star_coords.separation(galaxy_coord)

  # Calculate the new position angle using the original coordinates
  position_angle = star_coords.position_angle(galaxy_coord)

  # Scale the separation by the scale factor
  scaled_separation = np.arctan(np.tan(separation)*scale_factor)

  # Calculate the new star coordinates using the scaled separation and position angle
  new_star_coords = galaxy_coord.directional_offset_by(position_angle, scaled_separation)

  return new_star_coords.ra.value, new_star_coords.dec.value
  

class Analyzer(object):
    def __init__(self):
        """
        A class to visualize and analyze the simulated image

        Parameters
        ----------
        Imager.init()

        Returns
        -------
        None.

        """
    def __call__(self, df=None, wcs=None, data=None,
                 photometry=None, detect_sources=False, fwhm=3, sigma=13,
                 ZP=None):
        """
        Performs sim simulation and sim Photometry

        Imager.call()

        do_photometry : Bool, Default : True
                        Do Aperture Photometry
        """
        self.photometry_type = photometry
        if photometry == 'Aper':
            self.aper_photometry(data, wcs, df, fwhm, sigma, detect_sources,
                                 ZP)
        elif photometry == 'PSF':
            self.psf_photometry(data, wcs, df, fwhm, sigma, ZP)

    def aper_photometry(self, data, wcs, df, fwhm, sigma, detect, ZP):

        # if detect flag is set to True, detect sources in the image
        if detect:

            # calculate the mean, median and standard deviation of the data
            mean, median, std = sigma_clipped_stats(data, sigma=sigma)
            # create DAOStarFinder object to detect sources
            daofind = DAOStarFinder(fwhm=fwhm, threshold=median + sigma*std)
            # detect sources in the image
            sources = daofind(data)
            # get the source positions
            positions = np.transpose((sources['xcentroid'],
                                      sources['ycentroid']))

        else:

            # create SkyCoord object from ra and dec values in the dataframe
            c = SkyCoord(df['ra'], df['dec'], unit=u.deg)
            # convert the sky coordinates to pixel coordinates
            pix = wcs.world_to_array_index(c)
            positions = [(i, j) for i, j in zip(pix[1], pix[0])]

        # create circular aperture object
        self.aps = aper.CircularAperture(positions, r=2*fwhm)
        # count number of pixels within the aperture
        ap_pix = np.count_nonzero(self.aps.to_mask()[0])
        # create circular annulus object
        self.bags = aper.CircularAnnulus(positions, r_in=3*fwhm,
                                         r_out=5*fwhm)
        # count number of pixels within the annulus
        bag_pix = np.count_nonzero(self.bags.to_mask()[0])

        # perform aperture photometry on the data
        phot_table = aper.aperture_photometry(data, [self.aps, self.bags])

        # calculate sky flux
        phot_table['sky_flux'] = phot_table['aperture_sum_1']*(ap_pix/bag_pix)
        # calculate source flux
        phot_table['flux'] = phot_table['aperture_sum_0'].value - \
            phot_table['sky_flux'].value
        # calculate error on the source flux
        phot_table['flux_err'] = np.sqrt(phot_table['flux'].value +
                                         phot_table['sky_flux'].value)

        # calculate signal to noise ratio
        phot_table['SNR'] = phot_table['flux']/phot_table['flux_err']

        if not detect:

            phot_table['ra'] = df['ra'].values
            phot_table['dec'] = df['dec'].values
            phot_table['mag_in'] = df['mag'].values

        else:
            coords = np.array(wcs.pixel_to_world_values(positions))

            phot_table['ra'] = coords[:, 0]
            phot_table['dec'] = coords[:, 1]

            matched = Xmatch(phot_table, df, 3*fwhm)
            mag_in = []
            sep = []
            for i, j, k in matched:
                if j is not None:
                    mag_in.append(df['mag'].values[j])
                    sep.append(k)
                else:
                    mag_in.append(np.nan)
                    sep.append(np.nan)

            phot_table['mag_in'] = mag_in
            phot_table['sep'] = sep

        phot_table['mag_out'] = -2.5*np.log10(phot_table['flux']) + ZP
        phot_table['mag_err'] = 1.082/phot_table['SNR']

        coords = np.array(wcs.pixel_to_world_values(positions))
        phot_table['ra'] = coords[:, 0]
        phot_table['dec'] = coords[:, 1]
        self.phot_table = phot_table

    def psf_photometry(self, data, wcs, df, fwhm, sigma, ZP):

        mean, median, std = sigma_clipped_stats(data, sigma=3)

        psf_model = FittableImageModel(self.psf)
        self.psf_model = psf_model

        photometry = DAOPhotPSFPhotometry(crit_separation=2,
                                          threshold=mean + sigma*std,
                                          fwhm=fwhm,
                                          aperture_radius=3,
                                          psf_model=psf_model,
                                          fitter=LevMarLSQFitter(),
                                          fitshape=(11, 11),
                                          niters=3)

        phot_table = photometry(image=data)
        positions = np.array([phot_table['x_fit'], phot_table['y_fit']]).T
        coords = np.array(wcs.pixel_to_world_values(positions))

        phot_table['ra'] = coords[:, 0]
        phot_table['dec'] = coords[:, 1]
        phot_table['SNR'] = phot_table['flux_fit']/phot_table['flux_unc']


        phot_table['mag_out'] = -2.5*np.log10(phot_table['flux_fit'])
        phot_table['mag_out'] += ZP
        phot_table['mag_err'] = 1.082/phot_table['SNR']

        matched = Xmatch(phot_table, df, 3*fwhm)
        mag_in = []
        sep = []
        for i, j, k in matched:
            if j is not None:
                mag_in.append(df['mag'].values[j])
                sep.append(k)
            else:
                mag_in.append(np.nan)
                sep.append(np.nan)

        phot_table['mag_in'] = mag_in
        phot_table['sep'] = sep


        self.phot_table = phot_table

    def show_field(self, figsize=(12, 10), marker='.', cmap='jet'):
        """
        Function for creating a scatter plot of sources within the FoV

        Parameters
        ----------
        figsize : tuple,
                Figure size

        Returns
        -------
        fig, ax
        """
        figsize = figsize[0]*self.n_x/self.n_y, figsize[1]

        # Cropping Dataframe based on FoV
        left = (self.n_x_sim - self.n_x)//2
        right = left + self.n_x

        df = self.sim_df

        x_min_cut = (df['x'] > left)
        x_max_cut = (df['x'] < right)

        df = df[x_min_cut & x_max_cut]

        bottom = (self.n_y_sim - self.n_y)//2
        top = bottom + self.n_y

        y_min_cut = (df['y'] > bottom)
        y_max_cut = (df['y'] < top)

        df = df[y_min_cut & y_max_cut]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if hasattr(self, 'spec_bins'):
          x = df['x']
          y = df['y']
          c = None
          cmap = None
          color = None

          if 'mag' in df.key():
            c = df['mag']
            cmap = 'jet'
            color = None

          img = ax.scatter(x, y, c=c, color = color, cmap=cmap, marker=marker)
          cb = plt.colorbar(img)
          cb.set_label('mag (ABmag)')
          ax.set_title(f"""Requested Center : {self.name} | {len(df)} sources
          Fov(RA) : {fov_x} (deg) | Fov(Dec) : {fov_y} (deg)""")
          ax.set_xlabel('x (pix)')
          ax.set_ylabel('y (pix)')

          if hasattr(self, 'L'):
            B = self.B
            L = self.L
            l = len(self.spec_bins)
            PA = self.PA

            delta = np.arctan(B/L)
            d = np.sqrt(L**2 + B**2)/2
            om = PA - delta

            x_corr = self.n_x_sim//2 - d*np.cos(om) - l/2
            y_corr = self.n_y_sim//2 - d*np.sin(om) - B*np.cos(PA)

            start = B*np.sin(PA) + l/2 + x_corr
            end = start + L*np.cos(PA)
            x = np.linspace(start,end,100)
            y = np.tan(PA)*(x - B*np.sin(PA) - l/2 - x_corr) + y_corr

            ax.plot(x,y,color = 'red')

            start = l/2 + x_corr
            end = start + L*np.cos(PA)
            x = np.linspace(start, end, 100)
            y = np.tan(PA)*(x - l/2  - x_corr) + B*np.cos(PA) + y_corr

            ax.plot(x,y,color = 'red')

            start = y_corr
            end = start + B*np.cos(PA)
            y = np.linspace(start, end, 100)
            x = -np.tan(PA)*(y - B*np.cos(PA) - y_corr) + l/2 + x_corr

            ax.plot(x,y,color = 'red')

            start = y_corr +  L*np.sin(PA)
            end = start + B*np.cos(PA)
            y = np.linspace(start, end, 100)
            x = -np.tan(PA)*(y - B*np.cos(PA) - L*np.sin(PA) - y_corr) + L*np.cos(PA) + l/2 + x_corr

            ax.plot(x,y,color = 'red')
            ax.set_xlim(0,self.n_x_sim)
            ax.set_ylim(0,self.n_y_sim)

        else:

          x = df['ra']
          y = df['dec']
          c = df['mag']


          img = ax.scatter(x, y, c=c, marker=marker, cmap=cmap)
          cb = plt.colorbar(img)
          cb.set_label('mag (ABmag)')

          fov_x = (self.n_x*self.pixel_scale)/3600
          fov_y = (self.n_y*self.pixel_scale)/3600

          fov_x = np.round(fov_x, 4)
          fov_y = np.round(fov_y, 4)
          ax.set_title(f"""Requested Center : {self.name} | {len(df)} sources
          Fov(RA) : {fov_x} (deg) | Fov(Dec) : {fov_y} (deg)""")
          ax.invert_xaxis()
          ax.set_xlabel('RA (Degrees)')
          ax.set_ylabel('Dec (Degrees)')
          ax.invert_xaxis()
          ax.set_xlim(self.ra+fov_x/2, self.ra-fov_x/2)
          ax.set_ylim(self.dec-fov_y/2,self.dec+fov_y/2)


        return fig, ax

    def show_image(self, source='Digital', fig=None, ax=None, cmap='jet',
                   figsize=(15, 10), download=False, show_wcs=True,
                   overlay_apertures=False):
        """
        Function for plotting the simulated field image

        Source: str,
                Choose from
                            'Digital' : Final digial image
                            'Charge'  : electrons, Light(Source + sky) +
                                         Dark Current + Noises
                            'Source'  : Source + Sky + Noises
                            'Sky'     : Sky + shot_noise
                            'DC'      : Dark Current + DNFP
                            'QE'      : Quantum efficiency fluctuation across
                                        detector
                            'Bias'    : Charge offset
                            'PRNU'    : Photon Response Non-Uniformity
                            'DNFP'    : Dark Noise Fixed Pattern
                            'QN'      : Quantization Noise


        fig : matplotlib.pyplot.figure
            User defined figure
        ax  : matplotlib.pyplot.axes
            User defined axes
        cmap : str,
            matplotlib.pyplot colormap
        figsize : tuple
        download : bool
        show_wcs : bool
                If true adds WCS projection to the image
        Returns
        -------
        Image

        fig, ax
        """
        if hasattr(self,'digital'):
            if fig is None or ax is None:
                fig = plt.figure(figsize=figsize)
                if show_wcs:
                    ax = fig.add_subplot(projection=self.wcs)
                else:
                    ax = fig.add_subplot()

            norm = None
            if source == 'Digital':
                data = self.digital
                norm = col.LogNorm()
            elif source == 'Charge':
                data = self.charge
                norm = col.LogNorm()
            elif source == 'Source':
                data = self.light_array
                norm = col.LogNorm()
            elif source == 'Sky':
                data = self.sky_photons
            elif source == 'DC':
                data = self.DC_array
            elif source == 'Bias':
                data = self.bias_array
            elif source == 'PRNU':
                data = self.PRNU_array
            elif source == 'DNFP':
                norm = col.LogNorm()
                data = self.DNFP_array
            elif source == 'QN':
                data = self.QN_array
            else:
                print("Invalid Input")
                return None

            if data.min() < 0:
                print('Negative values in image. Increase Bias')
                data += data.min()

            img = ax.imshow(data, cmap=cmap, norm=norm)
            ax.grid(False)
            cb = plt.colorbar(img, ax=ax)
            cb.set_label('DN')
            ax.set_title(f'{source} \nRequested center : {self.name}')
            ax.grid(False)

            if overlay_apertures and self.photometry_type == "Aper":
                for aperture in self.aps:
                    if aperture is not None:
                        aperture.plot(ax=ax, color='red', lw=1.5)
                for aperture in self.bags:
                    if aperture is not None:
                        aperture.plot(ax=ax, color='yellow', lw=1.5)

            if hasattr(self, 'L'):
              B = self.B
              L = self.L
              l = len(self.spec_bins)
              PA = self.PA

              delta = np.arctan(B/L)
              d = np.sqrt(L**2 + B**2)/2
              om = PA - delta

              x_corr = self.n_x//2 - d*np.cos(om) - l/2
              y_corr = self.n_y//2 - d*np.sin(om) - B*np.cos(PA)

              start = B*np.sin(PA) + l/2 + x_corr
              end = start + L*np.cos(PA)
              x = np.linspace(start,end,100)
              y = np.tan(PA)*(x - B*np.sin(PA) - l/2 - x_corr) + y_corr

              ax.plot(x,y,color = 'red')

              start = l/2 + x_corr
              end = start + L*np.cos(PA)
              x = np.linspace(start, end, 100)
              y = np.tan(PA)*(x - l/2  - x_corr) + B*np.cos(PA) + y_corr

              ax.plot(x,y,color = 'red')

              start = y_corr
              end = start + B*np.cos(PA)
              y = np.linspace(start, end, 100)
              x = -np.tan(PA)*(y - B*np.cos(PA) - y_corr) + l/2 + x_corr

              ax.plot(x,y,color = 'red')

              start = y_corr +  L*np.sin(PA)
              end = start + B*np.cos(PA)
              y = np.linspace(start, end, 100)
              x = -np.tan(PA)*(y - B*np.cos(PA) - L*np.sin(PA) - y_corr) + L*np.cos(PA) + l/2 + x_corr

              ax.plot(x,y,color = 'red')

            if download:
                fig.savefig(f"{source}_ngc3368.png", format='png')
            return fig, ax

        else:
            print("Run Simulation")

    def show_hist(self, source='Digital', bins=None,
                  fig=None, ax=None, figsize=(15, 8)):
        """
        Function for plotting histogram of various stages of simulation

        Parameters
        ----------

        Source: str,
                Choose from
                        'Digital' : Final digial image
                        'Charge'  : electrons, Light(Source + sky)
                                    + Dark Current + Noises
                        'Source'  : Source + Sky + Noises
                        'Sky'     : Sky + shot_noise
                        'DC'      : Dark Current + DNFP
                        'QE'      : Quantum efficiency fluctuation across
                                    detector
                        'Bias'    : Charge offset
                        'PRNU'    : Photon Response Non-Uniformity
                        'DNFP'    : Dark Noise Fixed Pattern
                        'QN'      : Quantization Noise

        bins : numpy.array,
            bins for making histogram
        fig : matplotlib.pyplot.figure
            User defined figure
        ax  : matplotlib.pyplot.axes
            User defined axes
        figsize : tuple
        """

        if hasattr(self,'digital'):
            if fig is None or ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)

            if source == 'Digital':
                data = self.digital.ravel()
            elif source == 'Charge':
                data = self.charge.ravel()
            elif source == 'Source':
                data = self.light_array
            elif source == 'Sky':
                data = self.sky_photons.ravel()
            elif source == 'DC':
                data = self.DC_array.ravel()
            elif source == 'Bias':
                data = self.bias_array.ravel()
            elif source == 'PRNU':
                data = self.PRNU_array.ravel()
            elif source == 'DNFP':
                data = self.DNFP_array.ravel()
            elif source == 'QN':
                data = self.QN_array.ravel()

            if bins is None:
                bins = np.linspace(data.min(), data.max(), 20)
            ax.hist(data, bins=bins)
            ax.set_title(f'{source} histogram')
            ax.set_ylabel('Count')
            ax.set_yscale('log')
            return fig, ax
        else:
            print("Run Simulation")

    def getImage(self, source='Digital'):
        """
        Function of retrieving image array at different stages of simulation.

        Parameters
        ----------

        Source: str,
            Choose from
                        'Digital' : Final digial image
                        'Charge'  : electrons, Light(Source + sky)
                                    + Dark Current + Noises
                        'Source'  : Source + Sky + Noises
                        'Sky'     : Sky + shot_noise
                        'DC'      : Dark Current + DNFP
                        'QE'      : Quantum efficiency fluctuation across
                                    detector
                        'Bias'    : Charge offset
                        'PRNU'    : Photon Response Non-Uniformity
                        'DNFP'    : Dark Noise Fixed Pattern
                        'QN'      : Quantization Noise
        """

        if hasattr(self, 'digital'):
            if source == 'Digital':
                data = self.digital
            elif source == 'Charge':
                data = self.charge
            elif source == 'Sky':
                data = self.sky_photoelec
            elif source == 'DC':
                data = self.DC_array
            elif source == 'QE':
                data = self.qe_array
            elif source == 'Bias':
                data = (self.bias_array + self.DC_array)
            elif source == 'PRNU':
                data = self.PRNU_array
            elif source == 'DNFP':
                data = self.DNFP_array
            elif source == 'QN':
                data = self.QN_array
            else:
                data = 0
            return data
        else:
            print("Run Simulation")

    def writeto(self, name, source='Digital', user_source=None,
                with_dark_flat = False):

        """
        Function for downloading a fits file of simulated field image

        Parameters
        ----------
        name : str
            filename, Example : simulation.fits

        Source: str,
            Choose from
                        'Digital' : Final digial image
                        'Charge'  : electrons, Light(Source + sky)
                                    + Dark Current + Noises
                        'Source'  : Source + Sky + Noises
                        'Sky'     : Sky + shot_noise
                        'DC'      : Dark Current + DNFP
                        'Bias'    : Charge offset
                        'PRNU'    : Photon Response Non-Uniformity
                        'DNFP'    : Dark Noise Fixed Pattern
                        'QN'      : Quantization Noise

        user_source : numpy.ndarray
                    2D numpy array user wants to save as FITS
        with_dark_flat : bool
                          True : Output fits will provide dark and flat frames
        """
        if hasattr(self, 'digital'):
            if user_source is not None and isinstance(user_source) == np.ndarray:
                data = user_source
            elif source == 'Digital':
                data = self.digital
            elif source == 'Charge':
                data = self.charge
            elif source == 'Source':
                data = self.light_array
            elif source == 'Sky':
                data = self.sky_photoelec
            elif source == 'DC':
                data = self.DC_array
            elif source == 'Bias':
                data = self.bias_array
            elif source == 'PRNU':
                data = self.PRNU_array
            elif source == 'DNFP':
                data = self.DNFP_array
            elif source == 'QN':
                data = self.QN_array

            else:
                print(f"{source} is not a valid source")

            hdu = fits.PrimaryHDU(data, header=self.header)
            hdu.wcs = self.wcs
            hdus = [hdu]

            if with_dark_flat:
              dark = self.dark_frame/self.exp_time
              header = self.header
              header['Frame'] = 'Dark'

              hdu = fits.ImageHDU(dark, header=header)
              hdus.append(hdu)

              flat = self.flat_frame
              header['Frame'] = 'Flat'

              hdu = fits.ImageHDU(flat, header=header)
              hdus.append(hdu)

            hdul = fits.HDUList(hdus)
            hdul.writeto(f'{name}', overwrite=True, checksum=True)
        else:
            print("Run Simulation")

    def writecomp(self, name):

      hdu = CompImageHDU(data=self.digital, header=self.header)
      hdus = [fits.PrimaryHDU(), hdu]
      hdul = fits.HDUList(hdus)
      hdul.writeto(name, overwrite = True, checksum = True)
      

class SpecAnalyzer(object):

    def __init__(self):
        """
        A class to visualize and analyze the simulated image

        Parameters
        ----------
        Imager.init()

        Returns
        -------
        None.

        """
    def extract(self, data, wcs, df, width=2):

        """
        Extracts Spectra for Slitless Spectroscopy

        """
        c = SkyCoord(df['ra'], df['dec'], unit=u.deg)
        pix = wcs.world_to_array_index(c)
        positions = [(i, j) for i, j in zip(pix[1], pix[0])]

        bin_mid = len(self.spec_bins)//2

        fluxes = []
        for i, j in positions:

            start = i-bin_mid if i-bin_mid >= 0 else 0
            end = i+bin_mid if i+bin_mid <= self.n_x else self.n_x
            flux = data[j-width-1:j + width, start:end+1]

            fluxes.append(flux.sum(axis=0))

        self.phot_table = df[['ra', 'dec', 'z1', 'z2']]
        self.phot_table['flux'] = fluxes

    def generate_cal_flux(self, source_indices=[0, 1, 2]):

        z = zip(self.img_df['wav'][source_indices],
                self.img_df['flux'][source_indices],
                self.phot_table['flux'][source_indices],
                )
        cal_flux = 0

        for in_wav, in_flux, out_flux in z:
            binned_input = self.bin_xy(in_wav, in_flux)
            cal_flux += binned_input/out_flux

        cal_flux /= len(source_indices)

        self.cal_flux = cal_flux
        

#PSF

class PSF_Map(object):
  def __init__(self,n, m, x_sigmas, y_sigmas, thetas):
    if n<= x_sigmas.shape[0] and m<= x_sigmas.shape[1]:
      self.x_bins  = np.linspace(0,len(x_sigmas) -1,n)
      self.y_bins  = np.linspace(0,len(y_sigmas) -1,m)

    else:
      raise Exception(f"n or m value is out of bounds")

    self.x_cens = (self.x_bins[:-1] + self.x_bins[1:])//2
    self.y_cens = (self.y_bins[:-1] + self.y_bins[1:])//2

    self.x_sigmas = x_sigmas
    self.y_sigmas = y_sigmas

    self.thetas = thetas

  def get_params(self,x,y):
    if x == 0:
      x_point = self.x_cens[0]

    elif x == self.x_bins[-1]:
      x_point = self.x_cens[-1]

    else:
      x_point = self.x_cens[np.where(self.x_bins<x)[0][-1]]

    if y == 0:
      y_point = self.y_cens[0]

    elif y == self.y_bins[-1]:
      y_point = self.y_cens[-1]

    else:
      y_point = self.y_cens[np.where(self.y_bins<y)[0][-1]]

    x_point, y_point = int(x_point), int(y_point)

    x_sigma = self.x_sigmas[x_point,y_point]
    y_sigma = self.y_sigmas[x_point,y_point]
    theta   = self.thetas[x_point,y_point]

    return x_sigma, y_sigma, theta
    
#IMAGING

class Imager(Analyzer):
    """Imager class uses dataframe containing position and magntidue
    information to simulate image based on user defined telescope
    and detector characteristics
    """
    def __init__(self, df, coords=None, tel_params=None, n_x=1000,
                 n_y=1000, exp_time=100, plot=False, user_profiles=None):
        """
        Parameters
        ----------
        df     : pd.DataFrame,
                 Pandas dataframe with source catalog
        coords : (float, float),
                 (RA, Dec) in degrees
        tel_params : dict,
                     {'aperture'       : float,  cm
                      'pixel_scale'    : float,  arcsecs/pixels
                      'sim_file'       : fits,npy
                      'response_funcs' : list, [filename.dat, n] where n is
                                              number of times to multiply
                                              filter
                                              profile
                      'coeffs'         : float, filter coefficients if not
                                              response_funcs
                      }
        n_x      : int,
                   number of pixels along RA direction
        n_y      : int,
                   number of pixels along Dec direction
        exp_time : float
                   Exposure time in seconds
        """
        super().__init__()

        # Flags
        self.shot_noise = True
        self.QE = True
        self.sky = True
        self.PRNU = True
        self.DC = True
        self.DCNU = True
        self.DNFP = True
        self.QN = True

        # TBD
        self.cosmic_rays = False

        # Telescope and Detector Parameters
        psf_file = f'{data_path}/PSF/INSIST/off_axis_hcipy.npy'
        sky_resp = f'{data_path}/Sky_mag.dat'

        self.tel_params = {'aperture': 100,  # cm
                           'pixel_scale': 0.1,
                           'psf_file': psf_file,
                           'response_funcs': [],
                           'sky_resp': sky_resp,
                           'coeffs': 1,
                           'theta': 0
                           }

        self.user_profiles = {
                              'sky': None,
                              'PRNU': None,
                              'QE': None,
                              'T': None,
                              'DC': None,
                              'DNFP': None,
                              'Bias': None,
                             }
        if user_profiles is not None:
            self.user_profiles.update(user_profiles)
        if tel_params is not None:
            self.tel_params.update(tel_params)

        self.det_params = {
                          'shot_noise': 'Gaussian',
                          'M_sky': 27,
                          'qe_response':  [],     # Wavelength dependence
                          'qe_mean': 0.95,        # Effective QE
                          'bias': 35,             # electrons
                          'G1': 1,
                          'bit_res': 14,
                          'RN':  5,               # elec/pix
                          'PRNU_frac': 0.25/100,  # PRNU sigma
                          'T': 218,               # K
                          'DFM': 1.424e-2,        # 14.24 pA
                          'pixel_area': 1e-6,     # cm2
                          'DCNU': 0.1/100,        # percentage
                          'DNFP': 0.,             # electrons
                          'NF': 0.,               # electrons
                          'FWC': 1.4e5,           # electrons
                          'C_ray_r': 2/50         # hits/second
                          }


        self.df = df.copy()
        self.n_x = n_x
        self.n_y = n_y
        self.pixel_scale = self.tel_params['pixel_scale']
        self.theta = self.tel_params['theta']*np.pi/180

        self.response_funcs = self.tel_params['response_funcs']
        self.coeffs = self.tel_params['coeffs']

        # DN/electrons
        self.gain = pow(2, self.det_params['bit_res'])/self.det_params['FWC']
        self.gain *= self.det_params['G1']

        self.tel_area = np.pi*(self.tel_params['aperture']/2)**2

        self.exp_time = exp_time  # seconds

        self.psf_file = self.tel_params['psf_file']

        self.check_df()

        if coords is None:
            self.ra = np.median(self.df['ra'])
            self.dec = np.median(self.df['dec'])
        else:
            self.ra = coords[0]
            self.dec = coords[1]

        ra_n = np.round(self.ra,  3)
        dec_n = np.round(self.dec, 3)
        self.name = f" RA : {ra_n} degrees, Dec : {dec_n} degrees"

        self.generate_sim_field(plot)

        # Init Cosmic Rays
        # area = ((n_x*self.pixel_scale)/3600)*((n_y*self.pixel_scale)/3600)

        # eff_area = (0.17/area)*self.exp_time

        # self.n_cosmic_ray_hits = int(eff_area*self.det_params['C_ray_r'])

    def generate_sim_field(self, plot):
        """This function creates array with FoV a bit wider
        than user defined size for flux conservation"""
        if self.df is not None:
            self.calc_zp(plot=plot)
            self.init_psf_patch()

            # Cropping df to sim_field
            x_left = self.n_pix_psf//2
            x_right = self.n_x_sim - self.n_pix_psf//2
            y_left = self.n_pix_psf//2
            y_right = self.n_y_sim - self.n_pix_psf//2

            self.sim_df = self.init_df(df=self.df,
                                       n_x=self.n_x_sim, n_y=self.n_y_sim,
                                       x_left=x_left, x_right=x_right,
                                       y_left=y_left, y_right=y_right)
            if len(self.sim_df) < 1:
                print("Not Enough sources inside FoV. Increase n_x\
                                and n_y")
        else:
            print("df cannot be None")

    def check_df(self):
        # Input Dataframe
        if 'mag' not in self.df.keys():
            raise Exception("'mag' column not found input dataframe")

        if 'ra' not in self.df or 'dec' not in self.df.keys():
            if 'x' in self.df.keys() and 'y' in self.df.keys():
                print("Converting xy to ra-dec")
                self.df.x = self.df.x + 2
                self.df.y = self.df.y + 2
                self.df = self.xy_to_radec(self.df, self.n_x, self.n_y,
                                           self.pixel_scale)
            else:
                raise Exception("'ra','dec','x',or 'y', \
                 columns not found in input dataframe ")

    def calc_zp(self, plot=False):
        if len(self.response_funcs) > 0:
            wav = np.linspace(1000, 10000, 10000)
            flux = 3631/(3.34e4*wav**2)   # AB flux

            fig, ax, _, params = bandpass(wav, flux, self.response_funcs,
                                          plot=plot)

            lambda_phot, int_flux, int_flux_Jy, W_eff, flux_ratio = params

            self.lambda_phot = lambda_phot
            self.int_flux = int_flux
            self.W_eff = W_eff
            self.int_flux_Jy = int_flux_Jy
            self.flux_ratio = flux_ratio

            filt_dat = np.loadtxt(self.tel_params['sky_resp'])

            wav = filt_dat[:, 0]
            flux = filt_dat[:, 1]

            _, _, _, params = bandpass(wav, flux, self.response_funcs,
                                       plot=False)

            int_flux = params[1]
            self.det_params['M_sky'] = int_flux

        else:

            print("Response functions not provided. Using default values")
            self.int_flux_Jy = 3631
            self.W_eff = 1000
            self.lambda_phot = 2250
            self.flux_ratio = 1

        self.photons = 1.51e3*self.int_flux_Jy*(self.W_eff/self.lambda_phot)
        self.photons *= self.flux_ratio

        self.zero_flux = self.exp_time*self.tel_area*self.photons
        self.zero_flux *= self.coeffs
        self.M_sky_p = self.det_params['M_sky'] \
            - 2.5*np.log10(self.pixel_scale**2)
        self.sky_bag_flux = self.zero_flux*pow(10, -0.4*self.M_sky_p)

        if self.sky:
            if self.user_profiles['sky'] is not None:
                if self.user_profiles['sky'].shape == (self.n_x, self.n_y):
                    self.sky_photons = self.user_profiles['sky']
                else:
                    raise Exception(f"""User defined sky array shape: \
                    {self.user_profiles['sky'].shape} \
                    is not same as detector shape {(self.n_x, self.n_y)}""")
            else:
                self.sky_photons = self.compute_shot_noise(self.sky_bag_flux)
        else:
            self.sky_photons = 0

    def init_psf_patch(self, return_psf=False):
        """Creates PSF array from NPY or fits files"""
        ext = self.psf_file.split('.')[-1]

        if ext == 'npy':
            image = np.load(self.psf_file)
        elif ext == 'fits':
            image = fits.open(self.psf_file)[0].data

        image /= image.sum()  # Flux normalized to 1
        self.psf = image

        self.n_pix_psf = self.psf.shape[0]

        # Defining shape of simulation field
        self.n_x_sim = self.n_x + 2*(self.n_pix_psf-1)
        self.n_y_sim = self.n_y + 2*(self.n_pix_psf-1)

        if return_psf:
            return image*self.zero_flux

    def init_df(self, df, n_x, n_y, x_left, x_right, y_left, y_right):
        """Bounds sources to boundary defined by x and y limits"""
        wcs = self.create_wcs(n_x, n_y, self.ra, self.dec,
                              self.pixel_scale, self.theta)

        c = SkyCoord(df['ra'], df['dec'], unit=u.deg)
        pix = wcs.world_to_array_index(c)
        df['x'] = pix[1]
        df['y'] = pix[0]

        # Cropping Dataframe based on FoV
        x_min_cut = (df['x'] > x_left)
        x_max_cut = (df['x'] < x_right)

        df = df[x_min_cut & x_max_cut]

        y_min_cut = (df['y'] > y_left)
        y_max_cut = (df['y'] < y_right)

        df = df[y_min_cut & y_max_cut]

        return df

    def init_image_array(self, return_img=False):
        """
        Creates a base image array for adding photons

        Parameters
        ----------
        return_img : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        numpy.ndarray
            if return_img is true return base image array

        """
        image = np.zeros((self.n_y_sim, self.n_x_sim))
        wcs = self.create_wcs(self.n_x_sim, self.n_y_sim,
                                   self.ra, self.dec, self.pixel_scale,
                                   self.theta)

        return image, wcs

    def xy_to_radec(self, df, n_x, n_y, pixel_scale):

        w = WCS(naxis=2)
        w.wcs.crpix = [n_x//2, n_y//2]
        w.wcs.cdelt = np.array([-pixel_scale/3600, pixel_scale/3600])
        w.wcs.crval = [10, 10]
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        pos = np.array([df.x, df.y])
        coords = np.array(w.pixel_to_world_values(pos.T))
        df['ra'] = coords[:, 0]
        df['dec'] = coords[:, 1]

        return df

    def create_wcs(self, n_x, n_y, ra, dec, pixel_scale, theta=0):
        """
        Parameters
        ----------
        n_x : int
            number of pixels in RA direction
        n_y : int
            number of pixels in Dec direction
        ra : float (degrees)
            right ascension of center of image.
        dec : float (degrees)
            declination of center of image.
        pixel_scale : floats
            arcsecs/pixel.

        Returns
        -------
        w : wcs object

        """
        w = WCS(naxis=2)
        w.wcs.crpix = [n_x//2, n_y//2]
        w.wcs.cdelt = np.array([-pixel_scale/3600, self.pixel_scale/3600])
        w.wcs.crval = [ra, dec]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.pc = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
        return w

    def compute_shot_noise(self, array, type_='Poisson'):
        """
        Parameters
        ----------
        array : numpy.ndarray
            input array
        type_ : str, optional
             The default is 'Poisson'.
        Returns
        -------
        shot_noise : numpy.ndarray
             Return array with shot noise
        """
        if type(array) == np.float64 or type(array) == float:
            n_x = self.n_y
            n_y = self.n_x
        else:
            n_x = array.shape[0]
            n_y = array.shape[1]
        if type_ == 'Gaussian':
            shot_noise = np.random.normal(loc=array, scale=np.sqrt(array),
                                          size=(n_x, n_y))
        elif type_ == 'Poisson':
            shot_noise = np.random.poisson(lam=array,
                                           size=(n_x, n_y)
                                           ).astype(np.float64)
        else:
            print('Invalid type')
        return shot_noise

    def compute_coeff_arrays(self):
        """

        Computed coefficients based on input parameters
        Returns
        -------
        None.
        """
        n_x = self.n_y
        n_y = self.n_x

        if self.QE:
            if len(self.det_params['qe_response']) != 0:
                wav = np.linspace(1000, 10000, 10000)
                flux = 3631/(3.34e4*wav**2)   # AB flux

                _, _, _, params = bandpass(wav, flux,
                                           self.det_params['qe_response'],
                                           plot=False)

                _, _, _, _, flux_ratio = params
                self.det_params['qe_mean'] = flux_ratio
        else:
            self.det_params['qe_mean'] = 1

        if self.user_profiles['Bias'] is not None:
            if self.user_profiles['Bias'].shape == (n_x, n_y):
                self.bias_array = self.user_profiles['Bias']
            else:
                raise Exception(f"""User defined Bias array shape: \
                {self.user_profiles['Bias'].shape} \
                is not same as detector shape {(n_x,n_y)}""")
        else:
            self.bias_array = np.random.normal(loc=self.det_params['bias'],
                                               scale=self.det_params['RN'],
                                               size=(n_x, n_y))
        if self.PRNU:
            if self.user_profiles['PRNU'] is not None:
                if self.user_profiles['PRNU'].shape == (n_x, n_y):
                    self.PRNU_array = self.user_profiles['PRNU']
                else:
                    raise Exception(f"""User defined PRNU array shape: \
                    {self.user_profiles['PRNU'].shape} \
                    is not same as detector shape {(n_x,n_y)}""")
            else:
                scale = self.det_params['PRNU_frac']
                self.PRNU_array = np.random.normal(loc=0,
                                                   scale=scale,
                                                   size=(n_x, n_y))
        else:
            self.PRNU_array = 0

        if self.DC:
            if self.user_profiles['DC'] is not None:
                if self.user_profiles['DC'].shape == (n_x, n_y):
                    self.DR = self.user_profiles['DC']
                else:
                    raise Exception(f"""User defined DC array shape:
                    {self.user_profiles['DC'].shape}
                    is not same as detector shape {(n_x,n_y)}""")
            else:
                if self.user_profiles['T'] is not None:
                    if self.user_profiles['T'].shape == (n_x, n_y):
                        area = self.det_params['pixel_area']
                        self.DR = self.dark_current(self.user_profiles['T'],
                                                    self.det_params['DFM'],
                                                    area)
                    else:
                        raise Exception(f"""User defined DC array shape:
                        {self.user_profiles['DC'].shape}
                        is not same as detector shape {(n_x,n_y)}""")
                else:
                    self.DR = self.dark_current(self.det_params['T'],
                                                self.det_params['DFM'],
                                                self.det_params['pixel_area'])
            # Dark Current Non-uniformity
            if self.DCNU:
                sigma = self.det_params['DCNU']
                self.DCNU_array = np.random.lognormal(mean=0,
                                                      sigma=sigma,
                                                      size=(n_x, n_y))
                self.DR *= self.DCNU_array
            self.DC_array = self.compute_shot_noise(self.DR*self.exp_time)
            # Dark Current Fixed Pattern
            if self.DNFP:
                if self.user_profiles['DNFP'] is not None:
                    if self.user_profiles['DNFP'].shape == (n_x, n_y):
                        self.DNFP_array = self.user_profiles['DNFP']
                    else:
                        raise Exception(f"""User defined DNFP array shape:
                                            {self.user_profiles['DNFP'].shape}
                                is not same as detector shape {(n_x,n_y)}""")
                else:
                    arr = self.compute_shot_noise(self.det_params['DNFP'])
                    self.DNFP_array = arr
                self.DC_array += self.DNFP_array
        else:
            self.DR = 0
            self.DC_array = 0
        # Quantization Noise
        if self.QN:
            # electrons
            A = self.det_params['FWC']
            B = pow(2, self.det_params['bit_res'])*np.sqrt(12)
            self.QN_value = (A/B)
            self.QN_array = self.QN_value*np.random.randint(-1, 2,
                                                            size=(n_x, n_y))
        else:
            self.QN_array = 0

    def dark_current(self, T, DFM, pixel_area):
        """
          Parameters
          ----------
          T : float
              Detector Temperature
          DFM : float
              Dark current figure of merit
          pixel_area : float
              Area of pixel

          Returns
          -------
          DR : float
              Dark current rate e/s/pixels
        """
        Kb = 8.62e-5
        const = 2.55741439581387e15

        EgT = 1.1557 - (7.021e-4*T**2/(1108+T))
        DR = const*pixel_area*(T**1.5)*DFM*np.exp(-EgT/(2*Kb*T))
        return DR

    def generate_photons(self, image, patch_width, df):
        """
          This function creates sims based on ABmag  on a
          small patch (2D array) of size n_pix_s*n_pix_s.

          The patch with the sim is then added to the image array of size
          n_pix_m*n_pix_m using wcs object.

          Parameters
          ----------
          image       : numpy.ndarray
                        base image array for inserting star sims
          patch_width : int
                        number of pixels (length) in sim patch image
          df          : pandas.dataframe
                        Dataframe containing source list
          Returns
          -------
          image : numpy.ndarray
              Array with sims added based on df

        """
        patch_width_mid = patch_width//2

        x0, y0 = df['x'].astype(int), df['y'].astype(int)
        ABmag = df['mag'].values
        flux = self.zero_flux * 10**(-ABmag/2.5)
        fluxes=[]

        patch = self.psf

        x1 = x0 - patch_width_mid
        x2 = x1 + patch_width
        y1 = y0 - patch_width_mid
        y2 = y1 + patch_width

        for x1_, x2_, y1_, y2_, flux_ in zip(x1, x2, y1, y2, flux):
            image[y1_:y2_, x1_:x2_] += flux_*patch
            if flux_!=np.NaN:
              fluxes.append(flux_)
        self.t=fluxes
        image = image[patch_width-1:-patch_width+1,
                      patch_width-1:-patch_width+1]


        return image

    def make_ccd_image(self, light_array):
        # Compute Coefficient arrays
        self.compute_coeff_arrays()

        # QE pixel to pixel variation | Source photoelectrons
        self.source_photoelec = light_array*self.det_params['qe_mean']

        # Photon Response (Quantum Efficiency) Non Uniformity
        if self.PRNU:
            self.source_photoelec *= (1+self.PRNU_array)

        # Dark Current. Includes DCNU, DNFP and shot noise
        self.photoelec_array = self.source_photoelec + self.DC_array

        # Addition of Quantization error, Bias and Noise floor
        self.charge = self.photoelec_array + self.QN_array \
                                           + self.det_params['NF'] \
                                           + self.bias_array

        # Photoelec to ADUs
        digital = (self.charge*self.gain).astype(int)

        # Full well condition
        digital = np.where(digital >= pow(2, self.det_params['bit_res']),
                           pow(2, self.det_params['bit_res']), digital)
        return digital

    @property
    def dark_frame(self):
        return self.make_ccd_image(0)

    @property
    def flat_frame(self):
        flat = self.make_ccd_image(10)
        flat = flat/flat.max()
        return flat

    def __call__(self, det_params=None, n_stack=1, stack_type='median',
                 photometry='Aper', fwhm=3, sigma=3, detect_sources=False,
                 ZP=None, **kwargs):
        """
          Parameters
          ----------
        det_params: dict, optional
          Dictionary contianing detector parameters. The default is None.
                    {     'shot_noise' :  str,
                          'M_sky'      :  float,
                          'qe_mean'    :  float,  photons to photoelectrons
                          'bias'       :  float,       electrons
                          'G1'         :  float,
                          'bit_res'    :  int,
                          'RN'         :  float,       elec/pix
                          'PRNU_frac'  :  float,       PRNU sigma
                          'T'          :  float,       K
                          'DFM'        :  float,       pA
                          'pixel_area' :  float,
                          'DCNU'       :  float        fraction
                          'DNFP'       :  float,       electrons
                          'NF'         :  float,       electrons
                          'FWC'        :  float,       electrons
                          'C_ray_r'    :  float        hits/second
                      }
          n_stack    : int, optional
                      Number of observations to be stacked. The default is 1.

          stack_type : str, optional
                      Stacking method. The default is 'median'.
        photometry : str,
                      Type of photometry to be employed
                      Choose from
                      'Aper' : Aperture photometry using Photutils
                      'PSF'  : PSF photometry using DAOPHOT
                      None   : Simulate without photometry
        fwhm : float, pixels
                During aperture photometry,
                fwhm corresponds to FWHM circular aperture for
                aperture photometry
                During PSF photometry,
                fwhm corresponds FWHM kernel to use for PSF photometry
        sigma: float,
                The numbers of standard deviations above which source has to be
                detected
        detect: bool,
                If true, DARStarFinder is used to detect sources for aperture
                photometry

                if false, input catalog is used for getting positions
                of sources for aperture photometry
        ZP    : float,
                zero point of the telescope.
                Default None, zero point is calculated theoretically or using
                input catalog
        Simulates field by taking dataframe as input by inserting sim patches
        in image array using WCS
          Returns
          -------
          numpy.ndarray
          Final image array after adding all layers of simulation

        """

        if det_params is not None:

            self.det_params.update(det_params)
            A = pow(2, self.det_params['bit_res'])
            B = self.det_params['FWC']
            self.gain = A/B

            self.gain *= self.det_params['G1']

        digital_stack = []

        for i in range(n_stack):

            image, _ = self.init_image_array()

            # Source photons
            self.source_photons = self.generate_photons(image,
                                                        self.n_pix_psf,
                                                        self.sim_df)
            # Sky photons added to source
            self.light_array = (self.source_photons + self.sky_photons)

            # Source shot_noise
            if self.shot_noise:
                type_ = self.det_params['shot_noise']
                self.light_array = self.compute_shot_noise(self.light_array,
                                                           type_=type_)
            self.digital = self.make_ccd_image(self.light_array)

            digital_stack.append(self.digital)

        digital_stack = np.array(digital_stack)
        if n_stack > 1:
            if stack_type == 'median':
                self.digital = np.median(digital_stack, axis=0)
            elif stack_type == 'mean':
                self.digital = np.median(digital_stack, axis=0)
        if self.cosmic_rays:
            for i in range(self.n_cosmic_ray_hits):
                x = np.random.randint(0, self.n_x_main)
                y = np.random.randint(0, self.n_y_main)
                self.digital[x, y] = pow(2, self.det_params['bit_res'])

        self.digital = self.digital.astype(np.int16)
        self.wcs = self.create_wcs(self.n_x, self.n_y,
                                   self.ra, self.dec,
                                   self.pixel_scale, self.theta)

        self.sim_flag = True
        # Filtering out sources within Image
        x_left = 0
        x_right = self.n_x
        y_left = 0
        y_right = self.n_y

        self.img_df = self.init_df(df=self.sim_df.copy(),
                                   n_x=self.n_x, n_y=self.n_y,
                                   x_left=x_left, x_right=x_right,
                                   y_left=y_left, y_right=y_right)

        self.header = self.wcs.to_header()
        self.header['gain'] = self.det_params['G1']
        self.header['Temp'] = str(self.det_params['T']) + 'K'
        self.header['bias'] = self.det_params['bias']
        self.header['RN'] = self.det_params['RN']
        self.header['DR'] = np.mean(self.DR)
        self.header['NF'] = self.det_params['NF']
        self.header['EXPTIME'] = self.exp_time
        self.header['BUNIT'] = 'DN'

        self.org_digital = self.digital.astype(float).copy()

        if ZP is None:
            QE = self.det_params['qe_mean']
            zero_p_flux = self.zero_flux*QE + self.sky_bag_flux
            zero_p_flux += np.mean(self.DR)*self.exp_time
            zero_p_flux += self.det_params['NF'] + self.det_params['bias']

            zero_p_flux *= self.gain
            ZP = 2.5*np.log10(zero_p_flux)
            # Observed Zero Point Magnitude in DN
            self.ZP = ZP
        self.header['ZP'] = self.ZP

        super().__call__(df=self.img_df, wcs=self.wcs,
                         data=self.digital.astype(float),
                         photometry=photometry, fwhm=fwhm, sigma=sigma,
                         detect_sources=detect_sources, ZP=ZP)

    def add_distortion(self, xmap, ymap):
        """Function for addition distortion using
        x and y mappings"""
        self.x_map = xmap
        self.y_map = ymap
        # Interpolation to be added
        data = self.digital.astype(float).copy()
        distorted_img = cv2.remap(data, xmap.astype(np.float32),
                                  ymap.astype(np.float32), cv2.INTER_LANCZOS4)
        distorted_img = distorted_img.astype(int)
        self.digital = np.where(distorted_img > 0, distorted_img, 1)

    def remove_distortion(self):
        """Function for returning the image to state
        before adding distortion"""
        # undistort to be added
        self.digital = self.org_digital

    def __del__(self):
      for i in self.__dict__:
        del i
        
#MOSAICING

class Mosaic(Imager):
    """
    A class to split bigger images to tiles and stitch them together

    """

    def __init__(self, df=None, coords=None, ras=None, decs=None,
                 tel_params=None, exp_time=100,
                 n_x=1000, n_y=1000, mos_n=1, mos_m=1, **kwargs):

        """
        Analyzer.init()

        Parameters
        ----------
        mos_n : int,
                number of tiles in RA direction

        mos_m : int,
                number of tiles in Dec direction
        """

        super().__init__(df=df, coords=coords, exp_time=exp_time, n_x=n_x,
                         n_y=n_y, tel_params=tel_params, **kwargs)
        self.n = mos_n
        self.m = mos_m

        if ras is None or decs is None:
            self.mosaic_ra = self.ra
            self.mosaic_dec = self.dec
            self.mosaic_n_x = n_x
            self.mosaic_n_y = n_y
            self.mosaic_df = df
            self.mosaic_wcs = self.create_wcs(self.mosaic_n_x, self.mosaic_n_y,
                                              self.mosaic_ra, self.mosaic_dec,
                                              self.pixel_scale)

            self.df_split(df, self.n, self.m, n_x, n_y, self.mosaic_wcs)
        else:
            self.ras = ras
            self.decs = decs

    def df_split(self, df, n, m, n_x, n_y, wcs):

        """
        Function to split dataframe based shape and number of tiles

        Parameters
        ----------
        n : int,
            number of tiles in RA direction

        m : int,
            number of tiles in Dec direction

        n_x : int,
            number of pixels in RA direction

        n_y : int,
            number of pixels in Dec direction
        """

        x_bins = np.linspace(0, n_x, n+1)
        y_bins = np.linspace(0, n_y, m+1)

        x_cens = 0.5*(x_bins[:-1] + x_bins[1:]) - 1
        y_cens = 0.5*(y_bins[:-1] + y_bins[1:]) - 1
        cens = wcs.array_index_to_world_values(y_cens, x_cens)
        ra_cens = cens[0]
        ra_cens = np.where(np.round(ra_cens, 1) == 360, ra_cens - 360, ra_cens)
        dec_cens = cens[1]

        self.ras = ra_cens
        self.decs = dec_cens
        self.filenames = []

    def __call__(self, det_params=None, n_stack=1, stack_type='median',
                 photometry=True, fwhm=None):
        """
          Imager.call()

          Calls the Imager class iteratively to generate image tiles
          and stitches the tiles together

        """
        # Flags
        n_x = self.n_x//self.n + 50
        n_y = self.n_y//self.m + 50

        for i in range(self.n):
            for j in range(self.m):

                df = self.mosaic_df
                coords = (self.ras[i], self.decs[j])
                exp_time = self.exp_time
                tel_params = self.tel_params
                det_params['T'] += np.random.randint(-3, 3)
                np.random.seed(i+2*j)
                super().__init__(df=df, coords=coords, exp_time=exp_time,
                                 n_x=n_x,
                                 n_y=n_y, tel_params=tel_params)
                np.random.seed(i+2*j)
                super().__call__(det_params=det_params, n_stack=n_stack,
                                 stack_type=stack_type, photometry=None)

                self.writeto(f'{i}{j}_mosaic.fits')
                self.filenames.append(f'{i}{j}_mosaic.fits')

    def make_mosaic(self):
        hdus = []
        for fil in self.filenames:
            hdu = fits.open(fil)[0]
            hdus.append(hdu)

        wcs_out, shape_out = find_optimal_celestial_wcs(hdus, frame='icrs',
                                                        auto_rotate=True)

        array, fp = reproject_and_coadd(hdus,
                                        wcs_out, shape_out=(shape_out),
                                        reproject_function=reproject_interp)

        self.wcs = wcs_out
        self.digital = array
        self.footprint = fp
        
#SPECTROSCOPY

class Spectrometer(Imager, Analyzer, SpecAnalyzer):
    """Class for simulating Spectra"""
    def __init__(self, df, coords=None, tel_params={}, n_x=1000,
                 n_y=1000, exp_time=100, plot=False, user_profiles={},
                 **kwargs):

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.tel_params = tel_params
        bin_min = self.tel_params['lambda1'] - self.tel_params['dellambda']/2
        bin_max = self.tel_params['lambda2'] + self.tel_params['dellambda']
        self.bins = np.arange(bin_min,
                              bin_max,
                              self.tel_params['dellambda'])

        self.spec_bins = 0.5*(self.bins[1:] + self.bins[:-1])
        super().__init__(df, coords, tel_params, n_x, n_y, exp_time, plot,
                         user_profiles, **kwargs)

        self.generate_sim_field(plot)
        self.sky = False

        self.sim_df = redshift_corr(self.sim_df)

    def check_df(self):
        pass

    def generate_sim_field(self, plot):
        if self.df is not None:
            self.calc_zp(plot=plot)
            self.init_psf_patch()
            spec_corr = len(self.spec_bins)//2

            x_left  = spec_corr + self.n_pix_psf
            x_right = self.n_x_sim - spec_corr  - self.n_pix_psf
            y_left  = self.n_pix_psf
            y_right = self.n_y_sim - self.n_pix_psf - 1

            self.sim_df = self.init_df(df=self.df,
                                       n_x=self.n_x_sim, n_y=self.n_y_sim,
                                       x_left=x_left, x_right=x_right,
                                       y_left=y_left, y_right=y_right)
        else:
            print("df cannot be None")

    def init_MOS_df(self, L, B, PA, l, df):
      delta = np.arctan(B/L)
      d = np.sqrt(L**2 + B**2)/2
      om = PA - delta

      x_corr = self.n_x_sim//2 - d*np.cos(om) - l/2
      y_corr = self.n_y_sim//2 - d*np.sin(om) - B*np.cos(PA)

      t = df.copy()
      x0 = B*np.sin(PA) + l/2 + x_corr
      y0 = y_corr
      t = t[t['y'] > np.tan(PA)*(t['x'] - x0) + y0]

      x0 = l/2 + x_corr
      y0 = B*np.cos(PA) + y_corr
      t = t[t['y'] < np.tan(PA)*(t['x'] - x0) + y0]

      x0 = l/2 + x_corr
      y0 =  B*np.cos(PA) + y_corr
      t = t[t['x'] > -np.tan(PA)*(t['y'] - y0) + x0]

      x0 = L*np.cos(PA) + l/2 + x_corr
      y0 = B*np.cos(PA) + L*np.sin(PA) + y_corr
      t = t[t['x'] < -np.tan(PA)*(t['y'] - y0) + x0]

      return t

    def init_MOS(self,L,B,PA):
      PA *= np.pi/180
      l = len(self.spec_bins)
      x_size, y_size = calc_mos_size(L,B,PA,l)
      if x_size<self.n_x and y_size<self.n_y:
        self.L = L
        self.B = B
        self.PA = PA

        self.sim_df = self.init_MOS_df(L, B, PA, l, self.sim_df)
        self.sim_df['objid'] = np.arange(0,len(self.sim_df),1)
      else:
        print(f"""n_x should be greater than {x_size} \n
                n_y should be greater than {y_size}""")

    def select_MOS_sources(self, df=None, ids=[], radius=5, min_sep=8):
      if df is None:
        df = self.sim_df

      x = df['x'].value
      y = df['y'].value

      cen_x = x[np.argmin(abs(x-np.mean(x)))]
      cen_y = y[np.argmin(abs(y-np.mean(y)))]

      mos_df, res_df, cen_x, cen_y = select_mos(df[['objid','x','y']].to_pandas()
                                                ,cen_x, cen_y, ids, radius,
                                                min_sep, self.n_y_sim)
      t = []
      for id in mos_df['objid']:
        t.append(df[df['objid']==id])

      mos_df = vstack(t)
      self.sim_df = mos_df

      t = []
      for id in res_df['objid']:
        t.append(df[df['objid']==id])

      res_df = vstack(t)

      return mos_df, res_df, cen_x, cen_y

    def init_psf_patch(self, return_psf=False, plot=False):

        ext = self.psf_file.split('.')[-1]

        if ext == 'npy':
            image = np.load(self.psf_file)
        elif ext == 'fits':
            image = fits.open(self.psf_file)[0].data

        image /= image.sum()  # Flux normalized to 1

        self.psf = image

        self.n_pix_psf = self.psf.shape[0]

        spec_corr = len(self.spec_bins) + 1
        self.n_x_sim = self.n_x + spec_corr + 2*(self.n_pix_psf-1)
        self.n_y_sim = self.n_y + 2*(self.n_pix_psf-1)

        if return_psf:
            return image*self.zero_flux

    def calc_zp(self, plot=False):
        if len(self.response_funcs) > 0:

            wav = np.linspace(1000, 10000, 10000)
            flux = 3631/(3.34e4*wav**2)   # AB flux

            fig, ax, data, _ = bandpass(wav, flux, self.response_funcs,
                                        plot=plot)

            lambda_, _, Reff = data
            self.Reff = self.bin_xy_norm(lambda_, Reff)

            self.flux_coeffs = self.Reff.copy()
            self.flux_coeffs *= self.exp_time*self.coeffs*self.tel_area

            filt_dat = np.loadtxt(f'{data_path}/Sky_mag.dat')
            wav = filt_dat[:, 0]
            flux = filt_dat[:, 1]

            fig, ax, data, _ = bandpass(wav, flux, self.response_funcs,
                                        plot=False)
            lambda_, conv_flux, _ = data

            self.det_params['M_sky'] = conv_flux
            M_corr = 2.5*np.log10(self.pixel_scale**2)
            self.M_sky_p = self.det_params['M_sky'] - M_corr

            sky_bag_flux = 3631*pow(10, -0.4*self.M_sky_p)

            wav_new = np.arange(lambda_[0], lambda_[-1],
                                self.tel_params['dellambda']/10)

            flux_new = np.interp(wav_new, lambda_, sky_bag_flux)

            self.sky_bag_flux = self.bin_xy(wav_new, flux_new)*self.flux_coeffs
            dl = self.tel_params['dellambda']
            lam = self.spec_bins
            self.sky_photons = 1.51e3*self.sky_bag_flux*(dl/lam)
            self.sky_photons = 0
            self.zero_flux = 1
        else:
            raise Exception("Response Functions not Provided")

    def init_image_array(self, return_img=False):
        """
        Creates a base image array for adding photons

        Parameters
        ----------
        return_img : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        numpy.ndarray
            if return_img is true return base image array

        """
        image = np.zeros((self.n_y_sim, self.n_x_sim))
        self.mask = np.zeros((self.n_y_sim, self.n_x_sim))
        wcs = self.create_wcs(self.n_x_sim, self.n_y_sim,
                                   self.ra, self.dec, self.pixel_scale,
                                   self.theta)

        return image, wcs

    def bin_xy(self, x, y):
        y_binned = []
        for i in range(len(self.bins) - 1):
            indices = np.where((x > self.bins[i]) & (x <= self.bins[i+1]))[0]
            if len(indices > 0):
                y_binned.append(y[indices].sum())
            else:
                y_binned.append(0)
        return np.array(y_binned)

    def bin_xy_norm(self, x, y):
        y_binned = []
        for i in range(len(self.bins) - 1):
            indices = np.where((x > self.bins[i]) & (x <= self.bins[i+1]))[0]
            if len(indices > 0):
                y_binned.append(np.median(y[indices]))
            else:
                y_binned.append(0)
        return np.array(y_binned)

    def generate_photons(self, image, patch_width, df):
        """
        This function creates sims based on ABmag  on a
        small patch (2D array) of size n_pix_s*n_pix_s.

        The patch with the sim is then added to the image array of size
        n_pix_m*n_pix_m using wcs object.

        Parameters
        ----------
        image       : numpy.ndarray
                    base image array for inserting star sims
        patch_width : int
                    number of pixels (length) in sim patch image
        df          : pandas.dataframe
                    Dataframe containing source list


        Returns
        -------
        image : numpy.ndarray
            Array with sims added based on df

        """
        patch_width_mid = patch_width//2
        disp_width = len(self.spec_bins)
        disp_width_mid = disp_width//2

        patch = self.psf
        device = self.device

        # Convert to tensor once, before the loop
        image = torch.tensor(image, dtype=torch.float32, device=device)
        flux_coeffs = torch.tensor(self.flux_coeffs, dtype=torch.float32,
                                   device=device)
        del_lambda = torch.tensor(self.tel_params['dellambda'],
                                  dtype=torch.float32, device=device)
        lambda_ = torch.tensor(self.spec_bins, dtype=torch.float32,
                               device=device)
        const = torch.tensor(1.51e3, dtype=torch.float32, device=device)

        for row in df:

            x = row['wav']
            y = row['flux']

            flux = self.bin_xy(x, y)
            wav = self.bin_xy_norm(x, x)

            flux = flux*wav**2*3.34e4
            flux = torch.tensor(flux, dtype=torch.float32, device=device)
            flux *= flux_coeffs
            flux = const*flux*(del_lambda/lambda_)

            # Generating star using sim

            x0 = int(row['x'])
            y0 = int(row['y'])

            x1 = x0 - patch_width_mid - disp_width_mid + 1
            x2 = x1 + patch_width + disp_width - 1
            y1 = y0 - patch_width_mid
            y2 = y1 + patch_width

            data = torch.tensor(flux, dtype=torch.float32, device=device)
            data = data.view(1, 1, 1, -1)
            kernel = torch.tensor(patch, dtype=torch.float32, device=device)
            kernel = kernel.view(1, 1, patch_width, patch_width)
            spectrum = F.conv2d(data, kernel, padding=patch_width-1)

            image[y1:y2, x1:x2] += spectrum.squeeze()

            self.mask[y0 - 3:y0 + 4, x1:x2] += 0.5

            # Empty GPU cache after each iteration
            torch.cuda.empty_cache()

        # Convert to numpy array once, outside of the loop
        image = image.detach().cpu().numpy()
        spec_corr = (len(self.spec_bins) + 1)//2
        image = image[patch_width-1:-patch_width+1,  # Cropping Image
                      patch_width+disp_width_mid-1:-disp_width_mid-patch_width-1]

        return image

    def __call__(self, det_params=None, n_stack=1, stack_type='median',
                 photometry=None, fwhm=None, detect_sources=False, **kwargs):
        self.ZP = 1
        super().__call__(det_params=det_params, n_stack=n_stack,
                         photometry=None,
                         stack_type=stack_type, ZP=1,**kwargs)

        self.extract(data=self.digital, wcs=self.wcs, df=self.img_df)
        
file = '/home/hp/Downloads/HSCv3_ngc3368 (1).csv'
df_img = pd.read_csv(file)
print(df_img)

df2 = df_img["MatchRA"].mean()
RAcen=df2
#print(RAcen)
df3 = df_img["MatchDec"].mean()
deccen=df3
#print(deccen)

RA_col=df_img["MatchRA"]
RA_col = np.array(RA_col)
#print(RA_col)
dec_col=df_img["MatchDec"]
dec_col = np.array(dec_col)
#print(dec_col)

distance1=0.85*u.Mpc
distance2=50*u.Mpc

a,b=distance_transform(RA_col, dec_col, RAcen, deccen, distance1, distance2)
print(a)
print(b)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot()
ax.set_facecolor('black')
ax.scatter(a, b, s = 0.005, color  ='white' )
ax.invert_xaxis()
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
plt.savefig('image_NGC3368.pdf')

df_img['MatchRA']=a
df_img['MatchDec']=b

#y1 = df_img['INSIST_g']
#y2 = df_img['INSIST_U']
#y3 = df_img['INSIST_UV']

#IMAGER
y,x = 2*np.mgrid[0:100,0:100]/3600

ra  = 10 + x.ravel()
dec = 10 + y.ravel()
mag = np.linspace(22,30,10000)
df  = pd.DataFrame(zip(ra,dec,mag), columns = ['ra','dec','mag'])

tel_params ={
            'aperture'       : 100,
            'pixel_scale'    : 0.1,
            'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
            'response_funcs' :  [ f'{data_path}/INSIST/UV/Coating.dat,5,100',   # 6 mirrors
                                  f'{data_path}/INSIST/UV/Filter.dat,1,100',
                                  f'{data_path}/INSIST/UV/Dichroic.dat,2,100',   # 2 dichroics
                                ],
             'coeffs'       : 1, #0.17
             'theta'        : 0
            }

tel_params_U ={
            'aperture'       : 100,
            'pixel_scale'    : 0.1,
            'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
            'response_funcs' :  [ f'{data_path}/INSIST/U/M1.dat,5,100',   # 5 mirrors
                                  f'{data_path}/INSIST/U/Filter.dat,2,100',    #2 filters
                                  f'{data_path}/INSIST/U/Dichroic.dat,1,100',   # 1 dichroics
                                ],
             'coeffs'       : 1, #0.17
             'theta'        : 0
            }

tel_params_g ={
            'aperture'       : 100,
            'pixel_scale'    : 0.1,
            'psf_file'       : f'{data_path}/PSF/INSIST/off_axis_poppy.npy',
            'response_funcs' :  [ f'{data_path}/INSIST/U/M1.dat,5,100',   # 5 mirrors
                                  f'{data_path}/INSIST/U/Filter.dat,1,100',    #1 filters
                                  f'{data_path}/INSIST/U/Dichroic.dat,2,100',   # 2 dichroics
                                ],
             'coeffs'       : 1, #0.17
             'theta'        : 0
            }
            
df_img['mag']=df_img['W3_F275W']
df_img['ra']= df_img['MatchRA']
df_img['dec'] = df_img['MatchDec']
df_img = df_img.dropna()

sim = Imager(df = df_img,tel_params = tel_params_g, exp_time = 2400, plot = False,
             n_x = 2000, n_y = 2000)
det_params = {'shot_noise' :  'Poisson',
              'qe_response': [],# [f'{data_path}/INSIST/UV/QE.dat,1,100'],
              'qe_mean'    : 0.95,
              'G1'         :  1,
              'bias'       : 10,
              'PRNU_frac'  :  0.25/100,
              'DCNU'       :  0.1/100,
              'RN'         :  3,
              'T'          :  218,
              'DN'         :  0.01/100
              }
sim.sim_df['mag']
sim(det_params = det_params, photometry = None, fwhm = 2,
    detect_sources  = True, n_stack = 1)
fig,ax = sim.show_image(cmap='gray', download=True)
plt.savefig('ngc3368.png')


             













