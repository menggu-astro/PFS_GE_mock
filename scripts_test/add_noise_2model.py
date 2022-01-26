# The code is used to add noise to the mock spectra models

import sys, os, pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import prospect
from astropy.io import fits
from sedpy import observate



# ---------------------------------------------------------------- # 
def spec_maggie2shotnoise(inwave, inspec, throughput = 0.225):
    """
    INPUTS:
        inwave: in unit of Angstrom
        throughput = 0.225 # from Table 1 of SSP
    OUTPUT:
        spec_snr
    """
    delta_lam = np.concatenate(((inwave[1:] - inwave[:-1]), 
                                np.atleast_1d(inwave[-1]-inwave[-2])))
    
    spectral_energy = np.copy(inspec) * 3631*1e-23 # erg/s/cm^2/Hz
    spectral_energy = np.copy(spectral_energy)*(3e18/inwave**2) # erg / s / cm^2 / A
    flux_Ang = np.copy(spectral_energy)
    spectral_energy = np.copy(spectral_energy)*delta_lam # now it is erg/s/cm^2
    #area = np.pi*(820)**2 # in cm^2
    # > Question: why this is not np.pi*(820./2)**2?
    area = np.pi*(820/2.)**2 # in cm^2
    
    spectral_energy *= area # now it is in erg/s
    photon_energy = 6.626e-27 * (3e18/inwave)
    nphot_per_second = spectral_energy / photon_energy * throughput
    nphot = nphot_per_second * (12 * 60 * 60) 
    spec_snr = np.sqrt(nphot) 
    return spec_snr


# ---------------------------------------------------------------- # 

if __name__=='__main__':
    filters = ['bessell_U']
    filters +=  ['sdss_'+filt+'0' for filt in ['g','r','i','z']]
    filters += ['uvista_y_cosmos','UDS_J']
    filters += ['IRAC_CH1','IRAC_CH2']
    phot_snr = 20
    
    
    # convert the detector noise to maggies
    hdu = fits.open('../input/noise_spec_12hr.fits')
    wave_obs = hdu[1].data['WAV']
    detector_noise = hdu[1].data['NOISE'] /(3631e-23)* wave_obs**2/(3e18) # erg/s/cm^2/A
    detector_noise_err = hdu[1].data['ERR'] /(3631e-23) * wave_obs**2/(3e18) # erg/s/cm^2/A

    ### Remove repeated wavelength values
    _, idx = np.unique(wave_obs, return_index=True)
    wave_obs = wave_obs[idx]
    detector_noise = detector_noise[idx]
    detector_noise_err = detector_noise_err[idx]

    ### define masks
    mask = np.ones(len(wave_obs),dtype=bool)
    phot_mask = np.ones(len(filters),dtype='bool')

    ### load up photometry
    obs = {'filters':  []}
    # --- remember to define redshift later! #
    for filt in filters:
        print('loading '+filt)
        obs['filters'] += observate.load_filters([filt])
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask

    ### load up spectrum
    if wave_obs is not None:
        obs['wavelength'] = wave_obs[mask]
        obs['mask'] = np.ones(mask.sum(),dtype=bool)
    else:
        obs['wavelength'] = None
        
        
        
    # load the noise free models     
    noise_free = pickle.load(open('noise_free_model.pickle', 'rb'))
    cat=pickle.load(open('../input/prospector_3dhst_bestfit_catalog.pickle', 'rb'))

    
    
    noisy_mock_all = []
    filecount = 0
    for i in range(63413):
        # Units of maggies (Janskies divided by 3631)
        spec = np.copy(noise_free['spec'][i]) 
        flux = np.copy(noise_free['phot'][i])
        sm = np.copy(noise_free['sm'][i])
        zred = cat['z'][i]
        obs['redshift'] = zred
        
        ### save mocks
        obs['maggies'] = flux
        obs['maggies_unc'] = flux/phot_snr
    
        ### calculate photon-counting noise
        #throughput = 0.225 # from Table 1 of SSP
        spec_snr = spec_maggie2shotnoise(wave_obs, spec, throughput = 0.225)
        spec_shot_noise = spec/spec_snr
        
        # consider uncertainty of the detector_noise
        detector_noise_random = np.random.normal(detector_noise, 
                                                 detector_noise_err, 
                                                 len(detector_noise))
        
        # need to add uncertainty to detector_noise
        full_noise = np.sqrt(np.square(spec_shot_noise) + np.square(detector_noise_random))
        full_snr = np.copy(spec)/full_noise
        
        ### rescale spectral uncertainty and sum in quadrature
        #detector_snr = (np.copy(spec)/detector_noise_random)
        #full_snr = 1./np.sqrt((1./spec_snr)**2 + (1./detector_snr)**2)

        obs['spectrum'] = np.copy(spec)
        obs['unc'] = np.copy(spec)/full_snr
            
        ### add errors if desired
        """
        add_err = False
        if add_err:
            np.random.seed(51)
            phot_errs = np.random.normal(scale=obs['maggies_unc'])
            obs['maggies'] += phot_errs
    
            np.random.seed(6)
            spec_errs = np.random.normal(scale=obs['unc'])
            obs['spectrum'] += spec_errs
        """
        # > use deepcopy otherwise older dictionary will be updated
        noisy_mock_all.append(copy.deepcopy(obs))
        
        if i%5000==0 and i!=0:
            print('saving output')
            filecount +=1
            with open('mockspec_noise_{0}.pickle'.format(filecount), 'wb') as handle:
                pickle.dump(noisy_mock_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
            noisy_mock_all = []




