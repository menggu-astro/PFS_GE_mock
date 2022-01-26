# The code is used to generate noise free models
# with Prospector (alpha version)
# Parameters came from 
# `prospector_3dhst_bestfit_catalog.pickle`.
# There are 63413 objects.

import pickle, sys, os
import numpy as np
import pandas as pd
import prospect
import pfs_spectra_params 

from astropy.io import fits
from sedpy import observate

# ---- scaling relations ---- #
"""
import pandas as pd
import sklearn
from sklearn import linear_model

Choi2014 = {}
Choi2014['z'] = np.array([0.08,]*4+ [0.15,]*5 + [0.25,]*5 + [0.35,]*4 + 
                         [(0.4+0.55)/2,]*3 + [(0.55+0.7)/2,]*3 + [0.83, 0.83])
Choi2014['logmstar'] = np.array([10.6, 10.9, 11.2, 11.5, ] + [9.9, 10.2, 10.4, 10.7, 11.0,] 
                                 +[10.2, 10.5, 10.7, 11.0, 11.3,] + [10.5, 10.8, 11.1, 11.3,]
                                 +[10.8,11.1,11.3]+[10.9, 11.0, 11.3] + [10.9, 11.1])
Choi2014['MgFe'] = np.array([0.18, 0.21, 0.25, 0.31]+[0.04, 0.13, 0.16, 0.21, 0.23]+
                            [0.17, 0.19, 0.20, 0.22, 0.23]+[0.18, 0.22,0.21, 0.29]+
                            [0.23, 0.23, 0.30]+[0.05, 0.09, 0.19]+[0.31, 0.35])
Choi2014['MgFe_err'] = np.array([0,0,0,0.01]+[0.05, 0.03, 0.01, 0.01,0.02]+[0.06,0.02,0.01,0.02,0.04]
                                +[0.04,0.02,0.01,0.03]+[0.04,0.02,0.03]+[0.13,0.05,0.04]+[0.07,0.08])

Choi2014_df = pd.DataFrame.from_dict(Choi2014)

X = Choi2014_df[['logmstar']]
y = Choi2014_df['MgFe']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_, regr.intercept_)
"""

def Choi_2014(logmstar):
    return logmstar*0.11881615 - 1.0826891415131317

def Matthee_2018(logmstar, logssfr):
    return -0.282*(logmstar-10.0) + 0.290*(logssfr+logmstar) + 0.163

def est_afe(logmstar, logssfr):
    if logssfr > -10.5:
        return Matthee_2018(logmstar, logssfr)
    else:
        return Choi_2014(logmstar)
est_afe = np.vectorize(est_afe)
# -------------------------------- #

if __name__=='__main__':
    # ---- load 3dhst catalog ---- #
    cat=pickle.load(open('../input/prospector_3dhst_bestfit_catalog.pickle', 'rb'))
    print('display first 5 objects')
    for ikey in cat.keys():
        print(ikey, cat[ikey][:5])

    mod = pfs_spectra_params.build_model(objname='test',afe_on=True)
    mod.params['nebemlineinspec'] = True
    sps = pfs_spectra_params.build_sps()
    print('finish building sps')
    # FILTERS
    # UgrizyJ, IRAC1, IRAC2
    filters = ['bessell_U']
    filters +=  ['sdss_'+filt+'0' for filt in ['g','r','i','z']]
    filters += ['uvista_y_cosmos','UDS_J']
    filters += ['IRAC_CH1','IRAC_CH2']
    ### SPECTRUM
    hdu = fits.open('../input/noise_spec_12hr.fits')
    wave_obs = hdu[1].data['WAV']
    #detector_noise = hdu[1].data['ERR']/1e-23/3631 * wave_obs**2/(3e18) # erg/s/cm^2/A

    ### Remove repeated wavelength values
    print('REMOVING REPEATED WAVELENGTH VALUES')
    _, idx = np.unique(wave_obs, return_index=True)
    wave_obs = wave_obs[idx]
    #detector_noise = detector_noise[idx]

    ### define masks
    mask = np.ones(len(wave_obs),dtype=bool)
    phot_mask = np.ones(len(filters),dtype='bool')


    obs = {'filters': []}
    for filt in filters:
        #print('loading '+filt)
        obs['filters'] += observate.load_filters([filt])
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    if wave_obs is not None:
        obs['wavelength'] = wave_obs[mask]
        obs['mask'] = np.ones(mask.sum(),dtype=bool)
    else:
        obs['wavelength'] = None
    

    """
    theta_all_dict = {}
    for iparams in ['logzsol', 'dust2', 'logsfr_ratios_1', 
                    'logsfr_ratios_2', 'logsfr_ratios_3', 
                    'logsfr_ratios_4', 'logsfr_ratios_5', 
                    'logsfr_ratios_6', 'logmass']:
        theta_all_dict[iparams] = np.copy(cat[iparams])
    theta_all_dict['zred'] = np.copy(cat['z'])
    
    # cat['log_ssfr'] is actually just ssfr, not logarithmic ssfr
    theta_all_dict['afe'] = est_afe(cat['log_stellarmass'], np.log10(cat['log_ssfr']))
    theta_all_dict['dust_index'] = np.copy(cat['dust_index'])
    theta_all_dict['dust1_fraction'] = np.copy(cat['dust1_fraction'])
    
    # based on the LEGA-C mass-FJ relation
    # Eq1 in https://arxiv.org/pdf/1811.07900.pdf says log σ =(−0.85 ± 0.11) + (0.29 ± 0.01) log M
    theta_all_dict['sigma_smooth'] = 10**(0.29 * np.copy(cat['log_stellarmass']) - 0.85)
    
    # fix gas_logz to stellar metallicity
    theta_all_dict['gas_logz'] = np.copy(cat['logzsol'])
    theta_all_dict['gas_logu'] = np.zeros(cat['z'].shape)-2
    theta_all_dict['eline_sigma'] = np.copy(theta_all_dict['sigma_smooth'])
    
    theta_all_df = pd.DataFrame.from_dict(theta_all_dict)
    theta_all_df = theta_all_df.reindex(columns=mod.theta_labels())
    theta_all_df.to_csv('theta_all.csv')
    """
    
    theta_all_df = pd.read_csv('theta_all.csv')  
    theta_all_df = theta_all_df.reindex(columns=mod.theta_labels())

    noise_free_output = {}
    #noise_free_output['name'] = []
    noise_free_output['spec'] = np.zeros(shape=(63413, 12257)) + np.nan
    noise_free_output['phot'] = np.zeros(shape=(63413, 9)) + np.nan
    noise_free_output['sm'] = np.zeros(63413) + np.nan

    for i in range(63413):
        if i%1000==0:
            print(i, end=',')
        obs['redshift'] = float(theta_all_df['zred'][i])
        
        spec, photometric_fluxes, sm = mod.predict(theta =list(theta_all_df.loc[i].to_numpy()),
                                                   obs = obs,
                                                   sps = sps)
        #noise_free_output['name'].append(cat['name'][i])
        noise_free_output['spec'][i] = spec
        noise_free_output['phot'][i] = photometric_fluxes
        noise_free_output['sm'][i] = sm
    
    with open('noise_free_model.pickle', 'wb') as handle:
        pickle.dump(noise_free_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
      





      

