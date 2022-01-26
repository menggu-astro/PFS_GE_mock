# import modules
import sys, os
import numpy as np
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from astropy.io import fits
from sedpy import observate
from prospect.models.sedmodel import PolySpecModel, gauss
from scipy import optimize
from prospect.sources import FastStepBasis
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from astropy.cosmology import WMAP9 as cosmo
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated

# dictionaries for photometry
legac_translation_dict = {'K': 'uvista_ks_cosmos',
                    'Y': 'uvista_y_cosmos',
                    'H': 'uvista_h_cosmos',
                    'J': 'uvista_j_cosmos',
                    'Y': 'uvista_y_cosmos',
                    'B': 'b_cosmos',
                    'V': 'v_cosmos',
                    'ch1': 'irac1_cosmos',
                    'ch2': 'irac2_cosmos',
                    'ch3': 'irac3_cosmos',
                    'ch4': 'irac4_cosmos',
                    'ip': 'ip_cosmos',
                    'r': 'r_cosmos',
                    'rp': 'rp_cosmos',
                    'u': 'u_cosmos',
                    'zp': 'zp_cosmos',
                    'gp': 'g_cosmos',
                    'IA484': 'ia484_cosmos',
                    'IA527': 'ia527_cosmos',
                    'IA624': 'ia624_cosmos',
                    'IA679': 'ia679_cosmos',
                    'IA738': 'ia738_cosmos',
                    'IA767': 'ia767_cosmos',
                    'IB427': 'ia427_cosmos',
                    'IB464': 'ia464_cosmos',
                    'IB505': 'ia505_cosmos',
                    'IB574': 'ia574_cosmos',
                    'IB709': 'ia709_cosmos',
                    'IB827': 'ia827_cosmos',
                    'fuv': 'galex_FUV',
                    'nuv': 'galex_NUV',
                    'mips24': 'mips24'}

def build_obs(objname='PFS_qu', add_err=True, zred=1.4,
              **kwargs):
    """Load photometry and spectra
    """

    # REDSHIFT
    # fixed for now
    obs = {'redshift': 1.4}

    # FILTERS
    # UgrizyJ, IRAC1, IRAC2
    filters = ['bessell_U']
    filters +=  ['sdss_'+filt+'0' for filt in ['g','r','i','z']]
    filters += ['uvista_y_cosmos','UDS_J']
    filters += ['IRAC_CH1','IRAC_CH2']
    phot_snr = 20

    ### SPECTRUM
    hdu = fits.open('../input/noise_spec_12hr.fits')
    wave_obs = hdu[1].data['WAV']
    detector_noise = hdu[1].data['ERR']/1e-23/3631 * (wave_obs**2/(3e18)) # erg/s/cm^2/A

    ### Remove repeated wavelength values
    print('REMOVING REPEATED WAVELENGTH VALUES')
    _, idx = np.unique(wave_obs, return_index=True)
    wave_obs = wave_obs[idx]
    detector_noise = detector_noise[idx]

    ### define masks
    mask = np.ones(len(wave_obs),dtype=bool)
    phot_mask = np.ones(len(filters),dtype='bool')

    ### load up photometry
    obs['filters'] = []
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

    ### generate data
    mod = build_model(objname=objname,afe_on=True)
    sps = build_sps()
    mod.params['nebemlineinspec'] = True # careful!
    spec, flux, _ = mod.predict(mod.initial_theta,sps=sps,obs=obs)
    print(mod.initial_theta)
    print(mod.theta_labels())

    ### save mocks
    obs['maggies'] = flux
    obs['maggies_unc'] = flux/phot_snr

    ### calculate photon-counting noise
    throughput = 0.225 # from Table 1 of SSP
    delta_lam = np.concatenate(((wave_obs[1:] - wave_obs[:-1]), np.atleast_1d(wave_obs[-1]-wave_obs[-2])))
    spectral_energy = spec * 3631*1e-23 # erg / s / cm^2 / Hz
    spectral_energy *= (3e18/wave_obs**2) # erg / s / cm^2 / A
    spectral_energy *= delta_lam # now it is erg/s/cm^2
    area = np.pi*(820)**2 # in cm^2
    spectral_energy *= area # now it is in erg/s
    photon_energy = 6.626e-27 * (3e18/wave_obs)
    nphot_per_second = spectral_energy / photon_energy * throughput
    nphot = nphot_per_second * (12 * 60 * 60)
    spec_snr = np.sqrt(nphot)

    ### rescale spectral uncertainty and sum in quadrature
    detector_snr = (spec/detector_noise)
    full_snr = 1./np.sqrt((1./spec_snr)**2 + (1./detector_snr)**2)

    if wave_obs is not None:
        obs['spectrum'] = spec
        obs['unc'] = spec/full_snr
    else:
        obs['spectrum'] = None

    ### add errors if desired
    if add_err:
        np.random.seed(51)
        phot_errs = np.random.normal(scale=obs['maggies_unc'])
        obs['maggies'] += phot_errs

        np.random.seed(6)
        spec_errs = np.random.normal(scale=obs['unc'])
        obs['spectrum'] += spec_errs

    # plot SED to ensure everything is on the same scale
    if False:
        import matplotlib.pyplot as plt
        smask = obs['mask']
        plt.plot(obs['wavelength'][smask],obs['spectrum'][smask],'-',lw=2,color='red')
        plt.plot(obs['wave_effective'],obs['maggies'],'o',color='black',ms=8)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(obs['spectrum'].min()*0.5,obs['spectrum'].max()*2)
        plt.xlim(obs['wavelength'][smask].min()*0.9,obs['wavelength'][smask].max()*1.1)
        plt.savefig(objname+'_spec.png',dpi=200)
        plt.close()

    return obs

# add redshift scaling to agebins, such that
# t_max = t_univ
nbins_sfh = 7
def zred_to_agebins(zred=None,agebins=None,**extras):
    #tuniv = cosmo.age(zred).value[0]*1e9
    tuniv = cosmo.age(zred).value.item()*1e9
    tbinmax = (tuniv*0.9)
    agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
    agebins = zred_to_agebins(zred=zred)
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()
    return m1 * coeffs

# --------------
# Model Definition
# --------------
def build_model(objname='PFS_qu', add_duste=True, add_neb=True, add_agn=True, mixture_model=False,
                remove_spec_continuum=True, switch_off_spec=False,
                marginalize_neb=True, spec_jitter=False, 
                afe_on=True, zred=1.4, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param object_redshift:
        If given, given the model redshift to this value.
    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """

    # input basic continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    # fit for redshift
    # use catalog value as center of the prior
    model_params["zred"]["init"] =  zred
    model_params["zred"]['isfree'] = True
    model_params["zred"]["prior"] = priors.TopHat(mini=0.0, maxi=3.0)

    # modify to increase nbins
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,0.3),
                                                                      df=np.full(nbins_sfh-1,2))

    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses

    # metallicity
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)

    # afe
    if afe_on:
        model_params['afe'] = {"N":1,
                               'isfree':True,
                               'init':0.4,
                               'prior': priors.TopHat(mini=-0.2,maxi=0.6)}
    else:
        model_params['afe'] = {"N":1,
                               'isfree':False,
                               'init':0.0,
                               'prior': priors.TopHat(mini=-0.2,maxi=0.6)}

    # complexify the dust
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1, 
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction*dust2

    model_params['dust1'] = {"N": 1, 
                             "isfree": False, 
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params['add_igm_absorption'] = {'N': 1,
                                      'isfree': False,
                                      'init': 1,
                                      'prior': None}

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)

    # Change the model parameter specifications based on some keyword arguments
    if (add_duste):
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_gamma']['isfree'] = False
        model_params['duste_qpah']['isfree'] = False
        model_params['duste_umin']['isfree'] = False
        model_params['duste_gamma']['init'] = 0.01
        model_params['duste_qpah']['init'] = 2.0

    if add_agn:
        # Allow for the presence of an AGN in the mid-infrared
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = False
        model_params['agn_tau']['isfree'] = False
        model_params['fagn']['prior'] = priors.LogUniform(mini=1e-5, maxi=3.0)
        model_params['agn_tau']['prior'] = priors.LogUniform(mini=5.0, maxi=150.)

    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])

        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        model_params['nebemlineinspec'] = {'N': 1,
                                           'isfree': False,
                                           'init': False}
        _ = model_params["gas_logz"].pop("depends_on")

        # Turn this on while fitting!
        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            #model_params.update(TemplateLibrary['fit_eline_redshift'])
            model_params['eline_prior_width']['init'] = 0.1
            model_params['use_eline_prior']['init'] = True
        else:
            model_params['nebemlineinspec']['init'] = True
    else:
        model_params['nebemlineinspec']['init'] = False

    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    if remove_spec_continuum:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params['spec_norm']['isfree'] = False
        model_params['spec_norm']['prior'] = priors.Normal(mean=1.0, sigma=0.3)

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    if mixture_model:
        model_params['f_outlier_spec'] = {"N": 1, 
                                          "isfree": True, 
                                          "init": 0.01,
                                          "prior": priors.TopHat(mini=1e-5, maxi=0.5)}
        model_params['nsigma_outlier_spec'] = {"N": 1, 
                                              "isfree": False, 
                                              "init": 50.0}
        model_params['f_outlier_phot'] = {"N": 1, 
                                          "isfree": False, 
                                          "init": 0.00,
                                          "prior": priors.TopHat(mini=0, maxi=0.5)}
        model_params['nsigma_outlier_phot'] = {"N": 1, 
                                              "isfree": False, 
                                              "init": 50.0}


    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a good fit.
    if spec_jitter:
        model_params['spec_jitter'] = {"N": 1, 
                                       "isfree": True, 
                                       "init": 1.0,
                                       "prior": priors.TopHat(mini=1.0, maxi=3.0)}

    # Now instantiate the model using this new dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)  # special to remove redshifting issue
    return sps

# -----------------
# Noise Model
# ------------------
def build_noise(spec_jitter=False,**extras):
    if spec_jitter:
        jitter = Uncorrelated(parnames = ['spec_jitter'])
        spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    else:
        spec_noise = None
    return spec_noise, None

# -----------
# Everything
# ------------
def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()    

    # - Add custom arguments -
    parser.add_argument('--add_neb', action="store_true",default=True,
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--remove_spec_continuum', action="store_true",default=True,
                        help="If set, fit continuum.")
    parser.add_argument('--add_duste', action="store_true", default=True,
                        help="If set, add dust emission to the model.")
    parser.add_argument('--add_agn', action="store_true", default=False,
                        help="If set, add agn emission to the model.")
    parser.add_argument('--objname', default='PFS_qu',
                        help="Name of the object to fit.")
    parser.add_argument('--switch_off_spec', action="store_true", default=False,
                        help="If set, remove spectrum from obs.")

    args = parser.parse_args()
    run_params = vars(args)

    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 500 
    run_params['nested_dlogz_init'] = 0.01
    run_params['nested_maxcall'] = 1500000
    run_params['nested_maxcall_init'] = 1500000
    run_params['nested_sample'] = 'rwalk'
    run_params['stop_kwargs'] = {'target_neff':1000}
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    run_params['objname'] = str(run_params['objname'])
    run_params['afe_on'] = True

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__
    print(model)

    if args.debug:
        sys.exit()

    ### XXX name of the output file; change this if desired!
    hfile = run_params['objname']+'_mcmc.h5'
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
