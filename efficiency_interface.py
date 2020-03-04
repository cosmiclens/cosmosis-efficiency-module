from __future__ import print_function
from cosmosis.datablock import names, option_section

import numpy as np
from scipy.special import gammaincinv
from scipy.stats import gamma as gammadist
from scipy.integrate import solve_ivp
from astropy.cosmology import FlatLambdaCDM


def gammapdf(x, alpha, beta=1.):
    '''Compute the gamma distribution PDF with parameters alpha, beta.'''
    return gammadist.pdf(x, np.add(alpha, 1), scale=np.divide(1, beta))

def setup(options):
    nbin = options.get_int(option_section, 'nbin')
    nz = options.get_int(option_section, 'nz')
    in_section = options.get_string(option_section, 'input_section', default=names.number_density_params)
    out_section = options.get_string(option_section, 'output_section', default=names.wl_number_density)

    p = np.linspace(0, 1, nz, endpoint=False)

    # fiducial cosmology
    Om0 = options.get_double(option_section, 'Om0', default=-1)
    H0 = options.get_double(option_section, 'H0', default=-1)

    print('using {} tomographic bins'.format(nbin))
    print('using {} redshift bins'.format(nz))
    print('fixed cosmology: Om0 = {}, H0 = {}'.format(Om0, H0))

    return nbin, p, Om0, H0, in_section, out_section

def execute(block, config):
    nbin, p, Om0, H0, params, nz_section = config

    if Om0 < 0:
        Om0 = block['cosmological_parameters', 'omega_m']
    if H0 < 0:
        H0 = block['cosmological_parameters', 'hubble']

    alphabet = []
    x = []

    for i in range(nbin):
        keys = ['alpha', 'beta', 'mu', 'eta']
        vals = []

        for k in keys:
            name = '{}_{}'.format(k, i+1)
            if block.has_value(params, name):
                v = block.get_double(params, name)
            elif block.has_value(params, 'inv_' + name):
                v = 1/block.get_double(params, 'inv_' + name)
            elif block.has_value(params, 'log_' + name):
                v = np.power(10., block.get_double(params, 'log_' + name))
            else:
                v = np.nan
            vals.append(v)

        alpha, beta, mu, eta = vals

        if np.isnan(alpha):
            if not np.isnan(mu):
                if not np.isnan(beta):
                    alpha = mu*beta - 1
                elif not np.isnan(eta):
                    alpha = 1/(mu*eta - 1)
            elif not np.isnan(eta):
                if not np.isnan(beta):
                    alpha = beta/eta
        if np.isnan(beta):
            if not np.isnan(mu):
                if not np.isnan(alpha):
                    beta = (alpha+1)/mu
                elif not np.isnan(eta):
                    beta = eta/(mu*eta - 1)
            elif not np.isnan(eta):
                if not np.isnan(alpha):
                    beta = alpha*eta

        if alpha < 0 or beta < 0:
            return 1

        alphabet.append((alpha, beta))
        x = np.union1d(x, gammaincinv(alpha+1, p)/beta)

    cosmo = FlatLambdaCDM(Om0=Om0, H0=H0)

    dH = cosmo.hubble_distance.to('Mpc')

    sol = solve_ivp(lambda x, z: cosmo.efunc(z)/dH, (0, x[-1]), y0=[0], t_eval=x)

    if not sol.success:
        print('failed to invert comoving distance:', sol.message)
        print('requested x range:', x[0], 'to', x[-1])
        print('obtained x range:', sol.t[0], 'to', sol.t[-1])
        print('obtained z range:', sol.y[0, 0], 'to', sol.y[0, -1])
        return 2

    x, z  = sol.t, sol.y[0]
    dx_dz = dH.value*cosmo.inv_efunc(z)

    block[nz_section, 'nbin'] = nbin
    block[nz_section, 'nz'] = len(z)
    block[nz_section, 'z'] = z

    for i, (alpha, beta) in enumerate(alphabet):
        nz = gammapdf(x, alpha, beta)*dx_dz
        block[nz_section, 'bin_{}'.format(i+1)] = nz

    return 0
