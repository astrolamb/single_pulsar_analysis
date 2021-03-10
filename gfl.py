import numpy as np
import pandas as pd


class gfl:
    """
        A class to open single pulsar chain files and run a 2D factorised
        likelihood analysis
    """

    # move psd to another function
    def __init__(self, Tspan, no_freq, datadir_dict=None, pkl=None):
        """
            pulsars init

            TO-DO: add a dict of Tspans for various datasets, remove Tspan
            kwarg, add pkl potential
        """
        self.Tspan = Tspan  # Tspan
        self.no_freq = no_freq  # number of sample frequencies
        # list of frequencies separated by 1/Tspan
        self.freqs = np.arange(1, no_freq+1)*(1/Tspan)

        self.pulsar_names = datadir_dict.keys()  # full list of pulsar names
        self.pulsar_list = self.pulsar_names  # entry for a sublist of psrs
        self.n_psrs = len(self.pulsar_names)   # number of psrs

        # outdir dict for all pulsars
        self.outdirs = datadir_dict

        # parameter labels for each pulsar
        metadata_pars = ['log-likelihood', 'unweighted log-posterior',
                         'MCMC acceptance rate',
                         'interchain transitions acceptance rate']

        self.pars = {psr:
                     np.append(np.loadtxt(self.outdirs[psr] + '/pars.txt',
                               dtype=np.unicode_), metadata_pars)
                     for psr in self.pulsar_names}

        # sample frequency labels
        self.rho_labels = ['gw_log10_rho_'+str(i) for i in range(self.no_freq)]

        # load chain data to dict for each psr. Cut-off burn-in for each psr
        print('loading chains...')
        self.chains = {psr: pd.read_csv(self.outdirs[psr] + '/chain_1.txt',
                       sep='\t', dtype=float, header=None,
                       names=self.pars[psr], error_bad_lines=False)
                       for psr in self.pulsar_names}

        self.burns = {psr: int(0.25*self.chains[psr].shape[0])
                      for psr in self.pulsar_names}

        # some constants
        self.consts = gfl_consts()

        return

    def histograms(self, no_rho_bins=20, rho_bin_min=-9, rho_bin_max=-4):
        """
            function to create a dict of histograms
            columns = pulsars
            rows = sample frequencies
        """
        # common set of bins across all psrs
        rho_bins = np.linspace(rho_bin_min, rho_bin_max, num=no_rho_bins)
        self.rho_bins = rho_bins
        self.rho_bin_min = rho_bin_min
        self.rho_bin_max = rho_bin_max

        # multiply histograms of each psr for each rho. Save to dict
        # corresponding to each rho
        epsilon = 1e-20
        self.hists = {psr:
                      {rho:
                       np.histogram(self.chains[psr][rho][self.burns[psr]:],
                                    rho_bins, density=True)[0]+epsilon
                       for rho in self.rho_labels}
                      for psr in self.pulsar_names}

        return

    def select_pulsars(self, pulsar_list):
        """
            method to select specific pulsars
        """
        self.pulsar_list = pulsar_list

        return

    def powerlaw(self, params):
        """
            powerlaw psd
        """

        gamma = params[0]
        log10_A = params[1]

        plaw = (10**(2.0 * log10_A)
                * (self.freqs * self.const.fyr)**(-gamma)
                * self.const.pl_unit_cor * self.fyr**3 * self.freqs[0])

        return 0.5 * np.log10(plaw)

    def broken_powerlaw(self, params, delta, log10_fb, kappa=0.1):
        """
        Generic broken powerlaw spectrum.
        :param f: sampling frequencies
        :param A: characteristic strain amplitude [set for gamma at f=1/yr]
        :param gamma: negative slope of PSD for f > f_break [set for comparison
            at f=1/yr (default 13/3)]
        :param delta: slope for frequencies < f_break
        :param log10_fb: log10 transition frequency at which slope switches
                         from gamma to delta
        :param kappa: smoothness of transition (Default = 0.1)
        """

        gamma = params[0]
        log10_A = params[1]

        hcf = (10**log10_A * (self.freqs * self.const.fyr)**((3 - gamma) / 2)
               * (1 + (self.freqs/10**log10_fb)**(1 / kappa))
               ** (kappa * (gamma - delta) / 2))

        return 0.5 * np.log10(hcf**2 * self.const.pl_unit_cor/self.freqs**3
                              * self.freqs[0])

    def log_likelihood(self, params, cp_psd=powerlaw, **psd_kwargs):
        """
            log likelihood function for mcmc sampler
        """
        logrho = cp_psd(psd_kwargs)

        if any(logrho < self.rho_bin_min) or any(logrho > self.rho_bin_max):
            logpdf_sum = -np.inf

        else:
            # calculate log_prob over freq
            idx = np.digitize(logrho, self.rho_bins)
            idx -= 1  # correct OBO error

            pdf = np.array([[self.hists[psr][self.rho_labels[ii]][idx[ii]]
                             for ii in range(self.no_freq)]
                            for psr in self.pulsar_list])

            logpdf = np.log(pdf)
            logpdf_sum = np.sum(logpdf)

        return logpdf_sum

    def log_prior(self, params):
        """
            log prior function for mcmc sampler
        """
        gamma = params[0]
        log10A = params[1]

        if 0. < gamma < 7. and -18. < log10A < -12.:
            return 0.0
        else:
            return -np.inf


class gfl_consts:
    """
        constants used in gfl class
    """

    def __init__(self):
        self.fyr = 365.25 / 86400.0
        self.pl_unit_cor = 1 / 12.0 / np.pi**2.0

        return
