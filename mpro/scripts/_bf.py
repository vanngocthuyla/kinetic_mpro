"""
https://doi.org/10.1371/journal.pone.0273656
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import iqr

from sklearn.mixture import GaussianMixture
import pymbar

def log_marginal_likelihood(log_likelihoods):
    """
    :param log_likelihoods: 1d array
    :return: log_marg_llh, float, log of marginal likelihood
    """
    n = len(log_likelihoods)
    log_weights = -log_likelihoods
    a_const = log_weights.max()
    log_weights = log_weights - a_const
    total_weight = np.sum(np.exp(log_weights))

    log_marg_llh = np.log(n) - a_const - np.log(total_weight)
    return log_marg_llh


def log_marginal_lhs_bootstrap(extracted_loglhs, sample_size=None, bootstrap_repeats=1):
    """
    :param extracted_loglhs: 1d array
    :param sample_size: int
    :param bootstrap_repeats: int
    :return: all_sample_estimate, bootstrap_samples
    """
    if sample_size is None:
        sample_size = len(extracted_loglhs)
    all_sample_estimate = log_marginal_likelihood(extracted_loglhs)

    bootstrap_samples = []
    for _ in range(bootstrap_repeats):
        drawn_loglhs = np.random.choice(extracted_loglhs, size=sample_size, replace=True)

        bootstrap_samples.append(log_marginal_likelihood(drawn_loglhs))

    bootstrap_samples = np.array(bootstrap_samples)

    return all_sample_estimate, bootstrap_samples


def dict_to_list(dict_of_list):
    """
    :param dict_of_list: dict: varname --> ndarray
    :return: list of dic: [ {varname: float, ...}, ...  ]
    """
    keys = list(dict_of_list.keys())
    key0 = keys[0]
    for key in keys[1:]:
        assert len(dict_of_list[key0]) == len(dict_of_list[key]), key0 + " and " + key + " do not have same len."

    n = len(dict_of_list[key0])
    ls_of_dic = []
    for i in range(n):
        dic = {key: dict_of_list[key][i] for key in keys}
        ls_of_dic.append(dic)
    return ls_of_dic


def get_values_from_trace(model, trace, thin=1, burn=0):
    """
    :param model: pymc3 model
    :param trace: pymc3 trace object
    :param thin: int
    :param burn: int, number of steps to exclude
    :return: dict: varname --> ndarray
    """
    varnames = [var.name for var in model.vars]

    if isinstance(trace, dict):
        trace_values = {var: trace[var][burn::thin] for var in varnames}
        return trace_values

    trace_values = {var: trace.get_values(var, thin=thin, burn=burn) for var in varnames}
    return trace_values


def get_values_from_traces(model, traces, thin=1, burn=0):
    trace_value_list = [get_values_from_trace(model, trace, thin=thin, burn=burn) for trace in traces]
    keys = trace_value_list[0].keys()
    trace_values = {}
    for key in keys:
        trace_values[key] = np.concatenate([tr_val[key] for tr_val in trace_value_list])
    return trace_values


def std_from_iqr(data):
    return iqr(data) / 1.35


def fit_normal(x, sigma_robust=False):
    # mu, sigma = norm.fit(x[~np.isnan(x)])
    mu, sigma = norm.fit(x[np.isfinite(x)])
    if sigma_robust:
        sigma = std_from_iqr(x)
    res = {"mu": mu, "sigma": sigma}
    return res


def fit_normal_trace(trace_values, sigma_robust=False):
    res = {varname: fit_normal(trace_values[varname], sigma_robust=sigma_robust) for varname in trace_values}
    return res


def draw_normal_samples(mu_sigma_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = mu_sigma_dict.keys()
    samples = {k: rand.normal(loc=mu_sigma_dict[k]["mu"], scale=mu_sigma_dict[k]["sigma"], size=nsamples)
               for k in keys}
    return samples


def log_normal_pdf(mu, sigma, y):
    sigma2 = sigma * sigma
    res = - 0.5 * np.log(2 * np.pi * sigma2) - (0.5 / sigma2) * (y - mu) ** 2
    return res


def log_normal_trace(trace_val, mu_sigma_dict):
    keys = list(trace_val.keys())
    if len(keys) == 0:
        return 0.

    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        mu = mu_sigma_dict[k]["mu"]
        sigma = mu_sigma_dict[k]["sigma"]
        y = trace_val[k]
        logp += log_normal_pdf(mu, sigma, y)

    return logp


def fit_uniform(x, d=1e-100):
    lower = x.min() - d
    upper = x.max() + d
    res = {"lower": lower, "upper": upper}
    return res


def fit_uniform_trace(trace_values):
    res = {varname: fit_uniform(trace_values[varname]) for varname in trace_values}
    return res


def draw_uniform_samples(lower_upper_dict, nsamples, random_state=None):
    rand = np.random.RandomState(random_state)
    keys = lower_upper_dict.keys()
    samples = {k: rand.uniform(low=lower_upper_dict[k]["lower"],
                               high=lower_upper_dict[k]["upper"],
                               size=nsamples)
               for k in keys}
    return samples


def log_uniform_pdf(lower, upper, y):
    logp = np.zeros_like(y)
    logp[:] = -np.inf
    logp[(y > lower) & (y < upper)] = - np.log(upper - lower)
    return logp


def log_uniform_trace(trace_val, lower_upper_dict):
    keys = list(trace_val.keys())
    k0 = keys[0]
    for k in keys[1:]:
        assert len(trace_val[k0]) == len(trace_val[k]), k0 + " and " + k + " do not have same len."

    nsamples = len(trace_val[k0])
    logp = np.zeros(nsamples, dtype=float)
    for k in keys:
        lower = lower_upper_dict[k]["lower"]
        upper = lower_upper_dict[k]["upper"]
        y = trace_val[k]
        logp += log_uniform_pdf(lower, upper, y)

    return logp


class GaussMix(object):
    def __init__(self, n_components, covariance_type="full"):
        """
        :param n_components: int
        :param covariance_type: str, one of 'full', 'tied', 'diag', 'spherical'
        read here for explanation:
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        """
        self._n_components = n_components
        self._vars = []
        self._gm = GaussianMixture(n_components=self._n_components, covariance_type=covariance_type)

    def fit(self, sample_dict):
        """
        :param sample_dict: dict, var --> 1d array
        """
        self._vars = list(sample_dict.keys())
        X_train = self._dict_to_array(sample_dict)
        self._gm.fit(X_train)
        return self

    def score_samples(self, sample_dict):
        """return logp"""
        X = self._dict_to_array(sample_dict)
        logp = self._gm.score_samples(X)
        return logp

    def sample(self, n_samples=1):
        X = self._gm.sample(n_samples=n_samples)
        X = X[0]
        X_dict = {}
        for i, v in enumerate(self._vars):
            X_dict[v] = X[:, i]
        return X_dict

    def get_vars(self):
        return self._vars

    def get_model(self):
        return self._gm

    def get_gm_fited_params(self):
        weights = self._gm.weights_
        means = self._gm.means_
        covariances = self._gm.covariances_

        results = {"weights": weights, "means": means, "covariances": covariances}

        return results

    def get_n_components(self):
        return self._n_components

    def _dict_to_array(self, sample_dict):
        X = [sample_dict[v] for v in self._vars]
        X = np.stack(X, axis=1)
        return X


def log_posterior_trace(model, trace_values):
    """
    https://www.pymc.io/projects/docs/en/v3/developer_guide.html#:~:text=PyMC3%20expects%20the%20logp(),then%20used%20for%20fitting%20models
    """
    if "logp" in trace_values:
        print("_bayes_factor.log_posterior_trace: Use precomputed logp in trace_values")
        return trace_values["logp"]

    model_vars = model.vars
    trace_vars = trace_values.keys()

    trace_v = {}
    for var in model_vars:
        if var not in trace_vars:
            raise KeyError(var + " is not in trace")
        trace_v[var] = trace_values[var]

    logp = model.logp(trace_v)
    # trace_v = dict_to_list(trace_v)
    # get_logp = np.vectorize(model.logp)
    # logp = get_logp(trace_v)
    return logp


def pot_ener(sample, model):
    u = -log_posterior_trace(model, sample)
    return u


def pot_ener_normal_aug(sample, model, sample_aug, mu_sigma):
    u1 = -log_posterior_trace(model, sample)
    u2 = -log_normal_trace(sample_aug, mu_sigma)
    u = u1 + u2
    return u


def pot_ener_gauss_mix_aug(sample, model, sample_aug, gm_model):
    u1 = -log_posterior_trace(model, sample)
    u2 = - gm_model.score_samples(sample_aug)
    u = u1 + u2
    return u


def pot_ener_uniform_aug(sample, model, sample_aug, lower_upper):
    u1 = -log_posterior_trace(model, sample)
    u2 = -log_uniform_trace(sample_aug, lower_upper)
    u = u1 + u2
    return u


def bootstrap_BAR(w_F, w_R, repeats, sample_proportion):
    """
    :param w_F: ndarray
    :param w_R: ndarray
    :param repeats: int
    :return: std, float
    """
    assert 0 <= sample_proportion <= 1, "sample_proportion out of range"

    n_F = int(len(w_F) * sample_proportion)
    n_R = int(len(w_R) * sample_proportion)

    delta_Fs = []
    for _ in range(repeats):
        w_F_rand = np.random.choice(w_F, size=n_F, replace=True)
        w_R_rand = np.random.choice(w_R, size=n_R, replace=True)

        df = pymbar.BAR(w_F_rand, w_R_rand, compute_uncertainty=False, relative_tolerance=1e-6, verbose=False)
        delta_Fs.append(df)

    delta_Fs = np.asarray(delta_Fs)
    delta_Fs = delta_Fs[~np.isnan(delta_Fs)]
    delta_Fs = delta_Fs[~np.isinf(delta_Fs)]

    df_mean = delta_Fs.mean()
    df_std = delta_Fs.std()

    return df_mean, df_std


def is_var_starts_with(probe_str, str_to_be_probed):
    if str_to_be_probed.endswith("__"):
        suffix = str_to_be_probed.split("__")[0].split("_")[-1]
        suffix = "_" + suffix + "__"
        base_var_name = str_to_be_probed.split(suffix)[0]
        return probe_str == base_var_name
    else:
        return probe_str == str_to_be_probed


def var_starts_with(start_str, list_of_strs):
    """
    :param start_str: str
    :param list_of_strs: list
    :return: str
    """
    found = [s for s in list_of_strs if is_var_starts_with(start_str, s)]
    if len(found) == 0:
        raise ValueError("Found none")
    if len(found) > 1:
        raise ValueError("Found many: " + ", ".join(found))
    return found[0]


def filter_nan_inf(x):
    x = x[x != np.inf]
    x = x[x != -np.inf]
    x = x[~np.isnan(x)]
    return x


def filter_high(x, percentile=97.5):
    thres = np.percentile(x, percentile)
    return x[x < thres]


def filter_low(x, percentile=2.5):
    thres = np.percentile(x, percentile)
    return x[x > thres]


def bayes_factor_v1(model_ini, sample_ini, model_fin, sample_fin,
                    aug_with="Normal", sigma_robust=False,
                    n_components=1, covariance_type="full",
                    bootstrap=None, sample_proportion=None):
    """
    :param model_ini: mcmc model
    :param sample_ini: dict: var_name -> ndarray
    :param model_fin: mcmc model
    :param sample_fin: dict: var_name -> ndarray
    :param aug_with: str
    :param sigma_robust: bool, only used when aug_with="Normal"
    :param n_components: int, only used when aug_with="GaussMix"
    :param covariance_type: str, only used when aug_with="GaussMix"
    :param bootstrap: int
    :return: bf if bootstrap is None
             (bf, err) if bootstrap is an int
    """
    print("Use estimator version 1")

    if (sample_proportion is not None) and bootstrap is None:
        raise ValueError("When sample_proportion = %0.5f, bootstrap must not be None" % sample_proportion)

    assert aug_with in ["Normal", "Uniform", "GaussMix"], "Unknown aug_with: " + aug_with
    print("aug_with:", aug_with)

    if aug_with == "Normal":
        mu_sigma_fin = fit_normal_trace(sample_fin, sigma_robust=sigma_robust)
    elif aug_with == "Uniform":
        lower_upper_fin = fit_uniform_trace(sample_fin)
    elif aug_with == "GaussMix":
        print("n_components:", n_components)
        print("covariance_type:", covariance_type)
        gauss_mix = GaussMix(n_components=n_components, covariance_type=covariance_type)
    else:
        pass

    vars_ini = [var for var in sample_ini.keys() if var != "logp"]
    print("vars_ini:", vars_ini)
    vars_fin = [var for var in sample_fin.keys() if var != "logp"]
    print("vars_fin:", vars_fin)

    vars_redundant = [var for var in vars_ini if not var in vars_fin] + [var for var in vars_fin if not var in vars_ini]
    print("vars_redundant:", vars_redundant)

    if aug_with == "Normal":
        mu_sigma_fin = {var: mu_sigma_fin[var] for var in vars_redundant}
    elif aug_with == "Uniform":
        lower_upper_fin = {var: lower_upper_fin[var] for var in vars_redundant}
    elif aug_with == "GaussMix":
        sample_redun = {var: sample_fin[var] for var in vars_redundant}
        gauss_mix.fit(sample_redun)
    else:
        pass

    nsamples_ini = len(sample_ini[vars_ini[0]])
    print("nsamples_ini = %d" % nsamples_ini)
    nsamples_fin = len(sample_fin[vars_fin[0]])
    print("nsamples_fin = %d" % nsamples_fin)

    # potential for sample drawn from i estimated at state i
    if aug_with == "Normal":
        sample_aug_ini = draw_normal_samples(mu_sigma_fin, nsamples_ini)
        u_i_i = pot_ener_normal_aug(sample_ini, model_ini, sample_aug_ini, mu_sigma_fin)
    elif aug_with == "Uniform":
        sample_aug_ini = draw_uniform_samples(lower_upper_fin, nsamples_ini)
        u_i_i = pot_ener_uniform_aug(sample_ini, model_ini, sample_aug_ini, lower_upper_fin)
    elif aug_with == "GaussMix":
        sample_aug_ini = gauss_mix.sample(n_samples=nsamples_ini)
        u_i_i = pot_ener_gauss_mix_aug(sample_ini, model_ini, sample_aug_ini, gauss_mix)
    else:
        pass

    ini_final_var_match = [(var, var) for var in vars_ini if var in vars_fin]
    # potential for sample drawn from i estimated at state f
    sample_ini_comb = {}
    for ki, kf in ini_final_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_ini_comb[var_fin] = sample_ini[var_ini]
    sample_ini_comb.update(sample_aug_ini)
    u_i_f = pot_ener(sample_ini_comb, model_fin)

    # potential for sample drawn from f estimated at state f
    u_f_f = pot_ener(sample_fin, model_fin)

    # potential for sample drawn from f estimated at state i
    sample_fin_split = {}
    for ki, kf in ini_final_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_fin_split[var_ini] = sample_fin[var_fin]

    sample_redun_fin = {var: sample_fin[var] for var in vars_redundant}
    if aug_with == "Normal":
        u_f_i = pot_ener_normal_aug(sample_fin_split, model_ini, sample_redun_fin, mu_sigma_fin)
    elif aug_with == "Uniform":
        u_f_i = pot_ener_uniform_aug(sample_fin_split, model_ini, sample_redun_fin, lower_upper_fin)
    elif aug_with == "GaussMix":
        u_f_i = pot_ener_gauss_mix_aug(sample_fin_split, model_ini, sample_redun_fin, gauss_mix)
    else:
        pass

    w_F = u_i_f - u_i_i
    w_R = u_f_i - u_f_f

    w_F = filter_nan_inf(w_F)
    w_R = filter_nan_inf(w_R)

    if (len(w_F) == 0) or (len(w_R) == 0):
        print("Empty work arrays:", w_F.shape, w_R.shape)
        if bootstrap is None:
            return 0.
        else:
            return 0., 0.

    if sample_proportion is None:
        delta_F = pymbar.bar(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
        bf = -delta_F['Delta_f'] #ln_bf

        if bootstrap is None:
            print("log10(bf) = %0.5f" % (bf * np.log10(np.e)))
            return bf, None
        else:
            print("Running %d bootstraps to estimate error." % bootstrap)
            _, bf_err = bootstrap_BAR(w_F, w_R, bootstrap, sample_proportion=1.)
            print("log10(bf) = %0.5f +/- %0.5f" % (bf*np.log10(np.e), bf_err*np.log10(np.e)))
            return bf, bf_err

    else:
        bf, bf_err = bootstrap_BAR(w_F, w_R, bootstrap, sample_proportion=sample_proportion)
        return bf, bf_err


def bayes_factor_v2(model_ini, sample_ini, model_fin, sample_fin,
                    aug_with="GaussMix", sigma_robust=False,
                    n_components=1, covariance_type="full",
                    bootstrap=None, sample_proportion=None):
    """
    :param model_ini: mcmc model
    :param sample_ini: dict: var_name -> ndarray
    :param model_fin: mcmc model
    :param sample_fin: dict: var_name -> ndarray
    :param aug_with: str
    :param sigma_robust: bool, only used when aug_with="Normal"
    :param n_components: int, only used when aug_with="GaussMix"
    :param covariance_type: str, only used when aug_with="GaussMix"
    :param bootstrap: int
    :return: bf if bootstrap is None
             (bf, err) if bootstrap is an int
    """
    print("Use estimator version 2")

    if (sample_proportion is not None) and bootstrap is None:
        raise ValueError("When sample_proportion = %0.5f, bootstrap must not be None" % sample_proportion)

    assert aug_with in ["Normal", "Uniform", "GaussMix"], "Unknown aug_with: " + aug_with
    print("aug_with:", aug_with)

    vars_ini = [var for var in sample_ini.keys() if var != "logp"]
    print("vars_ini:", vars_ini)
    nsamples_ini = len(sample_ini[vars_ini[0]])
    print("nsamples_ini = %d" % nsamples_ini)

    vars_fin = [var for var in sample_fin.keys() if var != "logp"]
    print("vars_fin:", vars_fin)
    nsamples_fin = len(sample_fin[vars_fin[0]])
    print("nsamples_fin = %d" % nsamples_fin)

    # get redundant parameters from final state
    vars_redundant = [var for var in vars_ini if not var in vars_fin] + [var for var in vars_fin if not var in vars_ini]
    sample_redun_fin = {var: sample_fin[var] for var in vars_redundant}
    print("Vars of sample_redun_fin:", vars_redundant)

    # fit models to sample_redun_fin
    if aug_with == "Normal":
        mu_sigma_fin = fit_normal_trace(sample_redun_fin, sigma_robust=sigma_robust)
        sample_aug_ini = draw_normal_samples(mu_sigma_fin, nsamples_ini)
    elif aug_with == "Uniform":
        lower_upper_fin = fit_uniform_trace(sample_redun_fin)
        sample_aug_ini = draw_uniform_samples(lower_upper_fin, nsamples_ini)
    elif aug_with == "GaussMix":
        print("n_components:", n_components)
        print("covariance_type:", covariance_type)
        gauss_mix = GaussMix(n_components=n_components, covariance_type=covariance_type)
        gauss_mix.fit(sample_redun_fin)
        sample_aug_ini = gauss_mix.sample(n_samples=nsamples_ini)
    else:
        pass
    print("Vars of sample_aug_ini:", list(sample_aug_ini.keys()))

    # potential for sample drawn from i estimated at state i
    if aug_with == "Normal":
        u_i_i = pot_ener_normal_aug(sample_ini, model_ini, sample_aug_ini, mu_sigma_fin)
    elif aug_with == "Uniform":
        u_i_i = pot_ener_uniform_aug(sample_ini, model_ini, sample_aug_ini, lower_upper_fin)
    elif aug_with == "GaussMix":
        u_i_i = pot_ener_gauss_mix_aug(sample_ini, model_ini, sample_aug_ini, gauss_mix)
    else:
        pass

    ini_fin_var_match = [(var, var) for var in vars_ini if var in vars_fin]

    # potential for sample drawn from i estimated at state f
    sample_tmp_ini = {}
    for ki, kf in ini_fin_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_tmp_ini[var_fin] = sample_ini[var_ini]
    for var in vars_redundant:
        sample_tmp_ini[var] = sample_aug_ini[var]
    u_i_f = pot_ener(sample_tmp_ini, model_fin)
    del sample_tmp_ini

    # potential for sample drawn from f estimated at state i
    sample_tmp_fin = {}
    for ki, kf in ini_fin_var_match:
        var_ini = var_starts_with(ki, vars_ini)
        var_fin = var_starts_with(kf, vars_fin)
        sample_tmp_fin[var_ini] = sample_fin[var_fin]

    if aug_with == "Normal":
        u_f_i = pot_ener_normal_aug(sample_tmp_fin, model_ini, sample_redun_fin, mu_sigma_fin)
    elif aug_with == "Uniform":
        u_f_i = pot_ener_uniform_aug(sample_tmp_fin, model_ini, sample_redun_fin, lower_upper_fin)
    elif aug_with == "GaussMix":
        u_f_i = pot_ener_gauss_mix_aug(sample_tmp_fin, model_ini, sample_redun_fin, gauss_mix)
    else:
        pass
    del sample_tmp_fin

    # potential for sample drawn from f estimated at state f
    u_f_f = pot_ener(sample_fin, model_fin)

    w_F = u_i_f - u_i_i
    w_R = u_f_i - u_f_f

    w_F = filter_nan_inf(w_F)
    w_R = filter_nan_inf(w_R)

    if (len(w_F) == 0) or (len(w_R) == 0):
        print("Empty work arrays:", w_F.shape, w_R.shape)
        if bootstrap is None:
            return 0.
        else:
            return 0., 0.

    if sample_proportion is None:
        delta_F = pymbar.bar(w_F, w_R, compute_uncertainty=False, relative_tolerance=1e-12, verbose=True)
        bf = -delta_F['Delta_f']

        if bootstrap is None:
            print("log10(bf) = %0.5f" % (bf * np.log10(np.e))) #np.log10(np.exp(bf))
            return bf, None
        else:
            print("Running %d bootstraps to estimate error." % bootstrap)
            _, bf_err = bootstrap_BAR(w_F, w_R, bootstrap, sample_proportion=1.)
            print("log10(bf) = %0.5f +/- %0.5f" % (bf * np.log10(np.e), bf_err * np.log10(np.e)))
            return bf, bf_err

    else:
        bf, bf_err = bootstrap_BAR(w_F, w_R, bootstrap, sample_proportion=sample_proportion)
        return bf, bf_err