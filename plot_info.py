# Plot Banzhaf distribution by n

from kernelprob.utils.p_generator import get_p

import matplotlib.pyplot as plt
import numpy as np

beta_shap_sym = {
    "shapley" : "Shapley",
    "beta_shapley_2_2" : "Beta(2,2)",
    "beta_shapley_4_4" : "Beta(4,4)",
    "beta_shapley_8_8" : "Beta(8,8)",
    "beta_shapley_16_16" : "Beta(16,16)",
}

beta_shap_asym = {
    "shapley" : "Shapley",
    "beta_shapley_1_2" : "Beta(1,2)",
    "beta_shapley_1_4" : "Beta(1,4)",
    "beta_shapley_1_8" : "Beta(1,8)",
    "beta_shapley_1_16" : "Beta(1,16)",
}


weighted_banzhaf = {
    "banzhaf" : "Banzhaf",
    "weighted_banzhaf_0.6" : "WB(0.6)",
    "weighted_banzhaf_0.7" : "WB(0.7)",
    "weighted_banzhaf_0.8" : "WB(0.8)",
    "weighted_banzhaf_0.9" : "WB(0.9)",
}


weightings = {
    "Symmetric Beta Shapley" : beta_shap_sym,
    "Asymmetric Beta Shapley" : beta_shap_asym,
    "Weighted Banzhaf" : weighted_banzhaf,
}

linestyles = ['-', '--', '-.', ':', (5, (10, 3))]
cb_colors = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
]

def plot_probabilistic_values(n, figsize):
    # 3 plots in a row
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for plot_idx, (weightings_name, weighting_dict) in enumerate(weightings.items()):
        ax = axs[plot_idx]
        for idx, (weighting, weighting_name) in enumerate(weighting_dict.items()):
            p = get_p(n, weighting)
            ax.plot(np.arange(n), p, label=weighting_name, linestyle=linestyles[idx], color=cb_colors[idx])
        ax.set_title(weightings_name + fr" ($n={n}$)")
        ax.set_xlabel(r"$|S|$")
        if plot_idx == 0:
            ax.set_ylabel(r"$p_{|S|}$")
        ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    plt.savefig("images/probabilistic_values.pdf", bbox_inches='tight', dpi=300)

import scipy

def compute_bn(p, n):
    valid_sizes = np.array(list(range(n+1)))

    sum1, sum2 = 0, 0
    for ell in range(n):
        sum1 += scipy.special.binom(n-1, ell) * p[ell]**2
        if ell >= 1:
            sum2 += scipy.special.binom(n-2, ell-1) * (p[ell]-p[ell-1])**2
    an = 2*sum1 - sum2
    bn = sum2 / an
    return bn


def plot_bn(ns, figsize):
    # 3 plots in a row
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for plot_idx, (weightings_name, weighting_dict) in enumerate(weightings.items()):
        ax = axs[plot_idx]
        for idx, (weighting, weighting_name) in enumerate(weighting_dict.items()):
            bns = []
            for n in ns:
                p = get_p(n, weighting)
                bn = compute_bn(p, n)
                bns += [bn]
            ax.plot(ns, bns, label=weighting_name, linestyle=linestyles[idx], color=cb_colors[idx])
        ax.set_title(weightings_name)
        ax.set_xlabel(r"$b_n$")
        if plot_idx == 0:
            ax.set_ylabel(r"$b_n$")
        ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.legend()

    plt.tight_layout()
    try:
        plt.savefig("images/bn.pdf", bbox_inches='tight', dpi=300)
    except:
        plt.show()

figsize = (12,3)
plot_bn(ns=list(range(1,501)), figsize=figsize)

plot_probabilistic_values(n=10, figsize=figsize)
