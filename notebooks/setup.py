
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si

def blackScholes(S, T , X , r, sigma):
    if isinstance(S, np.ndarray):
        result = np.zeros( len(S) )
        for i, s in enumerate(S):
            if s == 0.:
                result[i] = 0.
            else:
                d1 = (np.log(s / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                d2 = (np.log(s / X) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                result[i] = (s * si.norm.cdf(d1, 0.0, 1.0) - X * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    else:
        d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / X) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        result = X * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) 
    return result

def f_plot(T , X , r, sigma):
    plt.figure(2)
    x = np.linspace(0, 2 * X, num=1000)
    plt.plot(x, blackScholes(x, T , X , r, sigma) )
    plt.ylim(0, X)
    plt.show()

def f_plot_interactive():
    return interact_manual(f_plot, T=(0.0, 5.0 , 0.25), X=(10, 1000, 10), r=(0, 0.1, 0.001), sigma=(0., 1., 0.01))
