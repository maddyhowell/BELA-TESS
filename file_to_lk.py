import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from  lightkurve import periodogram
from astropy import units as u
from astropy.units import cds
from astropy.time import Time

def txt_to_LC(filename, label = None, normalised = True):

    txt = np.loadtxt(filename).T

    time, flux = txt[0], txt[1]

    time = Time(time, scale='tdb', format='btjd')

    if normalised == True:
        flux = flux * cds.ppm           # assign units of ppm for normalised lc

    else:
        flux = flux * u.electron/u.s           # assign units of ppm for non-normalised lc

    lc_dict = {'time': time, 'flux': flux}

    lc = lk.LightCurve(lc_dict)

    return lc


def txt_to_PSD(filename, label = None, freq_min = 1, freq_max = 280):

    txt = np.loadtxt(filename).T

    freq, power = txt[0], txt[1]

    freq = freq* u.microhertz                           # assign freq_unit

    power = power * cds.ppm**2/u.microhertz             # assign units of flux^2/freq_unit

    psd = periodogram.Periodogram(freq, power, label = label)

    return psd
