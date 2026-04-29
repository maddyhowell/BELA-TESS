from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
import lightkurve as lk
from astropy.units import cds
from astropy.time import Time
from scipy import interpolate
from astropy.convolution import Box1DKernel, convolve, Gaussian1DKernel, convolve_fft
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
import warnings
from lightkurve import periodogram
from matplotlib import patches
from scipy.ndimage import binary_dilation
from scipy.ndimage import uniform_filter1d


""" Functions needed to detrend TESS FFI """

def correct_with_proper_motion(ra, dec, pm_ra, pm_dec, tpf):
    """Return proper-motion corrected RA / Dec.
       It also return whether proper motion correction is applied or not.
       Taken from light curve
       """
    # all parameters must have units
    ra,dec = ra * u.deg , dec * u.deg

    pm_unit = u.milliarcsecond / u.year   # THIS IS JUST FOR TESS
    pm_ra, pm_dec = pm_ra * pm_unit, pm_dec * pm_unit

    equinox = Time('2000', format='byear')

    new_time = tpf.time[0]

    if ra is None or dec is None or \
       pm_ra is None or pm_dec is None or (np.all(pm_ra == 0) and np.all(pm_dec == 0)) or \
       equinox is None:
        return ra, dec, False

    # To be more accurate, we should have supplied distance to SkyCoord
    # in theory, for Gaia DR2 data, we can infer the distance from the parallax provided.
    # It is not done for 2 reasons:
    # 1. Gaia DR2 data has negative parallax values occasionally. Correctly handling them could be tricky. See:
    #    https://www.cosmos.esa.int/documents/29201/1773953/Gaia+DR2+primer+version+1.3.pdf/a4459741-6732-7a98-1406-a1bea243df79
    # 2. For our purpose (ploting in various interact usage) here, the added distance does not making
    #    noticeable significant difference. E.g., applying it to Proxima Cen, a target with large parallax
    #    and huge proper motion, does not change the result in any noticeable way.
    #
    c = SkyCoord(ra, dec, pm_ra_cosdec=pm_ra, pm_dec=pm_dec,
                frame='icrs', obstime=equinox)

    # Suppress ErfaWarning temporarily as a workaround for:
    #   https://github.com/astropy/astropy/issues/11747
    with warnings.catch_warnings():
        # the same warning appears both as an ErfaWarning and a astropy warning
        # so we filter by the message instead
        warnings.filterwarnings("ignore", message="ERFA function")
        new_c = c.apply_space_motion(new_obstime=new_time)

    return new_c.ra, new_c.dec, True

def GaiaSearch(tpf, magnitude_limit = 18, GaiaDR = 3, pix_scale = 21.0):
    c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
    # Use pixel scale for query size
    # pix_scale = 4.0  # arcseconds / pixel for Kepler, default
    # We are querying with a diameter as the radius, overfilling by 2x.

    Vizier.ROW_LIMIT = -1
    if str(GaiaDR) == '2':
        result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
    elif str(GaiaDR) == '3':
        result = Vizier.query_region(c1, catalog=["I/355/gaiadr3"],
                                 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))

    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))

    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message

    if str(GaiaDR) == '2':
        result = result["I/345/gaia2"].to_pandas()
    elif str(GaiaDR) == '3':
        # print('Using GaiaDR3')
        result = result["I/355/gaiadr3"].to_pandas()

    result = result[result.Gmag < magnitude_limit]
    # if len(result) == 0:
    #     raise no_targets_found_message

    return result

def DitherApertureMasks(tpf, ra, dec, pmra, pmdec, n_dim = 3, pixel_shift = (0,0)):
    ## function to iterate on aperture masks. Can be a nxn or nxm mask
    ## pixel shift is used if the centroid doesn't do a good job finding the best central pixel
    ra_corr, dec_corr, __ = correct_with_proper_motion(ra, dec, pmra, pmdec, tpf)  # account for proper motions in TPFs

    radecs = np.vstack([ra_corr,dec_corr]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)

    # return pixel positions of star in coordinates of the TPF
    x_star, y_star = coords[:, 0] + pixel_shift[0], coords[:, 1]+ pixel_shift[0]
    x_star, y_star = int(x_star[0]), int(y_star[0])


    ap_mask_dict = {}

    option_idx = 0

    for di in [-1, 0, 1]:

        ap_mask_temp = np.zeros(np.shape(tpf)[1:])

        # compute new center
        ci = y_star + di
        cj = x_star

        # set n_dim x n_dim block
        if np.shape(n_dim) == ():
            ap_mask_temp[ci-1:ci+(n_dim-1), cj-1:cj+(n_dim-1)] = True
        elif np.shape(n_dim) == (2,):
            ap_mask_temp[ci-1:ci+(n_dim[0]-1), cj-1:cj+(n_dim[1]-1)] = True
        else:
            print('Not valid dimensions chosen for aperture mask')

        ap_mask_dict[f'Option{option_idx}'] = ap_mask_temp
        option_idx += 1

    for dj in [-1, 1]:
        ap_mask_temp = np.zeros(np.shape(tpf)[1:])

        # compute new center
        ci = y_star
        cj = x_star + dj

        # set n_dim x n_dim block
        if np.shape(n_dim) == ():
            ap_mask_temp[ci-1:ci+(n_dim-1), cj-1:cj+(n_dim-1)] = True
        elif np.shape(n_dim) == (2,):
            ap_mask_temp[ci-1:ci+(n_dim[0]-1), cj-1:cj+(n_dim[1]-1)] = True
        else:
            print('Not valid dimensions chosen for aperture mask')


        ap_mask_dict[f'Option{option_idx}'] = ap_mask_temp
        option_idx += 1


    return ap_mask_dict

def NeighbourStarSearch(tpf, ap_mask, ra, dec, pmra, pmdec, gaia_id, g_mag, pix_scale = 21.0, mag_limit=16, ax = None):

    results = GaiaSearch(tpf, magnitude_limit = mag_limit, pix_scale = pix_scale)
    print(f'Showing all Gaia sources brighter than Gmag = {mag_limit}')

    if ax == None:
        ax = tpf.plot();
    # else:
    #     tpf.plot(ax=ax);

    if isinstance(ap_mask, np.ndarray):
        in_aperture = np.where(ap_mask)
        ap_row = in_aperture[0] + tpf.row - 0.5
        ap_col = in_aperture[1] + tpf.column - 0.5
        for ii in range(len(ap_row)):
            rect = patches.Rectangle(
                (ap_col[ii], ap_row[ii]),
                1,
                1,
                fill=False,
                hatch="//",
                color='r',
            )
            ax.add_patch(rect)

    elif isinstance(ap_mask, dict):
        mask_colours = ['r', 'orange', 'cyan', 'black', 'magenta']
        mask_hatches = ['//', '\\', '+', '.', '*']
        mc_idx = 0
        for option, mask in ap_mask.items():
            in_aperture = np.where(mask)
            ap_row = in_aperture[0] + tpf.row - 0.5
            ap_col = in_aperture[1] + tpf.column - 0.5
            for ii in range(len(ap_row)):
                rect = patches.Rectangle(
                    (ap_col[ii], ap_row[ii]),
                    1,
                    1,
                    fill=False,
                    hatch=mask_hatches[mc_idx],
                    color=mask_colours[mc_idx],
                )
                ax.add_patch(rect)
            # print(f'Done {option}')
            mc_idx += 1


    ra_corr, dec_corr, __ = correct_with_proper_motion(ra, dec, pmra, pmdec, tpf)  # account for proper motions in TPFs

    radecs = np.vstack([ra_corr,dec_corr]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)

    # return pixel positions of star in coordinates of the TPF
    x, y = coords[:, 0] + tpf.column, coords[:, 1] + tpf.row
    x, y = round(x[0],2), round(y[0],2)

    ii = 0
    for idx,row in results.iterrows():

        if row.Source == gaia_id:
            continue

        else:
            ra_s, dec_s, pmra_s, pmdec_s = row.RA_ICRS, row.DE_ICRS, row.pmRA, row.pmDE
            ra_corr_s, dec_corr_s, __ = correct_with_proper_motion(ra_s, dec_s, pmra_s, pmdec_s, tpf)  # account for proper motions in TPFs

            radecs = np.vstack([ra_corr_s, dec_corr_s]).T
            coords = tpf.wcs.all_world2pix(radecs, 0)
            x_temp, y_temp = coords[:, 0] + tpf.column, coords[:, 1] + tpf.row
            x_temp, y_temp = round(x_temp[0],2), round(y_temp[0],2)

            if y_temp < tpf.row or y_temp > tpf.row + np.shape(tpf)[2]:
                if x_temp < tpf.column or x_temp > tpf.column + np.shape(tpf)[1]:

                    continue
            elif x_temp < tpf.column or x_temp > tpf.column + np.shape(tpf)[1]:
                if y_temp < tpf.row or y_temp > tpf.row + np.shape(tpf)[2]:

                    continue
            else:

                scale_fac = g_mag/row.Gmag
                # print(scale_fac)
                # fac = 150 - 250*(1- scale_fac)
                fac = 40 - 200*(1- scale_fac)

                if fac < 0:
                    fac = 1

                if ii == 0:
                    ax.scatter(x_temp, y_temp, c = 'grey', edgecolor="k", alpha = 0.9, s = fac, label = r'Neighbouring stars with $G_{\rm}< 16$')
                    ii += 1
                    # print('Gaia IDS for neighbouring stars')
                    # print('-------------------------------')
                else:
                    ax.scatter(x_temp, y_temp, c = 'grey', edgecolor="k", alpha = 0.9, s = fac)

                # print(f'Gaia {row.Source}')

    ax.set_xlim(tpf.column-0.5, tpf.column + np.shape(tpf)[2]-0.5)
    ax.set_ylim(tpf.row-0.5, tpf.row + np.shape(tpf)[1]-0.5)

    ax.scatter(x,y,facecolor="r", edgecolor="k", marker = 'X', s = 100, label = 'Target star')

    # ax.legend(bbox_to_anchor=(1.2, 1), fontsize = 12)

def CorrectPLD(tpf, aperture_mask, sector, pca_comps=7):

    # Correct for systematic trends identified in background pixels using PLD
    # default pca components taken from the TESS Utils python package. This can be played around with
    pld = lk.PLDCorrector(tpf, aperture_mask=ap_mask)
    pld.correct(pca_components=pca_comps)
    pld.diagnose()
    pld.diagnose_masks();

    lc_pld = pld.correct(pca_components=pca_comps, aperture_mask=ap_mask)

    return lc_pld

def PlotPCAComponents(dm, pca_comps):
    dm = dm.pca(pca_comps)
    plt.plot(tpf.time.value, dm.values + np.arange(pca_comps)*0.2, '.');

    dm.plot();

# below follows tutorial: https://heasarc.gsfc.nasa.gov/docs/tess/NoiseRemovalv2.html
def CorrectReggression(tpf, aperture_mask, sector, pca_comps=7, verbose = True):

    # define background aperture mask
    not_ap_mask = 1-aperture_mask

    tpf = tpf[tpf.quality == 0]  # remove bad quality flags

    dm = lk.DesignMatrix(tpf.flux[:,  not_ap_mask.astype(bool)], name='regressors').pca(pca_comps)

    # PlotPCAComponents(dm, pca_comps) # uncomment to plot the pca components in the design matrix (dm)

    dm = dm.append_constant()

    # not a fan of using fill_gaps because it is adding Gaussian noise (would prefer to do a linear interpolation) but its difficult to get the right product for the regression corrector
    # uncorrected_lc = tpf.to_lightcurve(aperture_mask=ap_mask).fill_gaps()
    uncorrected_lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

    # uncorrected_lc = uncorrected_lc[uncorrected_lc.quality == 0]   ##redundant due to doing this for the tpf

    try:
        reg_corr = lk.RegressionCorrector(uncorrected_lc)

    except ValueError:
        uncorrected_lc = tpf.to_lightcurve(aperture_mask=aperture_mask).fill_gaps()
        reg_corr = lk.RegressionCorrector(uncorrected_lc)
    else:
        reg_corr = lk.RegressionCorrector(uncorrected_lc)


    lc_temp = reg_corr.correct(dm)

    if verbose:
        reg_corr.diagnose()

    # Remove the scattered light, allowing for the large offset from scattered light in TESS
    lc_RC = uncorrected_lc - reg_corr.model_lc + np.percentile(reg_corr.model_lc.flux, 5)

    lc_final = lc_RC.remove_outliers(5).normalize('ppm')

    return lc_final

def detect_momentum_dump_segments(lc_final, deriv_thresh=8, expand_points=50, window=25, var_thresh=3):
    """
    From ChatGPT, key input parameters:
        deriv_thresh (≈ 4–7)
        → sensitivity to jumps
        expand_points (≈ 30–100)
        → how long the momentum dump lasts
        (depends on cadence: for 2-min TESS, 50 ≈ ~1.5 hours)
        var_thresh (≈ 2–4)
        → how noisy the post-dump region must be
    """

    time, flux = lc_final.time.value, lc_final.flux.value

    # --- Step 1: derivative (find sharp jumps) ---
    dflux = np.gradient(flux, time)
    dmed = np.nanmedian(dflux)
    dstd = 1.4826 * np.nanmedian(np.abs(dflux - dmed))  # robust std

    jump_mask = np.abs(dflux - dmed) > deriv_thresh * dstd

    # --- Step 2: local variability (confirm bad regions) ---
    mean = uniform_filter1d(flux, size=window)
    mean_sq = uniform_filter1d(flux**2, size=window)
    local_std = np.sqrt(mean_sq - mean**2)

    med = np.nanmedian(local_std)
    mad = np.nanmedian(np.abs(local_std - med))
    robust_std = 1.4826 * mad

    var_mask = local_std > (med + var_thresh * robust_std)

    # --- Step 3: combine ---
    seed_mask = jump_mask | var_mask

    # --- Step 4: grow into segments ---
    bad_segments_mask = binary_dilation(seed_mask, iterations=expand_points)

    lc_final = lc_final[~bad_segments_mask]

    return lc_final

def boxcar_high_pass_filter(data, window_size):
    """
    Apply a boxcar high-pass filter by subtracting a moving average

    Parameters:
    - data: array-like, the input signal
    - window_size: int, length of the moving average window

    Returns:
    - high_passed: np.ndarray, the high-pass filtered signal
    """
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    # Boxcar filter (simple moving average)

    boxcar = Box1DKernel(window_size)
    smoothed = convolve(data, boxcar, boundary='extend', normalize_kernel=True)

    high_passed = data - smoothed

    return high_passed

def triangular_high_pass_filter(data, window_size):
    """
    Apply a triangular (boxcar twice) high-pass filter by subtracting a moving average

    Parameters:
    - data: array-like, the input signal
    - window_size: int, length of the moving average window

    Returns:
    - high_passed: np.ndarray, the high-pass filtered signal
    """

    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    # Boxcar filter (simple moving average)

    boxcar = Box1DKernel(window_size)
    smoothed = convolve(data, boxcar, boundary='extend', normalize_kernel=True)
    smoothed2 = convolve(smoothed, boxcar, boundary='extend', normalize_kernel=True)


    high_passed = data - smoothed2

    return high_passed

def Gaussian_high_pass_filter(data, window_size):
    """
    Apply a Gaussian (boxcar four times) high-pass filter by subtracting a moving average

    Parameters:
    - data: array-like, the input signal
    - window_size: int, length of the moving average window

    Returns:
    - high_passed: np.ndarray, the high-pass filtered signal
    """

    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    # Boxcar filter (simple moving average)

    gauss = Gaussian1DKernel(window_size)
    smoothed = convolve(data, gauss, boundary='extend', normalize_kernel=True)


    high_passed = data - smoothed

    return high_passed

def HighPassFilter(lc, highpass_filter, method):

    # lc = lc.remove_outliers(3).normalize('ppm')
    ### LIGHT CURVE SHOULD BE NORMALISED WHEN PASSED THROUGH FUNCTION

    if method == 'Flatten':
        # High-pass filter. Default filter width is 4 days (attenuates frequency below 3 microHz)
        window_length =  int(np.shape(lc)[0]/((lc.time.value[-1]-lc.time.value[0])/highpass_filter))

        window_length = window_length ## get rid of the dimensions and convert to an int
        if window_length % 2 == 0:
            window_length += 1

        lc_corr = lc.flatten(window_length)


    elif method == 'Boxcar':
        filter_width = int(np.shape(lc)[0]/((lc.time.value[-1]-lc.time.value[0])/highpass_filter))

        high_passed = boxcar_high_pass_filter(lc.flux.value, window_size = filter_width)

        time = Time(lc.time.value, scale='tdb', format='btjd')

        lc_dict = {'time': time, 'flux': high_passed * cds.ppm}

        lc_corr = lk.LightCurve(lc_dict)


    elif method == 'Triangular':
        filter_width = int(np.shape(lc)[0]/((lc.time.value[-1]-lc.time.value[0])/highpass_filter))

        high_passed = triangular_high_pass_filter(lc.flux.value, window_size = filter_width)

        time = Time(lc.time.value, scale='tdb', format='btjd')

        lc_dict = {'time': time, 'flux': high_passed * cds.ppm}

        lc_corr = lk.LightCurve(lc_dict)

    elif method == 'Gaussian':
        filter_width = int(np.shape(lc)[0]/((lc.time.value[-1]-lc.time.value[0])/highpass_filter))

        high_passed = Gaussian_high_pass_filter(lc.flux.value, window_size = filter_width)

        time = Time(lc.time.value, scale='tdb', format='btjd')

        lc_dict = {'time': time, 'flux': high_passed * cds.ppm}

        lc_corr = lk.LightCurve(lc_dict)


    lc_final = lc_corr

    return lc_final

def Find_Dnu_relations(keyword):

    Dnu_relations = {'GC_RGB': [0.3,0.75],
                     'GC_RHB': [0.3, 0.86],
                     'GC_AGB': [0.3, 0.77],
                     'Yu18': [0.267, 0.764],
                     }

    return Dnu_relations[keyword]


def ps_smooth(frequency, power, numax_est, Dnu_relation, sm):
    """ This function should be executed after ps_no_slope """

    if type(Dnu_relation) == str:
        ## go to file and find Dnu_relation coefficient and exponent

        dnu_coefficient, dnu_exponent = Find_Dnu_relations(Dnu_relation)

    else:
        dnu_coefficient, dnu_exponent = Dnu_relation


    dnu_est = dnu_coefficient*(numax_est)**dnu_exponent

    resolution = frequency[1] - frequency[0]

    ## following code taken from pysyd target function line 693
    if sm == None:
        numax_sun = 3090.00
        sm = 4.*(numax_est/numax_sun)**0.2
        if sm < 1.:
            sm = 1.


    sig = (sm*(dnu_est/resolution))/np.sqrt(8.0*np.log(2.0))
    # print(sig)   # sig is same as pySYD
    pssm = convolve_fft(np.copy(power), Gaussian1DKernel(int(sig)))


    return pssm

def remove_small_gaps(lc, cadence = 30):

    idx_insert = []
    xvalue_insert = []
    yvalue_insert = []

    if cadence == 30:
        point_offset = 30/(60*24)    # 30 min cadence in days
    elif cadence == 10:
        point_offset = 10/(60*24)    # 10 min cadence in days
    elif cadence == 200:
        point_offset = 200/(60*60*24)    # 200 sec cadence in days


    for idx, t in enumerate(lc.time.value):
        if idx < len(lc.time.value)-1:
            if not np.abs(lc.time.value[idx] - lc.time.value[idx+1]) < point_offset + 0.5*point_offset:                 # if gap is longer than cadence (plus tolerance of 50% of cadence
                if not np.abs(lc.time.value[idx] - lc.time.value[idx+1]) > 4*point_offset:                              # if gap is 4 or more cadences long that don't fill in

                    x = [lc.time.value[idx], lc.time.value[idx+1]]
                    y = [lc.flux.value[idx], lc.flux.value[idx+1]]

                    f = interpolate.interp1d(x, y)
                    yvalue_insert.append(f(x[0]+ point_offset)), xvalue_insert.append(x[0] + point_offset), idx_insert.append(idx+1)

                    if np.abs(lc.time.value[idx] - lc.time.value[idx+1]//point_offset) == 2.0:
                        yvalue_insert.append(f(x[0]+2*point_offset)), xvalue_insert.append(x[0]+2*point_offset), idx_insert.append(idx+1)
                    elif np.abs(lc.time.value[idx] - lc.time.value[idx+1]//point_offset) == 3.0:
                        yvalue_insert.append(f(x[0]+2*point_offset)), xvalue_insert.append(x[0]+2*point_offset), idx_insert.append(idx+1)
                        yvalue_insert.append(f(x[0]+3*point_offset)), xvalue_insert.append(x[0]+3*point_offset), idx_insert.append(idx+1)



    time_temp = np.insert(lc.time.value, idx_insert, xvalue_insert)
    flux_temp = np.insert(lc.flux.value, idx_insert, yvalue_insert)

    time_new = Time(time_temp, scale='tdb', format='btjd')

    flux_new = flux_temp * lc.flux.unit           # assign units of ppm for normalised lc


    lc_dict = {'time': time_new, 'flux': flux_new}

    lc_new = lk.LightCurve(lc_dict)   # return new normalised light curve

    return lc_new

def calc_Tprime(lc, cadence):

    time_total = lc.time.value[-1] - lc.time.value[0]

    N_expectedpoints = time_total/(cadence/(60*24))

    N_observedpoints = len(lc.time.value)

    ratio_lc_observed = N_observedpoints/N_expectedpoints

    return ratio_lc_observed

def PSD_fixedfreqgrid(lc,  minimum_frequency, maximum_frequency, cadence, oversample_factor):
    # If Lightcurve contains NaN values, these are removed before creating the periodogram.
    if np.isnan(lc.flux).any() or (hasattr(lc.flux, 'unmasked') and np.isnan(lc.flux.unmasked).any()):
            lc = lc.remove_nans()

    freq_unit = u.microhertz

    time = lc.time.copy()


    nyquist = nyquist = 0.5 * (1/(cadence*60))*10**6   # assumes cadence is in minutes
    # Approximate Nyquist Frequency and frequency bin width in terms of days


    ### -------------------------------------------------
    ## Change this line with the new idea. 'fs' is frequency bin width (or frequency spacing I assume)
    ratio_lc_observed = calc_Tprime(lc, cadence)
    Tprime = ratio_lc_observed * (lc.time.value[-1] - lc.time.value[0])
    fs = (1.0 / (Tprime*24*60*60)) * 10**6  / oversample_factor
    print(f'new frequency spacing is {fs} muHz. Old frequency spacing was {1.0/((lc.time.value[-1] - lc.time.value[0])*24*60*60)*10**6} muHz')
    ### -------------------------------------------------

    # Convert these values to requested frequency unit
    # nyquist = nyquist * freq_unit
    # fs = fs * freq_unit

    ls_method =  "fast"   # default in lightkurve
    nterms = 1            # default in lightkurve

    # Do unit conversions if user input min/max frequency or period

    # If nothing has been passed in, set them to the defaults
    if minimum_frequency is None:
        minimum_frequency = fs
    if maximum_frequency is None:
        maximum_frequency = nyquist

    # Create frequency grid evenly spaced in frequency
    frequency = np.arange(minimum_frequency, maximum_frequency, fs)


    # Convert to desired units
    frequency = u.Quantity(frequency, freq_unit)


    LS = LombScargle(time, lc.flux, nterms=nterms, normalization="psd")
    power = LS.power(frequency, method=ls_method)



    # Rescale from the unnormalized power output by Astropy's
    # Lomb-Scargle function to units of flux_variance / [frequency unit]
    # that may be of more interest for asteroseismology.
    fs = fs * freq_unit
    power *= 2.0 / (len(time) * fs * oversample_factor)

    psd = periodogram.Periodogram(frequency, power)

    return psd

def calc_PSD(lc, method = 'original', min_freq = 1, max_freq = 277.78, oversample = 5, cadence = 30):
    """ methods to calculate the power spectral density:
    'original': uses the frequency resolution defined by the baseline in the inputted light curve
    'setfreqres': Defines the frequency resolution on the number of data points in the light curve (e.g. if there is several large gaps)
    """

    if method == 'original':
        psd = lc.to_periodogram(method='lombscargle', normalization='psd', maximum_frequency = max_freq, minimum_frequency = min_freq, oversample_factor = oversample)
    elif method == 'setfreqres':
        psd = PSD_fixedfreqgrid(lc, minimum_frequency = min_freq, maximum_frequency = max_freq, cadence = cadence, oversample_factor = oversample)
    else:
        print('Provided method is not an option.')
        psd = None

    return psd
