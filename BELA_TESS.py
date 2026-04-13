import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TESS_lightcurves as TESS_LC
from matplotlib import gridspec
from matplotlib.widgets import RectangleSelector
from astropy.visualization import   PercentileInterval
import os
import file_to_lk as f2lk
import lightkurve as lk

def FinaliseSectorLightCurve(tpf, ap_mask, sector, numax_guess):
    #### Step 1: Do background correction with regression corrector
    lc_RC = TESS_LC.CorrectReggression(tpf, ap_mask, sector, pca_comps = 7, verbose = False)

    cadence = tpf.hdu[1].header['EXPOSURE']*(24*60*60)  # cadence in seconds (originally in days in header)


    #### Step 2: bin data to 30mins if using varying sampled data
    if cadence > 1420:  # assuming some padding, for is for cadences of 30mins
        # print(f'Sector {sector} has 30min cadence')

        ## FILL GAPS HERE

        lc_RC = TESS_LC.remove_small_gaps(lc_RC)

    else:
        # print(f'Sector {sector} DOES NOT has 30min cadence')
        lc_RC = lc_RC.bin(time_bin_size = 30/(60*24))

    #### Step 3: Take a high-pass filter, which normalises the light curve
    if numax_guess <= 20:
        hp = 0
        lc_final = lc_RC
    else:    # also for when numax_guess is None
        hp = 4
        lc_final = TESS_LC.HighPassFilter(lc_RC, highpass_filter = hp, method = 'Triangular')



    return lc_final

class TPFMaskSelector:
    def __init__(self, star_id, tpf, ra, dec, pmra, pmdec, gmag, gaia_id, numax_guess, frame=0):
        """
        Interactive TPF mask selector and time mask for light curve, with live light curve and power spectra update.
        """
        self.star_id = star_id
        self.ra = ra
        self.dec = dec
        self.pmra = pmra
        self.pmdec = pmdec
        self.gmag = gmag
        self.gaia_id = gaia_id
        self.numax_guess = numax_guess

        self.tpf = tpf
        self.lc = None
        self.psd = None

        self.prev_lc = []
        self.prevprev_lc = []

        self.prev_psd = []
        self.prevprev_psd = []

        #### Convert frame to numpy array -------------------------------------
        self.image = tpf.flux[frame]
        if hasattr(self.image, "value"):
            self.image = self.image.value
        self.image = np.array(self.image)

        self.ny, self.nx = self.image.shape
        self.pixel_mask = np.zeros_like(self.image, dtype=bool)


        #### Initialize figure with 3 subplots: image and light curve and power spectra -------------------------------------
        plt.close('all')

        self.fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 3, figure=self.fig, width_ratios=[1, 2, 2], height_ratios=[1], wspace = 0.2)

        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_lc = self.fig.add_subplot(gs[0, 1])
        self.ax_psd = self.fig.add_subplot(gs[0, 2])

        #### Figure position **kwargs -------------------------------------

        pos0 = self.ax_img.get_position()
        pos1 = self.ax_lc.get_position()
        pos2 = self.ax_psd.get_position()

        self.ax_lc.set_position([pos1.x0 + 0.03, pos1.y0, pos1.width, pos1.height])  # bigger shift
        self.ax_psd.set_position([pos2.x0 + 0.04, pos2.y0, pos2.width, pos2.height])  # smaller shift

        #### Plot TPF image -------------------------------------
        self.PlotTPF()

        #### Plot nearby sources surrounding target
        self.NearbySources()

        self.sector = self.tpf.hdu[0].header['sector']
        self.fig.suptitle(f'TIC {self.star_id} - Sector {self.sector} Photometric Analysis')

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        print('CLLIICCKKK')
        # Only respond if click is inside TPF image axes
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)

        # Bounds check
        if x < 0 or x >= self.nx or y < 0 or y >= self.ny:
            return

        # Toggle mask
        self.pixel_mask[y, x] = not self.pixel_mask[y, x]

        # Update scatter overlay
        ys, xs = np.where(self.pixel_mask)
        self.scatter.set_offsets(np.c_[xs, ys])

        # Update light curve
        self.update_lightcurve()

        # Update power spectra
        self.update_powerspectra()

        # Redraw figure
        self.fig.canvas.draw_idle()

    def PlotTPF(self):

        vmin_default, vmax_default = PercentileInterval(95.0).get_limits(self.image)
        self.ax_img.imshow(self.image, origin='lower', cmap='viridis',  aspect='equal', vmin = vmin_default, vmax = vmax_default)

        # Overlay scatter for selected pixels in aperture mask
        self.scatter = self.ax_img.scatter([], [], s=180, facecolors='none', edgecolors='red', linewidths=1, marker = 's', hatch = '///', alpha = 0.6)
                # Identify target star and bright nearby sources
        ra_corr, dec_corr, __ = TESS_LC.correct_with_proper_motion(self.ra, self.dec, self.pmra, self.pmdec, self.tpf)  # account for proper motions in TPFs
        radecs = np.vstack([ra_corr,dec_corr]).T
        coords = self.tpf.wcs.all_world2pix(radecs, 0)

        # return pixel positions of star in coordinates of the TPF
        # x, y = coords[:, 0] + self.tpf.column, coords[:, 1] + self.tpf.row
        x_star, y_star = coords[:, 0], coords[:, 1]
        x_star, y_star = round(x_star[0],2), round(y_star[0],2)
        self.ax_img.scatter(x_star,y_star,facecolor="r", edgecolor="k", marker = 'X', s = 100)

        self.ax_img.set_xlabel('Pixel Column Number')
        self.ax_img.set_ylabel('Pixel Row Number')
        self.ax_img.set_title('')

    def NearbySources(self):
        results = TESS_LC.GaiaSearch(self.tpf, magnitude_limit = self.gmag+2)
        # print(results)

        ii = 0
        for idx,row in results.iterrows():

            if row.Source == self.gaia_id:
                continue

            else:
                ra_s, dec_s, pmra_s, pmdec_s = row.RA_ICRS, row.DE_ICRS, row.pmRA, row.pmDE
                ra_corr_s, dec_corr_s, __ = TESS_LC.correct_with_proper_motion(ra_s, dec_s, pmra_s, pmdec_s, self.tpf)  # account for proper motions in TPFs

                radecs = np.vstack([ra_corr_s, dec_corr_s]).T
                coords = self.tpf.wcs.all_world2pix(radecs, 0)
                x_temp, y_temp = coords[:, 0], coords[:, 1]
                x_temp, y_temp = round(x_temp[0],2), round(y_temp[0],2)

                if y_temp < -0.5 or y_temp >  np.shape(self.tpf)[2]:
                    if x_temp < self.tpf.column or x_temp > self.tpf.column + np.shape(self.tpf)[1]:

                        continue
                elif x_temp < -0.5 or x_temp > np.shape(self.tpf)[1]:
                    if y_temp < -0.5 or y_temp >  np.shape(self.tpf)[2]:

                        continue
                else:
                    scale_fac = self.gmag/row.Gmag
                    # print(scale_fac)
                    # fac = 150 - 250*(1- scale_fac)
                    fac = 40 - 200*(1- scale_fac)

                    if fac < 0:
                        fac = 1

                    if ii == 0:
                        self.ax_img.scatter(x_temp, y_temp, c = 'grey', edgecolor="k", alpha = 0.9, s = fac, label = r'Neighbouring stars with $G_{\rm}< 16$')
                        ii += 1
                        # print('Gaia IDS for neighbouring stars')
                        # print('-------------------------------')
                    else:
                        self.ax_img.scatter(x_temp, y_temp, c = 'grey', edgecolor="k", alpha = 0.9, s = fac)




    def update_lightcurve(self):
        """
        Compute light curve from current mask and update plot.
        """

        if np.any(self.pixel_mask):

            self.ax_lc.cla()

            self.lc = FinaliseSectorLightCurve(self.tpf, self.pixel_mask, self.sector, self.numax_guess)


            self.lc.plot(ax=self.ax_lc, c = 'k')


            ### Plot last two previous light curves
            if self.prev_lc == []:
                pass
            elif self.prevprev_lc == []:
                self.ax_lc.plot(self.prev_lc.time.value, self.prev_lc.flux.value, c = 'r', alpha = 0.1, label = 'Last option')
                self.ax_lc.legend(loc = 'lower left')

            else:
                self.ax_lc.plot(self.prevprev_lc.time.value, self.prevprev_lc.flux.value, c = 'b', alpha = 0.1, label = 'Last option')
                self.ax_lc.plot(self.prev_lc.time.value, self.prev_lc.flux.value, c = 'r', alpha = 0.1, label = 'Two option previous')


                self.ax_lc.legend(loc = 'lower left')

            ### Save last two previous light curves
            self.prevprev_lc = self.prev_lc.copy()
            self.prev_lc = self.lc.copy()

            self.ax_lc.relim()
            self.ax_lc.autoscale_view()


        return self.lc

    def update_powerspectra(self):
        """
        Compute spectra from current mask and light curve, and update plot.
        """

        if np.any(self.pixel_mask):
            self.ax_psd.cla()

            self.psd = TESS_LC.calc_PSD(self.lc, method = 'original', oversample=1, min_freq=0.01)
            try:
                pssm = TESS_LC.ps_smooth(self.psd.frequency.value, self.psd.power.value, self.numax_guess, 'Yu18', 2)
            except:
                pssm = np.full(np.shape(self.psd.frequency.value), np.nan)


            self.psd.plot(ax=self.ax_psd, c = 'k')
            self.ax_psd.plot(self.psd.frequency.value, pssm, lw = 3, c = 'mediumpurple')
            self.ax_psd.set_xlim(1,200)
            self.ax_psd.axvline(self.numax_guess, c = 'r', ls = 'dashed', alpha = 0.5)
            self.ax_psd.set_xscale('log'), self.ax_psd.set_yscale('log')

            ### Plot last two previous power spectra
            if self.prev_psd == []:
                pass
            elif self.prevprev_psd == []:
                self.ax_psd.plot(self.prev_psd.frequency.value, self.prev_psd.power.value, c = 'r', alpha = 0.1)
            else:
                self.ax_psd.plot(self.prevprev_psd.frequency.value, self.prevprev_psd.power.value, c = 'b', alpha = 0.1)
                self.ax_psd.plot(self.prev_psd.frequency.value, self.prev_psd.power.value, c = 'r', alpha = 0.1)



            ### Save last two previous power spectra
            self.prevprev_psd = self.prev_psd.copy()
            self.prev_psd = self.psd.copy()

            self.ax_psd.relim()
            self.ax_psd.autoscale_view()


        return self.psd


    def get_mask(self):
        return self.pixel_mask

class FinalLCSelector:
    def __init__(self, lc_files, numax_guess, star_id):

        #### Initialize input parameters --------------------------
        self.numax_guess = numax_guess
        self.lc_files = np.array(sorted(lc_files))
        self.lc_list = []
        self.sector_list = []
        self.psd_all = None
        self.sector_times = None
        self.removed_sectors = set()
        self.star_id = star_id

        self.test_parameter = None

        #### save the tess sectors based on the txt light curve files --------------------------
        for lf in lc_files:
            sec = lf.split('/')[-1].split('Sector')[-1].split('_')[0]
            self.sector_list.append(int(sec))

        # print(os.getcwd())
        self.sector_times = pd.read_csv('./TESS_Sectors_Times.csv')

        self.lc_sector_times = self.sector_times.loc[self.sector_times['Sector'].isin(self.sector_list), 'mid_time'].values

        #### Create Figure  --------------------------
        plt.close('all')
        self.fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[5,2], height_ratios=[1], wspace = 0.2)

        self.ax_lc = self.fig.add_subplot(gs[0, 0])
        self.ax_psd = self.fig.add_subplot(gs[0, 1])

        self.fig.suptitle(f'TIC {self.star_id} Full Light Curve')


        self.update_lightcurve()

        self.update_powerspectra()

        plt.show()

        #### Connect click event  --------------------------
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes != self.ax_lc:
            return


        match = self.sector_times[(self.sector_times['start_time'] <= event.xdata) & (self.sector_times['end_time'] >= event.xdata)]

        if len(match) == 0:
            return  # clicked outside any sector

        clicked_sector = int(match['Sector'].iloc[0])
        self.test_parameter = match


        # Toggle sector removal
        if clicked_sector in self.removed_sectors:
            self.removed_sectors.remove(clicked_sector)
            print(f"Restored sector {clicked_sector}")
        else:
            self.removed_sectors.add(clicked_sector)
            print(f"Removed sector {clicked_sector}")

        self.update_lightcurve()

        self.update_powerspectra()

        self.fig.canvas.draw_idle()


    def update_lightcurve(self):
        self.ax_lc.clear()

        mask = ~np.isin(self.sector_list, list(self.removed_sectors))

        ## Plot full light curve
        self.lc_list = []
        for file in self.lc_files:
            with open(file, 'r') as f:
                lc_temp = f2lk.txt_to_LC(file)
                self.lc_list.append(lc_temp)

        lc_collection = lk.LightCurveCollection(self.lc_list)
        lc_full = lc_collection.stitch(lambda x: x )

        lc_full.plot(ax=self.ax_lc, c = 'k', alpha = 0.2);

        # self.ax_lc.relim()
        # self.ax_lc.autoscale_view()

        xmin_default, xmax_default = np.min(lc_full.time.value), np.max(lc_full.time.value)
        dt = 50  # in days. pad the xlims
        self.ax_lc.set_xlim(xmin_default-dt, xmax_default+dt)

        for idx,sec in self.sector_times.iterrows():
            self.ax_lc.axvline(sec.start_time, ls = 'dashed', c = 'r', alpha = 0.5)


            if int(sec.Sector) in self.sector_list:
                x = sec['mid_time']
                label = str(int(sec['Sector']))

                txt = self.ax_lc.text(
                    x, 1.02, label,
                    transform=self.ax_lc.get_xaxis_transform(),  # key line
                    ha='center',
                    va='bottom',
                    fontsize=8
                )




        ## Plot masked light curve
        self.lc_list = []
        for file in self.lc_files[mask]:
            with open(file, 'r') as f:
                lc_temp = f2lk.txt_to_LC(file)
                self.lc_list.append(lc_temp)

        lc_collection = lk.LightCurveCollection(self.lc_list)
        self.lc_all = lc_collection.stitch(lambda x: x )

        self.lc_all.plot(ax=self.ax_lc, c = 'k');



        return self.lc_all

    def update_powerspectra(self):
        self.ax_psd.clear()

        self.psd_all = TESS_LC.calc_PSD(self.lc_all, method = 'setfreqres', oversample=1, min_freq=0.01)
        pssm = TESS_LC.ps_smooth(self.psd_all.frequency.value, self.psd_all.power.value, self.numax_guess, 'Yu18', 2)

        self.psd_all.plot(ax=self.ax_psd, c = 'k')
        self.ax_psd.plot(self.psd_all.frequency.value, pssm, lw = 3, c = 'mediumpurple')
        self.ax_psd.set_xlim(1,200)
        self.ax_psd.set_yscale('log'), self.ax_psd.set_xscale('log')
        self.ax_psd.axvline(self.numax_guess, c = 'r', ls = 'dashed', alpha = 0.5)

        return self.psd_all

    def save_figure(self, filename="tess_lc.png", dpi=300):
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {filename}")
