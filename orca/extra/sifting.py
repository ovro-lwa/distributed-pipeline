from typing import List, Tuple, Optional, Union
from datetime import datetime, timedelta
import numpy as np

import ipywidgets as widgets
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from matplotlib.colors import Normalize
import logging

from orca.extra.catalogutils import add_id_column
from orca.extra.classes import Classes
from orca.utils import fitsutils, coordutils, gitutils
from orca.metadata.pathsmanagers import OfflinePathsManager

WIDTH = 128

SNR_CUTOFF = 6.5

PERSISTENT_CAT = '/home/yuping/catalog-scripts/persistent_source_cat_with_flux.fits'

# Maximum distance for associating a source with a persistent source
MAX_ASSOC_DIST = 10 * u.arcmin


# TODO need a back button
class SiftingWidget(widgets.HBox):
    def __init__(self, catalogs: List[str], diff_ims: List[str],
                 before_ims: List[str], after_ims: List[str], outputs: List[str],
                 min_alt_deg: float = None, min_amplification: float = None,
                 max_dec_deg: Optional[Union[List[float], np.ndarray, Tuple[float]]] = None):
        """

        Args:
            catalogs: List of catalog fits files.
            diff_ims: Lit of differenced images, ordered as catalogs.
            before_ims: List of before_ims, ordered as catalogs
            after_ims: List of after_ims, ordered as catalogs
            outputs: List of output catalog names, ordered as catalogs
            min_alt_deg: minimum altitude angle cutoff in degrees. None means not doing cutoff.
            min_amplification: Factor with which a persistent source should brighten before being considered a
                candidate. None means not looking at the persistent source catalog.
            max_dec_deg: A list of maximum declinations, i.e. minimum distances from the NCP,
                one for each scan. None means not doing NCP masking.
        """
        super().__init__()
        self.min_alt_deg = min_alt_deg
        self.min_brightening_factor = min_amplification
        if self.min_brightening_factor is not None:
            self.persist_cat = Table.read(PERSISTENT_CAT)
            self.persist_cat_coords = SkyCoord(self.persist_cat['ra'], self.persist_cat['dec'], unit=u.deg)
        if max_dec_deg is not None:
            try:
                assert len(max_dec_deg) == len(diff_ims), \
                    'max_dec_deg and diff_ims (and other lists) should have the same len.'
            except TypeError:
                # It's a float
                pass
        self.max_dec_deg = max_dec_deg
        self.catalogs = catalogs
        self.diff_ims = diff_ims
        self.before_ims = before_ims
        self.after_ims = after_ims
        self.outputs = outputs

        self.curr_scan = 0
        self.cat, self.coords, self.alt_deg = self._load_catalog(self.catalogs[self.curr_scan])
        self.diff_im, self.header = fitsutils.read_image_fits(self.diff_ims[self.curr_scan])
        self.before_im, _ = fitsutils.read_image_fits(self.before_ims[self.curr_scan])
        self.after_im, _ = fitsutils.read_image_fits(self.after_ims[self.curr_scan])
        self.curr = 0
        snapshot = widgets.Output()
        cutouts = widgets.Output()
        spectrum = widgets.Output()
        buttons = self.make_buttons()
        self.text = widgets.HTML()

        rms = np.median(np.unique(self.cat['local_rms']))
        with snapshot:
            self.fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 8), dpi=72)
            self.fig1.canvas.header_visible = False
            ax1.set_frame_on(False)
            ax1.tick_params(labelbottom=False, labelleft=False, length=0)
            self.big_diff_imshow = ax1.imshow(
                self.diff_im, norm=Normalize(-6 * rms, 6 * rms), cmap='gray', origin='lower')

        # TODO swap for a vbox of three plots so that I can lock the colorscale?
        with cutouts:
            self.fig2, axs2 = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3.5), dpi=72)
            self.fig2.canvas.header_visible = False
            peak = np.abs(self.cat['peak_flux'][self.curr])
            norm = Normalize(vmin=-peak, vmax=peak)
            for ax in axs2:
                ax.set_frame_on(False)
                ax.tick_params(labelbottom=False, labelleft=False, length=0)
            self.diff_imshow = axs2[0].imshow(
                self.diff_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
                self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T,
                cmap='gray', norm=norm, origin='lower')
            self.before_imshow = axs2[1].imshow(
                self.before_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
                self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T,
                cmap='gray', norm=norm, origin='lower')
            self.after_imshow = axs2[2].imshow(
                self.after_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
                self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T,
                cmap='gray', norm=norm, origin='lower')
            reticle_style_kwargs = {'linewidth': 2, 'color': 'm'}
            inner, outer = 0.02, 0.07
            for a in axs2:
                a.axvline(x=WIDTH, ymin=0.5 - inner, ymax=0.5 - outer,
                          **reticle_style_kwargs)
                a.axhline(y=WIDTH, xmin=0.5 - inner, xmax=0.5 - outer,
                          **reticle_style_kwargs)
            self.load_text()

        self.vmin_ctrl = widgets.Text(value=f'{-peak:.1f}', placeholder='-10', description='vmin:',
                                      disabled=False, layout=widgets.Layout(width='20%'), continuous_update=False)
        self.vmin_ctrl.observe(self._update_vmin, names='value')
        self.vmax_ctrl = widgets.Text(value=f'{peak:.1f}', placeholder='10', description='vmax:',
                                      disabled=False, layout=widgets.Layout(width='20%'), continuous_update=False)
        self.vmax_ctrl.observe(self._update_vmax, names='value')
        self.cmap_ctrl = widgets.Dropdown(value='gray', options=['gray', 'gray_r', 'viridis', 'viridis_r'],
                                          description='cmap:', disable=False, layout=widgets.Layout(width='20%'))
        self.cmap_ctrl.observe(self._update_cmap, names='value')
        plot_control = widgets.HBox([self.vmin_ctrl, self.vmax_ctrl, self.cmap_ctrl])

        with spectrum:
            fig3, ax3 = plt.subplots(constrained_layout=True, figsize=(10, 3), dpi=72)
            fig3.canvas.header_visible = False

        self.children = [snapshot, widgets.VBox([cutouts, plot_control, self.text, spectrum]), buttons]

    def make_buttons(self):
        buttons = [widgets.Button(description=name) for name, _ in Classes.__members__.items()]
        skip = widgets.Button(description='skip')
        buttons.append(skip)
        for b in buttons:
            b.on_click(self.update)
        back = widgets.Button(description='back')
        back.on_click((self._back))
        buttons.append(back)
        return widgets.VBox(buttons)

    def update(self, b):
        if b.description != 'skip':
            self.cat['class'][self.curr] = Classes[b.description.upper()]
        self.curr += 1
        if self.curr == len(self.cat):
            self._init_next_scan()
        self.load_mpl_im()
        self.fig2.canvas.draw()
        self.load_text()

    def _update_vmin(self, change):
        try:
            norm = Normalize(vmin=float(change['new']), vmax=float(self.vmax_ctrl.value))
            self.diff_imshow.set_norm(norm)
            self.before_imshow.set_norm(norm)
            self.after_imshow.set_norm(norm)
            self.fig2.canvas.draw()
        except Exception:
            logging.exception('Exception while updating vmin.')

    def _update_vmax(self, change):
        try:
            norm = Normalize(vmin=float(self.vmin_ctrl.value), vmax=float(change['new']))
            self.diff_imshow.set_norm(norm)
            self.before_imshow.set_norm(norm)
            self.after_imshow.set_norm(norm)
            self.fig2.canvas.draw()
        except Exception:
            logging.exception('Exception while updating vmax.')

    def _update_cmap(self, change):
        self.diff_imshow.set_cmap(change['new'])
        self.before_imshow.set_cmap(change['new'])
        self.after_imshow.set_cmap(change['new'])
        self.fig2.canvas.draw()

    def _back(self, b):
        if self.curr == 0:
            # if first cand of scan, go back 2 scans, load next scan. Need to re-open the output cat file.
            self._init_prev_scan()
            self.curr = len(self.cat) - 1
        else:
            self.curr -= 1

        self.load_mpl_im()
        self.fig2.canvas.draw()
        self.load_text()
        self.cat['class'][self.curr] = Classes.NA

    def _init_next_scan(self):
        while True:
            if self.curr_scan > -1:
                self._save_curr_catalog(self.outputs[self.curr_scan])
            self._load_scan_data(increment=1)
            if len(self.cat) == 0:
                logging.info(f'{self.catalogs[self.curr_scan]} has 0 filtered candidates, skipping.')
                self._save_curr_catalog(self.outputs[self.curr_scan])
            else:
                break
        self.fig1.canvas.draw()

    def _init_prev_scan(self):
        while True:
            self._load_scan_data(increment=-1)
            self.cat = Table.read(self.outputs[self.curr_scan])
            assert len(self.cat) == len(self.alt_deg), "I did not get the back button correct."
            if len(self.cat) == 0:
                logging.info(f'{self.catalogs[self.curr_scan]} has 0 filtered candidates, skipping.')
            else:
                break
        self.fig1.canvas.draw()

    def _load_scan_data(self, increment: int):
        self.curr_scan += increment
        self.cat, self.coords, self.alt_deg = self._load_catalog(self.catalogs[self.curr_scan])
        self.diff_im, self.header = fitsutils.read_image_fits(self.diff_ims[self.curr_scan])
        self.before_im, _ = fitsutils.read_image_fits(self.before_ims[self.curr_scan])
        self.after_im, _ = fitsutils.read_image_fits(self.after_ims[self.curr_scan])
        rms = np.median(np.unique(self.cat['local_rms']))
        self.big_diff_imshow.set_data(self.diff_im)
        self.big_diff_imshow.set_norm(Normalize(vmin=-6 * rms, vmax=6 * rms))
        self.curr = 0

    def load_mpl_im(self):
        peak = np.abs(self.cat['peak_flux'][self.curr])
        norm = Normalize(vmin=-peak, vmax=peak)
        self.diff_imshow.set_data(
            self.diff_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.before_imshow.set_data(
            self.before_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.after_imshow.set_data(
            self.after_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.diff_imshow.set_norm(norm)
        self.before_imshow.set_norm(norm)
        self.after_imshow.set_norm(norm)
        self.diff_imshow.set_cmap('gray')
        self.before_imshow.set_cmap('gray')
        self.after_imshow.set_cmap('gray')

        self.vmax_ctrl.unobserve(self._update_vmax, names='value')
        self.vmin_ctrl.unobserve(self._update_vmin, names='value')
        self.cmap_ctrl.unobserve(self._update_cmap, names='value')
        self.vmax_ctrl.value = f'{peak:.1f}'
        self.vmin_ctrl.value = f'{(-peak):.1f}'
        self.cmap_ctrl.value = 'gray'
        self.vmax_ctrl.observe(self._update_vmax, names='value')
        self.vmin_ctrl.observe(self._update_vmin, names='value')
        self.cmap_ctrl.observe(self._update_cmap, names='value')

    def load_text(self):
        coord = self.coords[self.curr]
        self.text.value = f"{self.cat.meta['DATE']} ---" \
                          f"{self.curr + 1}/{len(self.cat)} candidates, " \
                          f"{self.curr_scan + 1}/{len(self.catalogs)} scans --- " \
                          f"{coord.ra.to_string(u.hour, sep=' ', precision=1)}, "\
                          f"{coord.dec.to_string(sep=' ', precision=1)} " \
                          f"<br> alt={self.alt_deg[self.curr]:.1f} deg " \
                          f"x={self.cat['x'][self.curr]}, y={self.cat['y'][self.curr]} " \
                          f"{self.cat['a'][self.curr] * 60:.1f}' x {self.cat['b'][self.curr] * 60:.1f}' " \
                          f"pk={self.cat['peak_flux'][self.curr]:.1f} Jy, " \
                          f"noise={self.cat['local_rms'][self.curr]:.2f} Jy"

    def _load_catalog(self, cat_fits) -> Tuple[Table, SkyCoord, np.ndarray]:
        t = Table.read(cat_fits)
        add_id_column(t)
        t = t[t['peak_flux']/t['local_rms'] > SNR_CUTOFF]
        # The split gets rid of the fractional second
        timestamp = datetime.strptime(t.meta['DATE'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
        coords = SkyCoord(t['ra'], t['dec'], unit=u.deg)
        alt = coordutils.get_altaz_at_ovro(coords, timestamp).alt
        mask = np.ones_like(alt.value, dtype=bool)
        if self.min_alt_deg:
            mask &= ((alt.to(u.deg).value > self.min_alt_deg) & (~np.isnan(alt.value)))
        if self.min_brightening_factor is not None:
            idx, sep2d, _ = coords.match_to_catalog_sky(self.persist_cat_coords)
            coincidence_mask = ((sep2d > MAX_ASSOC_DIST) |
                                (t['peak_flux'] / self.persist_cat['peak_flux'][idx] > self.min_brightening_factor))
            logging.info(f'{np.sum(~coincidence_mask)} sources in persistent source catalog.')
            mask &= coincidence_mask
        if self.max_dec_deg:
            mask &= (coords.dec.to(u.deg).value < self.max_dec_deg[self.curr_scan])
        t = t[mask]
        alt = alt[mask]
        coords = coords[mask]
        t.add_column(Classes.NA.value, name='class')
        t.meta['COMMIT'] = gitutils.get_commit_id()
        return t, coords, alt.to(u.deg).value

    def _save_curr_catalog(self, cat_fits):
        # Save table to another place
        self.cat.write(cat_fits, overwrite=True)


class OfflineSifter(SiftingWidget):
    def __init__(self, pm: OfflinePathsManager, start_time: datetime, end_time: datetime, interval: timedelta,
                 subtraction_interval: timedelta,
                 diff: str, before: str, after: str, output_suffix: str, min_alt_deg: Optional[float] = None,
                 min_amplification: Optional[float] = None, max_dec_deg: Optional[float] = None):
        """

        Args:
            pm:
            start_time:
            end_time:
            interval:
            subtraction_interval:
            diff:
            before:
            after:
            output_suffix: For example '_sfind_sift.fits'
            min_alt_deg:
        """
        # Or use the first scan to anchor the thing. Also need integration time though.
        times = pm.utc_times_mapping.keys()
        assert start_time in times, 'start_time must be a valid scan time.'
        assert start_time + interval in times, 'interval must be a multiple of the snapshot integration time.'
        assert start_time != end_time, 'To look at one integration, add a small time delta to end_time (say, 1s).'
        assert output_suffix.endswith('.fits'), 'output_suffix needs to end with .fits.'
        ts = start_time
        catalogs, diff_ims, before_ims, after_ims, outputs = [], [], [], [], []
        while ts < end_time:
            catalogs.append(pm.dpp(ts, diff, '_sfind.fits', 'diff'))
            diff_ims.append(pm.dpp(ts, diff, '.fits', 'diff'))
            before_ims.append(pm.dpp(ts, before, '.fits'))
            after_ims.append(pm.dpp(ts + subtraction_interval, after, '.fits'))
            outputs.append(pm.dpp(ts, diff, output_suffix, 'diff'))
            ts += interval

        super(OfflineSifter, self).__init__(catalogs, diff_ims, before_ims, after_ims, outputs,
                                            min_alt_deg=min_alt_deg, min_amplification=min_amplification,
                                            max_dec_deg=max_dec_deg)
