from typing import List
from datetime import datetime, timedelta
import numpy as np

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from matplotlib.colors import Normalize

from orca.extra.catalogutils import add_id_column
from orca.extra.classes import Classes
from orca.utils import fitsutils, coordutils, gitutils
from orca.metadata.pathsmanagers import OfflinePathsManager

WIDTH = 128

SNR_CUTOFF = 6.5


# TODO need a back button
class SiftingWidget(widgets.HBox):
    def __init__(self, catalogs: List[str], diff_ims: List[str],
                 before_ims: List[str], after_ims: List[str], outputs: List[str], min_alt_deg: float = None):
        super().__init__()
        self.min_alt_deg = min_alt_deg
        self.catalogs = catalogs
        self.diff_ims = diff_ims
        self.before_ims = before_ims
        self.after_ims = after_ims
        self.outputs = outputs

        self.curr_scan = 0
        self.cat = self._load_catalog(self.catalogs[self.curr_scan])
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
            self.load_text()

        with spectrum:
            fig3, ax3 = plt.subplots(constrained_layout=True, figsize=(10, 3), dpi=72)
            fig3.canvas.header_visible = False

        self.children = [snapshot, widgets.VBox([cutouts, self.text, spectrum]), buttons]

    def make_buttons(self):
        buttons = [widgets.Button(description=name) for name, _ in Classes.__members__.items()]
        skip = widgets.Button(description='skip')
        buttons.append(skip)
        for b in buttons:
            b.on_click(self.update)
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

    def _init_next_scan(self):
        if self.curr_scan > -1:
            self._save_curr_catalog(self.outputs[self.curr_scan])
        self.curr_scan += 1
        self.cat = self._load_catalog(self.catalogs[self.curr_scan])
        self.diff_im, self.header = fitsutils.read_image_fits(self.diff_ims[self.curr_scan])
        self.before_im, _ = fitsutils.read_image_fits(self.before_ims[self.curr_scan])
        self.after_im, _ = fitsutils.read_image_fits(self.after_ims[self.curr_scan])
        rms = np.median(np.unique(self.cat['local_rms']))
        self.big_diff_imshow.set_data(self.diff_im)
        self.big_diff_imshow.set_norm(Normalize(vmin=-6 * rms, vmax=6 * rms))
        self.fig1.canvas.draw()
        self.curr = 0

    def load_mpl_im(self):
        peak = np.abs(self.cat['peak_flux'][self.curr])
        norm = Normalize(vmin=-peak, vmax=peak)
        self.diff_imshow.set_data(
            self.diff_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.diff_imshow.set_norm(norm)
        self.before_imshow.set_data(
            self.before_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.before_imshow.set_norm(norm)
        self.after_imshow.set_data(
            self.after_im.T[self.cat['x'][self.curr] - WIDTH: self.cat['x'][self.curr] + WIDTH,
            self.cat['y'][self.curr] - WIDTH: self.cat['y'][self.curr] + WIDTH].T)
        self.after_imshow.set_norm(norm)

    def load_text(self):
        coord = SkyCoord(ra=self.cat['ra'][self.curr] * u.deg, dec=self.cat['dec'][self.curr] * u.deg)
        self.text.value = f"{self.cat.meta['DATE']} <br>" \
                          f"{self.curr + 1}/{len(self.cat)} candidates, {self.curr_scan + 1}/{len(self.catalogs)} scans" \
                          f"<br> {coord.ra.to_string(u.hour)} {coord.dec.to_string()} " \
                          f"x={self.cat['x'][self.curr]}, y={self.cat['y'][self.curr]} " \
                          f"{self.cat['a'][self.curr] * 60:.2f}' x {self.cat['b'][self.curr] * 60:.2f}' " \
                          f"pk={self.cat['peak_flux'][self.curr]:.1f} Jy, " \
                          f"noise={self.cat['local_rms'][self.curr]:.2f} Jy"

    def _load_catalog(self, cat_fits) -> Table:
        t = Table.read(cat_fits)
        add_id_column(t)
        if self.min_alt_deg:
            # The split gets rid of the fractional second
            timestamp = datetime.strptime(t.meta['DATE'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
            alt = coordutils.get_altaz_at_ovro(SkyCoord(t['ra'], t['dec'], unit=u.deg), timestamp).alt
            t = t[(alt > (self.min_alt_deg * u.deg)) & (t['peak_flux']/t['local_rms'] > SNR_CUTOFF)]
        t.add_column(Classes.NA.value, name='class')
        t.meta['COMMIT'] = gitutils.get_commit_id()
        return t

    def _save_curr_catalog(self, cat_fits):
        # Save table to another place
        self.cat.write(cat_fits)


class OfflineSifter(SiftingWidget):
    def __init__(self, pm: OfflinePathsManager, start_time: datetime, end_time: datetime, interval: timedelta,
                 subtraction_interval: timedelta,
                 diff: str, before: str, after: str, min_alt_deg: float = None):
        # Or use the first scan to anchor the thing. Also need integration time though.
        times = pm.utc_times_mapping.keys()
        assert start_time in times, 'start_time must be a valid scan time.'
        assert start_time + interval in times, 'interval must be a multiple of the snapshot integration time.'
        assert start_time != end_time, 'To look at one integration, add a small time delta to end_time (say, 1s).'
        ts = start_time
        catalogs, diff_ims, before_ims, after_ims, outputs = [], [], [], [], []
        while ts < end_time:
            catalogs.append(pm.dpp(ts, diff, '_sfind.fits', 'diff'))
            diff_ims.append(pm.dpp(ts, diff, '.fits', 'diff'))
            before_ims.append(pm.dpp(ts, before, '.fits'))
            after_ims.append(pm.dpp(ts + subtraction_interval, after, '.fits'))
            outputs.append(pm.dpp(ts, diff, '_sfind_sift.fits', 'diff'))
            ts += interval

        super(OfflineSifter, self).__init__(catalogs, diff_ims, before_ims, after_ims, outputs, min_alt_deg)
