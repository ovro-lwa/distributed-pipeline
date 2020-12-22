from typing import List

import ipywidgets as widgets
import matplotlib.pyplot as plt
from orca.utils import fitsutils
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from matplotlib.colors import Normalize

WIDTH = 128


class SiftingWidget(widgets.HBox):
    def __init__(self, labels: List[str]):
        super().__init__()
        # TODO add start_time and end_time
        self.init_single_scan(
            '/lustre/yuping/0-100-hr-reduction/final-narrow/sidereal_narrow_diff/2018-03-21/hh=01/diff_2018-03-21T01:38:41-image_sfind.npz',
            '/lustre/yuping/0-100-hr-reduction/final-narrow/sidereal_narrow_diff/2018-03-21/hh=01/diff_2018-03-21T01:38:41-image.fits',
            '/lustre/yuping/0-100-hr-reduction/final-narrow/narrow_long/2018-03-21/hh=01/2018-03-21T01:38:41-image.fits',
            '/lustre/yuping/0-100-hr-reduction/final-narrow/narrow_long/2018-03-22/hh=01/2018-03-22T01:34:45-image.fits')

        snapshot = widgets.Output()
        cutouts = widgets.Output()
        spectrum = widgets.Output()
        buttons = self.make_buttons(labels)
        self.text = widgets.HTML()

        with snapshot:
            fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 8), dpi=72)
            fig1.canvas.header_visible = False
            ax1.set_frame_on(False)
            ax1.tick_params(labelbottom=False, labelleft=False, length=0)
            ax1.imshow(self.diff_im, norm=Normalize(-2, 5), origin='lower')

        # TODO swap for a vbox of three plots so that I can lock the colorscale?
        with cutouts:
            self.fig2, axs2 = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3.5), dpi=72)
            self.fig2.canvas.header_visible = False
            for ax in axs2:
                ax.set_frame_on(False)
                ax.tick_params(labelbottom=False, labelleft=False, length=0)
            self.diff_imshow = axs2[0].imshow(
                self.diff_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
                self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH].T)
            self.before_imshow = axs2[1].imshow(
                self.before_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
                self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH].T)
            self.after_imshow = axs2[2].imshow(
                self.after_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
                self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH].T)
            self.load_text()

        with spectrum:
            fig3, ax3 = plt.subplots(constrained_layout=True, figsize=(10, 3), dpi=72)
            fig3.canvas.header_visible = False

        self.children = [snapshot, widgets.VBox([cutouts, self.text, spectrum]), buttons]

    def init_single_scan(self, catalog, diff, before, after):
        self.cat = np.load(catalog)
        self.diff_im, self.header = fitsutils.read_image_fits(diff)
        self.before_im, _ = fitsutils.read_image_fits(before)
        self.after_im, _ = fitsutils.read_image_fits(after)
        self.curr = 0

    def make_buttons(self, labels):
        buttons = [widgets.Button(description=label) for label in labels]
        skip = widgets.Button(description='skip')
        skip.on_click(self.update)
        buttons.append(skip)
        return widgets.VBox(buttons)

    def update(self, b):
        print(b.description)
        self.curr += 1
        if b.description != 'skip':
            # should save the label
            print(f'label {b.description}')
        self.load_mpl_im()
        self.fig2.canvas.draw()
        self.load_text()

    def load_mpl_im(self):
        self.diff_imshow.set_data(
            self.diff_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
            self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH])
        self.before_imshow.set_data(
            self.before_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
            self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH])
        self.after_imshow.set_data(
            self.after_im.T[self.cat['xpos_abs'][self.curr] - WIDTH: self.cat['xpos_abs'][self.curr] + WIDTH,
            self.cat['ypos_abs'][self.curr] - WIDTH: self.cat['ypos_abs'][self.curr] + WIDTH])

    def load_text(self):
        coord = SkyCoord(ra=self.cat['ra_abs'][self.curr] * u.deg, dec=self.cat['dec_abs'][self.curr] * u.deg)
        self.text.value = f"{coord.ra.to_string(u.hour)} {coord.dec.to_string()} " \
                          f"x={self.cat['xpos_abs'][self.curr]}, y={self.cat['ypos_abs'][self.curr]} " \
                          f"{self.cat['bmaj_abs'][self.curr] * 60:.2f}' x {self.cat['bmin_abs'][self.curr] * 60:.2f}' " \
                          f"pk={self.cat['pkflux_abs'][self.curr]:.1f} Jy"

    def save_and_load_next_scan(self):
        pass
