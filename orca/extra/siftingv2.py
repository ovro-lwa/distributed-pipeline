import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from datetime import datetime

from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from astropy import wcs

# Reference sources
S1 = SkyCoord('13h49m39.28s', '+21deg07m28.2s', frame=ICRS)
S2 = SkyCoord('13h57m4.71s',  '+19deg19m7.7s', frame=ICRS)
S3 = SkyCoord('13h54m40.61s', '+16deg14m44.9s', frame=ICRS)

class TauBooSearch(widgets.HBox):
    def __init__(self, start_date: datetime, end_time: datetime, cutout_width=125):
        super().__init__()

        
        cutouts = widgets.Output()
        with cutouts:
            self.fig2, axs2 = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 3.5), dpi=72)
            self.fig2.canvas.header_visible = False
            peak = np.abs(self.cat['peak_flux'][self.curr])
            norm = Normalize(vmin=-peak, vmax=peak)
            for ax in axs2:
                ax.set_frame_on(False)
                ax.tick_params(labelbottom=False, labelleft=False, length=0)
            self.diff_imshow = axs2[0].imshow(
                self.diff_im.T[self.cat['x'][self.curr] - cutout_width : self.cat['x'][self.curr] + cutout_width ,
                self.cat['y'][self.curr] - cutout_width : self.cat['y'][self.curr] + cutout_width ].T,
                cmap='gray', norm=norm, origin='lower')
            self.before_imshow = axs2[1].imshow(
                self.before_im.T[self.cat['x'][self.curr] - cutout_width : self.cat['x'][self.curr] + cutout_width ,
                self.cat['y'][self.curr] - cutout_width : self.cat['y'][self.curr] + cutout_width ].T,
                cmap='gray', norm=norm, origin='lower')
            self.after_imshow = axs2[2].imshow(
                self.after_im.T[self.cat['x'][self.curr] - cutout_width : self.cat['x'][self.curr] + cutout_width ,
                self.cat['y'][self.curr] - cutout_width : self.cat['y'][self.curr] + cutout_width ].T,
                cmap='gray', norm=norm, origin='lower')
            reticle_style_kwargs = {'linewidth': 2, 'color': 'm'}
            inner, outer = 0.02, 0.07
            for a in axs2:
                a.axvline(xcutout_width =, ymin=0.5 - inner, ymax=0.5 - outer,
                          **reticle_style_kwargs)
                a.axhline(ycutout_width =, xmin=0.5 - inner, xmax=0.5 - outer,
                          **reticle_style_kwargs)
            self.load_text()

        plot_control = self._make_plot_control()
        buttons = self._make_buttons()
        self.children = [cutouts, plot_control, buttons]

    def _make_plot_control(self):
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
        return plot_control

    def _make_buttons(self):
        next_button = widgets.Button(description='next')
        next_button.on_click(self._next_frame)
        back_button = widgets.Button(description='back')
        back_button.on_click(self._prev_frame)
        return widgets.VBox([next_button, back_button])