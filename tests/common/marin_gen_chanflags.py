"""
Initially written by Marin Anderson
"""
# outstanding problems:
# 1. flags XY together
# 2. if subband missing, will fill in with nan values

import numpy as np
import pyrap.tables as pt
import sys
import os
from scipy.ndimage import filters
from orca.util.constants import LWA_N_ANT, LWA_N_CHAN, LWA_N_SPW


def generate_chan_flags(ms_name):
    ms_table = pt.table(ms_name)
    spw_table = pt.table(ms_name + '/SPECTRAL_WINDOW')
    autocorr_table = ms_table.query('ANTENNA1=ANTENNA2')

    # iterate over antennas
    ant_flagged = []
    ant_good = []
    amp_all_ants = np.zeros((LWA_N_ANT, LWA_N_SPW * LWA_N_CHAN))
    freq_all_bands = np.zeros(LWA_N_SPW * LWA_N_CHAN)
    for ant_ind, tant in enumerate(autocorr_table.iter("ANTENNA1")):
        ampXallbands = np.zeros(LWA_N_SPW * LWA_N_CHAN)
        ampYallbands = np.zeros(LWA_N_SPW * LWA_N_CHAN)
        tmpind = 0
        # iterate over subbands
        # (the tedious if statements are to correct for missing subbands in the ms file)
        for ant_ind, tband in enumerate(tant):
            if ant_ind != 0:
                if spw_table[ant_ind]['REF_FREQUENCY'] == spw_table[ant_ind - 1]['REF_FREQUENCY']:
                    continue
                elif spw_table[ant_ind]['REF_FREQUENCY'] != (spw_table[ant_ind - 1]['REF_FREQUENCY'] + spw_table[ant_ind - 1]['TOTAL_BANDWIDTH']):
                    numpad = int((spw_table[ant_ind]['REF_FREQUENCY'] - spw_table[ant_ind - 1]['REF_FREQUENCY']) /
                                 spw_table[ant_ind - 1]['TOTAL_BANDWIDTH']) - 1
                    amppad = np.zeros(LWA_N_CHAN * numpad).fill(np.nan)
                    ampXallbands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + numpad)] = amppad
                    ampYallbands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + numpad)] = amppad
                    if ant_ind == 0:
                        frqpadstart = spw_table[ant_ind - 1]['REF_FREQUENCY'] + spw_table[ant_ind - 1]['TOTAL_BANDWIDTH'] + \
                                      spw_table[ant_ind - 1]['EFFECTIVE_BW'][0] / 2.
                        frqpadend = frqpadstart + (spw_table[ant_ind - 1]['TOTAL_BANDWIDTH'] * numpad)
                        frqpad = np.linspace(frqpadstart, frqpadend + spw_table[ant_ind - 1]['EFFECTIVE_BW'][0] / 2.,
                                             spw_table[ant_ind - 1]['NUM_CHAN'] * numpad)
                        freq_all_bands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + numpad)] = frqpad
                    tmpind += numpad
            ampX = np.absolute(tband["DATA"][:, 0])
            ampY = np.absolute(tband["DATA"][:, 3])
            ampXallbands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + 1)] = ampX
            ampYallbands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + 1)] = ampY
            if ant_ind == 0:
                freq = spw_table[ant_ind]['CHAN_FREQ']
                freq_all_bands[tmpind * LWA_N_CHAN:LWA_N_CHAN * (tmpind + 1)] = freq
            tmpind += 1
        amp_all_ants[ant_ind, :] = (ampXallbands + ampYallbands) / 2.
        # add antenna to antflags list if median amplitude < 1.e9
        if np.sum(np.isnan(amp_all_ants[ant_ind])):
            ant_flagged.append(ant_ind)
        else:
            ant_good.append(ant_ind)
    # average over individual antennas
    ampavgants = np.mean(amp_all_ants[ant_good], axis=0)
    ampstdants = np.std(amp_all_ants[ant_good], axis=0)
    # median filter over average bandpass
    ampmedfilt = filters.median_filter(ampavgants, size=LWA_N_CHAN)

    """
    plt.figure(figsize=(15, 5), edgecolor='Black')
    plt.plot(freq_all_bands, ampmedfilt)
    plt.plot(freq_all_bands, ampavgants + 4.e9)
    plt.ylim([0, 8.e9])
    plt.show()
    """

    # generate flags (for now, flagging XY together)
    antchanflags = np.zeros((LWA_N_ANT, LWA_N_SPW * LWA_N_CHAN), dtype=bool)
    for ant_ind, antband in enumerate(amp_all_ants):
        # median filter over antenna band
        antmedfilt = filters.median_filter(antband, size=LWA_N_CHAN)
        # divide average antenna median filter by single antenna median filter and take median
        # if a single frequency channel is N times larger than median * average antenna median filter, flag frequency channel
        # if more than 2/3 of antenna channels are flagged, flag the entire antenna
        # plt.figure(figsize=(15,5),edgecolor='Black')
        # plt.plot(freqallbands,antband/antmedfilt)
        # plt.plot(freqallbands,antband/ampmedfilt)
        # plt.show()
        # subfilt = ampmedfilt - antmedfilt
        for frqind, antchan in enumerate(antband):
            # if 10*np.log10((antchan+subfilt[frqind])/ampmedfilt[frqind]) > 4. or 10*np.log10(ampmedfilt[frqind]/(antchan+subfilt[frqind])) > 4.:
            # if 10*np.log10(antchan/antmedfilt[frqind]) > 2.:
            if 10 * np.log10(antchan / antmedfilt[frqind]) > 1:
                antchanflags[ant_ind, frqind] = False
            else:
                antchanflags[ant_ind, frqind] = True
        """
        plt.figure(figsize=(15, 5), edgecolor='Black')
        plt.plot(freq_all_bands[antchanflags[ant_ind, :]], antband[antchanflags[ant_ind, :]])
        plt.plot(freq_all_bands, antband + 1.e9)
        plt.plot(freq_all_bands, ampmedfilt)
        plt.plot(freq_all_bands, antmedfilt + 2.5e9)
        plt.title('%03d' % (ant_ind + 1))
        plt.ylim([0, 8.e9])
        plt.show()
        plt.figure(figsize=(15, 5), edgecolor='Black')
        plt.plot(freq_all_bands[antchanflags[ant_ind, :]],
                 antband[antchanflags[ant_ind, :]] / ampmedfilt[antchanflags[ant_ind, :]])
        plt.show()
        """
        return antchanflags, ant_flagged
