"""Peeling related transforms
"""
import logging

from orca.utils.sourcemodels import RFI_B, CYG_A_UNPOLARIZED_RESOLVED, CAS_A_UNPOLARIZED_RESOLVED
from orca.wrapper import ttcal
from typing import List, Optional
from datetime import datetime
import tempfile
from os import path
import json
import numpy as np

import casacore.tables as tables
from orca.utils import coordutils
from orca.utils.calibrationutils import gen_model_ms_stokes
from casatasks import ft, bandpass, polcal, applycal

log = logging.getLogger(__name__)
CORRECTED_DATA = 'CORRECTED_DATA'
DATA = 'DATA'


def model_data_uvsub(msfile,msfilemodels,gain=1,add=False):
    msfileallmodels = path.splitext(msfile)[0]+'_allmodels.ms'
    msfilemodels_XX = path.splitext(msfilemodels)[0]+'XX.ms'
    msfilemodels_XY = path.splitext(msfilemodels)[0]+'XY.ms'
    msfilemodels_YX = path.splitext(msfilemodels)[0]+'YX.ms'
    msfilemodels_YY = path.splitext(msfilemodels)[0]+'YY.ms'
    # put back into MODEL_DATA of ms
    xx = tables.table(msfilemodels_XX)
    xy = tables.table(msfilemodels_XY)
    yx = tables.table(msfilemodels_YX)
    yy = tables.table(msfilemodels_YY)
    #xxmodel = xx.getcol('CORRECTED_DATA')
    xxmodel = xx.getcol('DATA')
    xymodel = xy.getcol('CORRECTED_DATA')
    yxmodel = yx.getcol('CORRECTED_DATA')
    #yymodel = yy.getcol('CORRECTED_DATA')
    yymodel = yy.getcol('DATA')
    t     = tables.table(msfile,readonly=False)
    data  = t.getcol('DATA')
    model = t.getcol('MODEL_DATA')
    model[:,:,0] = xxmodel[:,:,0]
    model[:,:,1] = xymodel[:,:,1]
    model[:,:,2] = yxmodel[:,:,2]
    model[:,:,3] = yymodel[:,:,3]
    if add:
        t.putcol('CORRECTED_DATA',data+model*gain)
    else:
        t.putcol('CORRECTED_DATA',data-model*gain)
    #t.putcol('MODEL_DATA',model*gain)
    #t.close()
    xx.close()
    xy.close()
    yx.close()
    yy.close()
    if not path.exists(msfileallmodels):
        tables.tablecopy(msfilemodels,msfileallmodels)
        tallmodels = tables.table(msfileallmodels,readonly=False)
        allmodels  = model*gain
    else:
        tallmodels = tables.table(msfileallmodels,readonly=False)
        oldmodels  = tallmodels.getcol('MODEL_DATA')
        allmodels  = oldmodels + model*gain
    tallmodels.putcol('MODEL_DATA',allmodels)
    t.putcol('MODEL_DATA',allmodels)
    t.close()
    tallmodels.close()


def model_data_rearrange(modelmsfile):
    tmodels = tables.table(modelmsfile,readonly=False)
    data    = tmodels.getcol('DATA')
    corr    = tmodels.getcol('CORRECTED_DATA')
    newdata = np.copy(data)
    tmodels.close()
    msfilemodels_XX = path.splitext(modelmsfile)[0]+'XX.ms'
    msfilemodels_XY = path.splitext(modelmsfile)[0]+'XY.ms'
    msfilemodels_YX = path.splitext(modelmsfile)[0]+'YX.ms'
    msfilemodels_YY = path.splitext(modelmsfile)[0]+'YY.ms'
    tables.tablecopy(modelmsfile,msfilemodels_XX)
    tables.tablecopy(modelmsfile,msfilemodels_XY)
    tables.tablecopy(modelmsfile,msfilemodels_YX)
    tables.tablecopy(modelmsfile,msfilemodels_YY)
    xx = tables.table(msfilemodels_XX,readonly=False)
    xy = tables.table(msfilemodels_XY,readonly=False)
    yx = tables.table(msfilemodels_YX,readonly=False)
    yy = tables.table(msfilemodels_YY,readonly=False)
    # XXmodel
    newdata[:,:,0] = corr[:,:,0]
    newdata[:,:,1] = corr[:,:,0]*1/data[:,:,0]*data[:,:,1]
    newdata[:,:,2] = corr[:,:,0]*1/data[:,:,0]*data[:,:,2]
    newdata[:,:,3] = corr[:,:,0]*1/data[:,:,0]*data[:,:,3]
    xx.putcol('DATA',newdata)
    xx.close()
    # XYmodel
    newdata[:,:,0] = corr[:,:,1]*1/data[:,:,1]*data[:,:,0]
    newdata[:,:,1] = corr[:,:,1]
    newdata[:,:,2] = corr[:,:,1]*1/data[:,:,1]*data[:,:,2]
    newdata[:,:,3] = corr[:,:,1]*1/data[:,:,1]*data[:,:,3]
    xy.putcol('DATA',newdata)
    xy.close()
    # YXmodel
    newdata[:,:,0] = corr[:,:,2]*1/data[:,:,2]*data[:,:,0]
    newdata[:,:,1] = corr[:,:,2]*1/data[:,:,2]*data[:,:,1]
    newdata[:,:,2] = corr[:,:,2]
    newdata[:,:,3] = corr[:,:,2]*1/data[:,:,2]*data[:,:,3]
    yx.putcol('DATA',newdata)
    yx.close()
    # YYmodel
    newdata[:,:,0] = corr[:,:,3]*1/data[:,:,3]*data[:,:,0]
    newdata[:,:,1] = corr[:,:,3]*1/data[:,:,3]*data[:,:,1]
    newdata[:,:,2] = corr[:,:,3]*1/data[:,:,3]*data[:,:,2]
    newdata[:,:,3] = corr[:,:,3]
    yy.putcol('DATA',newdata)
    yy.close()
    return msfilemodels_XX, msfilemodels_XY, msfilemodels_YX, msfilemodels_YY


def model_to_data(msfile):
    t      = tables.table(msfile, readonly=False)
    model_data = t.getcol('MODEL_DATA')
    t.putcol('DATA',model_data)
    t.close()
    return msfile


def cal_reverse(bcalfile: str, dcalfile: str):
    bcalfileflags = path.splitext(bcalfile)[0]+'_flags.bcal'
    tables.tablecopy(bcalfile,bcalfileflags)
    b      = tables.table(bcalfile, readonly=False)
    d      = tables.table(dcalfile, readonly=False)
    bgains = b.getcol('CPARAM')   # bgains.shape = (256,109,2)
    bflags = b.getcol('FLAG')
    dgains = d.getcol('CPARAM')
    dflags = d.getcol('FLAG')
    b.putcol('CPARAM',1/bgains)
    d.putcol('CPARAM',-dgains)
    b.close()
    d.close()
    bf            = tables.table(bcalfileflags, readonly=False)
    bgains[:,:,:] = 1
    #if np.shape(bflags)[1] != 109:
    #    bflags = np.repeat(bflags,int(np.ceil(109/np.shape(bflags)[1])),axis=1)[:,0:109,:]
    #    bgains = np.repeat(bgains,int(np.ceil(109/np.shape(bflags)[1])),axis=1)[:,0:109,:]
    bfflags       = bflags | dflags
    bf.putcol('CPARAM',bgains)
    bf.putcol('FLAG',bflags)
    bf.close()
    return bcalfile, dcalfile


def zest_with_casa(ms: str, reverse: bool = False):
    """
    Polarized peeling with CASA. Currently, peeling list will only include at most CasA & CygA.
    The peeled visibilities will be placed in CORRECTED_DATA.
    :param ms: The measurement set.
    :param reverse: Reverse the peeling process. Default is False.
    """
    cllist = gen_model_ms_stokes(ms, zest=True)
    for srcind, clfile in enumerate(cllist):
        dcalfile        = path.splitext(path.abspath(ms))[0]+'_'+str(srcind)+'.dcal'
        bcalfile        = path.splitext(path.abspath(ms))[0]+'_'+str(srcind)+'.bcal'
        bcalfileflags   = path.splitext(path.abspath(bcalfile))[0]+'_flags.bcal'
        modelmsfile     = path.splitext(path.abspath(ms))[0]+'_'+str(srcind)+'_models.ms'
        # move calibrated data from CORRECTED_DATA into DATA column
        t = tables.table(ms, readonly=False)
        corrected_data = t.getcol('CORRECTED_DATA')
        t.putcol('DATA',corrected_data)
        t.close()
        # FT component list
        ft(ms, complist=clfile, usescratch=True)
        # bandpass solve then polcal solve with poltype='Df'
        bandpass(ms, bcalfile, refant='34', uvrange='>15lambda', solint='inf,12ch')
        polcal(ms, dcalfile, refant='', uvrange='>15lambda', poltype='Df', gaintable=[bcalfile], solint='inf,12ch')
        # reorder both leakage and gain term tables
        cal_reverse(bcalfile,dcalfile)
        # put MODEL_DATA into DATA column of new measurement set, 
        # to run applycal like normal with gains
        tables.tablecopy(ms,modelmsfile)
        model_to_data(modelmsfile)
        applycal(modelmsfile, gaintable=[bcalfile], flagbackup=False)
        applycal(ms, gaintable=[bcalfileflags], flagbackup=False)
        # Reorder model measurement set for applying leakage gains
        msfilemodels_XX, msfilemodels_XY, msfilemodels_YX, msfilemodels_YY = model_data_rearrange(modelmsfile)
        applycal(msfilemodels_XY, gaintable=[dcalfile], flagbackup=False)
        applycal(msfilemodels_YX, gaintable=[dcalfile], flagbackup=False)
        # put back into MODEL_DATA of ms, uvsub from DATA, and put result in CORRECTED_DATA
        model_data_uvsub(ms, modelmsfile)
    return ms


def ttcal_peel_from_data_to_corrected_data(ms: str, utc_time: datetime, include_rfi_source: bool = True) -> str:
    """ Use TTCal to peel. Read from DATA column and write to CORRECTED_DATA
    If the CORRECTED_DATA column exists, it does not do anything.

    Args:
        ms: Path to measurement set.
        utc_time: datetime object to figure out what sources are up.
        include_rfi_source: Include near-field generic RFI sources in peel.

    Returns: The output measurement set (which is the same thing as the input measurement set).

    """
    with tables.table(ms, readonly=False) as t:
        # Copied from https://github.com/casacore/python-casacore/blob/master/casacore/tables/msutil.py#L48
        column_names = t.colnames()
        if CORRECTED_DATA not in column_names:
            dminfo = t.getdminfo(DATA)
            cdesc = t.getcoldesc(DATA)
            dminfo['NAME'] = 'correcteddata'
            cdesc['comment'] = 'The corrected data column'
            t.addcols(tables.maketabdesc(tables.makecoldesc(CORRECTED_DATA, cdesc)), dminfo)
            t.putcol(CORRECTED_DATA, t.getcol(DATA))
        else:
            log.info(f'{ms} already has {CORRECTED_DATA} column. Not peeling.')
            return ms
    log.info(f'Generating sources.json for {ms}')
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_json = path.join(tmpdir, 'sources.json')
        if _write_peeling_sources_json(utc_time, sources_json, include_rfi_source=include_rfi_source):
            # This reads from the just-created CORRECTED_DATA column and writes to CORRECTED_DATA column.
            ttcal.peel_with_ttcal(ms, sources_json)
    return ms


def _write_peeling_sources_json(utc_timestamp: datetime, out_json: str, include_rfi_source: bool) -> Optional[str]:
    sources = _get_peeling_sources_list(utc_timestamp, include_rfi_source)
    if sources:
        log.info(f'{len(sources)} sources to peel for {utc_timestamp.isoformat()}.')
        with open(out_json, 'w') as out_file:
            json.dump(sources, out_file)
        return out_json
    else:
        log.info(f'No sources to peel for {utc_timestamp.isoformat()}')
        return None


def _get_peeling_sources_list(utc_timestamp: datetime, include_rfi_source: bool) -> List[dict]:
    sources = []

    if coordutils.is_visible(coordutils.CYG_A, utc_timestamp):
        sources.append(
            CYG_A_UNPOLARIZED_RESOLVED)
    if coordutils.is_visible(coordutils.CAS_A, utc_timestamp):
        sources.append(
            CAS_A_UNPOLARIZED_RESOLVED
        )

    if include_rfi_source:
        sources.append(
            RFI_B
        )
    return sources
