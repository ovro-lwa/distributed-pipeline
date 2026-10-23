"""Simple all-sky dirty images (uniform weighting, 30-lambda inner Tukey taper) from extracted I visibilities."""
import os
import numpy as np

C = 299792458.0
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {"Cas A": (350.866, 58.811), "Cyg A": (299.868, 40.734), "Vir A": (187.706, 12.391)}
RFI_AZEL = None


def load(tag, snap):
    p = f"{HERE}/npz/{tag}_{snap}.npz"
    if os.path.exists(p):
        d = np.load(p)
        return d["I"], d["flag"]
    p = f"{HERE}/mem/{tag}_{snap}.bin"
    with open(p, "rb") as f:
        nb, nf = np.frombuffer(f.read(16), dtype=np.int64)
        I = np.frombuffer(f.read(nb * nf * 8), dtype=np.complex64).reshape(nf, nb).T
        fl = np.frombuffer(f.read(nb * nf), dtype=np.uint8).reshape(nf, nb).T.astype(bool)
    return I, fl


def geometry(snap):
    d = np.load(f"{HERE}/npz/prep_{snap}.npz")
    return d["uvw"], d["freq"], float(d["time"]), d["phase_dir"]


def dirty(I, flag, uvw, freq, N=2048, sign=1):
    lam = C / freq
    u = uvw[:, 0:1] / lam[None, :]
    v = uvw[:, 1:2] / lam[None, :]
    ok = ~flag
    du = 0.5                         # image spans l, m in [-1, 1]
    uu = np.concatenate([u[ok], -u[ok]]); vv = np.concatenate([v[ok], -v[ok]])
    vis = np.concatenate([I[ok], np.conj(I[ok])])
    r = np.hypot(uu, vv)
    taper = np.where(r < 30, 0.5 * (1 - np.cos(np.pi * r / 30)), 1.0)
    iu = np.round(uu / du).astype(int) % N; iv = np.round(vv / du).astype(int) % N
    cnt = np.zeros((N, N)); np.add.at(cnt, (iv, iu), 1)
    w = taper / cnt[iv, iu]
    G = np.zeros((N, N), complex); np.add.at(G, (iv, iu), w * vis)
    img = (np.fft.fft2(G) if sign > 0 else np.fft.ifft2(G) * N * N).real / w.sum()
    img = np.fft.fftshift(img)
    return img


def lm_of(ra_deg, dec_deg, pd):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    a0, d0 = pd
    l = np.cos(d) * np.sin(a - a0)
    m = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)
    n = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(a - a0)
    return l, m, n


def pix(l, m, N=2048):
    # fft2 of grid indexed [v, u]: pixel (row, col) = (m, l) * N/2 + N/2 (sign checked on Cas A)
    return N / 2 + l * N / 2, N / 2 + m * N / 2
