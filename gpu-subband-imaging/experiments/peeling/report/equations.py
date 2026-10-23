"""Render the report's equations (matplotlib mathtext) to PNG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eq"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})

EQ = {
 "measurement": r"$V_{pq}(\nu)=\sum_{d}\,J_{p,d}(\nu)\,M_{pq,d}(\nu)\,J_{q,d}^{H}(\nu)\;+\;N_{pq}$",
 "zest": r"$\mathrm{for\ } d:\quad V \leftarrow V + J_dM_dJ_d^H,\quad J_d \leftarrow \mathrm{solve}(V, M_d),\quad V \leftarrow V - J_dM_dJ_d^H\qquad(\times\ \mathrm{peeliter})$",
 "farfield": r"$M_{pq}(\nu)=\sum_{c}\,\mathbf{S}_c(\nu)\;e^{-2\pi i\,\nu\,(\tau_q-\tau_p)/c}\;\,G_c(\mathbf{b}_{pq},\nu),\qquad \tau_p=\mathbf{r}_p\cdot(\hat{\mathbf{s}}_c-\hat{\mathbf{s}}_0),\quad \hat{\mathbf{s}}=R(t)\,\hat{\mathbf{s}}_{\mathrm{J2000}}$",
 "stokes": r"$\mathbf{S}=\begin{pmatrix} I+Q & U-iV\\ U+iV & I-Q\end{pmatrix},\qquad \log_{10}I(\nu)=\log_{10}I_0+\sum_k\alpha_k\,[\log_{10}(\nu/\nu_0)]^k$",
 "gauss": r"$G_c=\exp\!\left[-w_{\mathrm{maj}}\left(\frac{\mathbf{b}_{pq}\cdot\hat{\mathbf{e}}_{\mathrm{maj}}}{\lambda}\right)^{2}-w_{\mathrm{min}}\left(\frac{\mathbf{b}_{pq}\cdot\hat{\mathbf{e}}_{\mathrm{min}}}{\lambda}\right)^{2}\right],\quad w=\frac{\pi^{2}\sin^{2}\theta_{\mathrm{FWHM}}}{4\ln 2},\quad \mathbf{b}_{pq}=\mathbf{r}_q-\mathbf{r}_p$",
 "rfi": r"$M_{pq}^{\mathrm{RFI}}(\nu)=\mathbf{S}(\nu)\,e^{\,2\pi i\,\nu\,(\rho_q-\rho_p)/c},\qquad \rho_p=|\mathbf{P}-\mathbf{r}_p|+\hat{\mathbf{s}}_0\cdot\mathbf{r}_p$",
 "stefcal": r"$J_j^{\mathrm{new}}=\left[\Big(\sum_{i\neq j}(J_iM_{ij})^{H}(J_iM_{ij})\Big)^{-1}\sum_{i\neq j}(J_iM_{ij})^{H}V_{ij}\right]^{H},\qquad f(J)=J^{\mathrm{new}}-J$",
 "rk4": r"$k_1=f(J),\;k_2=f(J+\tfrac{1}{2}k_1),\;k_3=f(J+\tfrac{1}{2}k_2),\;k_4=f(J+k_3),\quad J\leftarrow J+\tfrac{1}{6}(k_1+2k_2+2k_3+k_4)\ \ \mathrm{until}\ \frac{\|\Delta J\|}{\|J\|}<\mathrm{tol\ (per\ channel)}$",
 "damped": r"$\mathrm{production:}\quad J\leftarrow J+\tfrac{1}{2}f(J)\quad \mathrm{until}\ \|f(J)\|<\mathrm{tol}\,\|J\|\ \ (\mathrm{all\ channels\ together})$",
 "minuvw": r"$|\mathbf{r}_p-\mathbf{r}_q|<\mathrm{minuvw}\cdot c/\nu\ \Rightarrow\ V_{pq},M_{pq}\ \mathrm{excluded\ from\ the\ solve};\qquad V_{pq}\ \mathrm{flagged}\ \Rightarrow\ M_{pq}=0\ \ (\mathrm{model\ flags})$",
 "uvw": r"$(u,v,w)^{T}=E\,R(t)^{T}\,\mathbf{b}_{pq},\qquad E=\big(\hat{\mathbf{e}}_u,\hat{\mathbf{e}}_v,\hat{\mathbf{e}}_w\big)^{T}\ \mathrm{at\ PHASE\_DIR};\quad R=\mathrm{(GAST)\cdot(nutation)\cdot(IAU76\ precession)}$",
 "tukey": r"$P(\phi)=\begin{cases}1 & |\phi|\leq\phi_s\\ \frac{1}{2}+\frac{1}{2}\cos\frac{\pi(|\phi|-\phi_s)}{\phi_w} & \phi_s<|\phi|<\phi_s+\phi_w\\ 0 & \mathrm{otherwise}\end{cases},\qquad \phi_s=\frac{2\pi\nu\rho_j\sin\Delta\theta}{c}$",
 "regcost": r"$\min_J\ \sum_{pq}\|V_{pq}-J_pM_{pq}J_q^{H}\|^{2}\;-\;\lambda\sum_j\log P(\arg J_j)$",
 "guard": r"$\mathrm{guard:\ keep\ the\ new\ } J_d\ \mathrm{only\ if}\ \sum_{\mathrm{unflagged}}\left(|V_{xx}|^2+|V_{yy}|^2\right)\ \mathrm{decreased}$",
 "herm": r"$M_{ij}=M_{ji}^{H}\;\Rightarrow\;\mathrm{thread}\ j\ \mathrm{reads\ column}\ M_{\cdot j}\ \mathrm{as\ row}\ M_{j\cdot}:\ \mathrm{coalesced\ loads,\ same\ arithmetic}$",
 "cost": r"$\mathrm{model\ cost}\ \propto N_{\mathrm{base}}N_{\nu}N_{\mathrm{comp}}\approx 61776\times48\times3\approx9\times10^{6};\qquad \mathrm{StefCal\ step}\ \propto N_{\mathrm{ant}}^{2}N_{\nu}\approx6\times10^{6}\ \mathrm{Jones\ products}$",
}

for k, tex in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=15)
    fig.savefig(f"{OUT}/{k}.png", dpi=200, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)
print(sorted(os.listdir(OUT)))
