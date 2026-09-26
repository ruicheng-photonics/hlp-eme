import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt


def lin2db(t: np.ndarray | list, is_power: bool = False) -> np.ndarray:
    """
    Convert linear field amplitude or optical power to decibels (dB).
    Uses 20 * log10(|t|) for field amplitude and 10 * log10(|t|^2) for power.
    """
    t = np.asarray(t)
    factor = 10 if is_power else 20
    return factor * np.log10(np.abs(t) + 1e-15)


def linearphase(p, a):
    """
    Subtract a linear phase ramp determined by two reference indices.
    """
    p = np.asarray(p, dtype=float)
    la = a[1] - a[0]
    slope = (p[a[1]] - p[a[0]]) / la
    indices = np.arange(1, len(p) + 1)
    pp = p - slope * indices
    return pp


def db_to_alpha(loss_db_per_cm):
    """
    Convert propagation loss from dB/cm to the field attenuation coefficient alpha (1/m).
    """
    loss_per_m_linear = 10 ** (loss_db_per_cm / 20.0)
    alpha = np.log(loss_per_m_linear) / 0.01
    return alpha


def matlab_phase(x):
    """
    Replicate MATLAB's phase(x) = unwrap(angle(x)).
    Essential for consistent group delay calculation matching Lumerical conventions.
    """
    return np.unwrap(np.angle(x))


def lam2det(ng, lam_center, lams):
    """
    Convert optical wavelengths to wavenumber detuning delta.
    """
    detuning = 2 * np.pi * ng * (1.0 / lams - 1.0 / lam_center)
    return detuning[::-1]


def piecewised(qn, n_periods_per_block: int = 20, plot: bool = False):
    """
    Discretize a complex coupling profile qn into spatially uniform piecewise-constant blocks.

    Averages magnitude and phase separately to prevent artificial amplitude degradation 
    caused by destructive interference when directly summing complex values.
    """
    qn = np.asarray(qn)
    qn_out = np.empty_like(qn, dtype=complex)
    N = len(qn)

    # 1. Extract global amplitude profile and unwrapped phase profile
    qn_amp = np.abs(qn)
    qn_pha = np.unwrap(np.angle(qn))

    for i in range(0, N, n_periods_per_block):
        # 2. Compute the mean amplitude and phase within the current block
        amp_mean = np.mean(qn_amp[i : i + n_periods_per_block])
        pha_mean = np.mean(qn_pha[i : i + n_periods_per_block])

        # 3. Reconstruct complex coupling coefficient and assign to block
        qn_out[i : i + n_periods_per_block] = amp_mean * np.exp(1j * pha_mean)

    # --- Verification Plots ---
    if plot:
        zn = range(N)
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained")
        ax0, ax1 = axes[0], axes[1]

        # --- Left Panel: Coupling Coefficient Amplitude ---
        ax0.plot(zn, qn_amp / 1e3, label="Original")
        ax0.plot(zn, np.abs(qn_out) / 1e3, "--", label="Discretized")
        ax0.set_xlabel("Period Index")
        ax0.set_ylabel(r"$\kappa$ (1/mm)")
        ax0.legend()
        ax0.grid(True)

        # --- Right Panel: Grating Phase Profile ---
        ax1.plot(zn, qn_pha / np.pi, label="Original")
        ax1.plot(zn, np.unwrap(np.angle(qn_out)) / np.pi, "--", label="Discretized")
        ax1.set_xlabel("Period Index")
        ax1.set_ylabel(r"Grating Phase ($\pi$)")
        ax1.grid(True)
        plt.show()

    return qn_out


def kappa_cal_r(R, L):
    """
    Calculate coupling coefficient kappa from peak power reflectivity R and grating length L.

    Parameters:
        R (float or np.ndarray): Peak power reflectivity (range 0 to 1).
        L (float or np.ndarray): Grating length in meters (m).

    Returns:
        kappa (float or np.ndarray): Coupling coefficient in inverse meters (1/m).
    """
    return np.arctanh(np.sqrt(R)) / L


def kappa_cal_bw(delta_lambda, ng, lambda_center, L):
    """
    Calculate coupling coefficient kappa from reflection spectrum bandwidth.

    Parameters:
        delta_lambda (float or np.ndarray): Reflection bandwidth (3 dB or null-to-null), 
                                            same units as lambda_center.
        ng (float or np.ndarray): Group index of the waveguide mode.
        lambda_center (float or np.ndarray): Bragg center wavelength, same units as delta_lambda.
        L (float or np.ndarray): Grating length in meters (m).

    Returns:
        kappa (float or np.ndarray): Coupling coefficient in inverse meters (1/m).
    """
    term1 = (delta_lambda * np.pi * ng) / (lambda_center ** 2)
    term2 = np.pi / L
    return np.sqrt(term1 ** 2 - term2 ** 2)


def grating_response_cal(
    qn: np.ndarray,
    sim_lam_params: dict,
    ng: float = 4.2,
    plot: bool = True,
    grating_period_nm: float = 317,
):
    """
    Simulate grating reflection and transmission spectra using the Transfer Matrix Method (TMM).
    """
    lam_nm_center = sim_lam_params.get("lam_nm_center", 1550)
    lam_nm_span = sim_lam_params["lam_nm_span"]
    n_lams = sim_lam_params["n_lams"]
    lam_nm_center_offset = sim_lam_params.get("lam_nm_center_offet", 0)

    lams_nm = np.linspace(
        lam_nm_center - lam_nm_span / 2, lam_nm_center + lam_nm_span / 2, n_lams
    )
    lams = lams_nm * 1e-9

    # Detuning computation
    detuning = lam2det(ng, (lam_nm_center + lam_nm_center_offset) * 1e-9, lams)
    detuning = detuning[::-1]

    # Transfer Matrix Method (TMM) solver
    period = grating_period_nm * 1e-9
    r, t = tmm_solver(qn, period, detuning)

    r_db = lin2db(r)
    t_db = lin2db(t)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(lams_nm, r_db, label="Reflection", linewidth=2)
        plt.plot(lams_nm, t_db, label="Transmission", linewidth=1.5)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Response (dB)")
        plt.title("TMM Reflection/Transmission Spectra")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.show()

    return lams_nm, t, r


def get_dispersion(wavelengths, lam_center_wg, neff0, ng, alpha):
    """
    Compute wavelength-dependent effective index (neff) and complex propagation constant (beta).
    wavelengths and lam_center_wg must share the same units.
    """
    dw_2_dn = (neff0 - ng) / lam_center_wg
    neffs = neff0 + dw_2_dn * (wavelengths - lam_center_wg)
    betas = 2 * np.pi / (wavelengths / neffs) - 1j * alpha
    return neffs, betas


def tmm_solver(q, dz, detuning):
    """
    Transfer Matrix Method (TMM) solver for coupled-mode equations in Bragg gratings.
    """
    N = len(q)
    Nlambda = len(detuning)

    MT = np.zeros((Nlambda, 2, 2), dtype=np.complex128)
    MT[:, 0, 0] = 1
    MT[:, 1, 1] = 1
    det = detuning.reshape(-1, 1, 1)

    for j in range(N):
        q_val = q[j]
        qabs = np.abs(q_val)
        p = np.sqrt(qabs ** 2 - det ** 2 + 0j)

        sinh_pdz = np.sinh(p * dz)
        cosh_pdz = np.cosh(p * dz)

        with np.errstate(divide="ignore", invalid="ignore"):
            term_sinh_div_p = sinh_pdz / p
            term_sinh_div_p[np.abs(p) < 1e-12] = dz

        T = np.zeros((Nlambda, 2, 2), dtype=np.complex128)
        val_11 = cosh_pdz + 1j * det * term_sinh_div_p
        val_12 = q_val * term_sinh_div_p
        val_21 = np.conj(q_val) * term_sinh_div_p
        val_22 = cosh_pdz - 1j * det * term_sinh_div_p

        T[:, 0, 0] = val_11[:, 0, 0]
        T[:, 0, 1] = val_12[:, 0, 0]
        T[:, 1, 0] = val_21[:, 0, 0]
        T[:, 1, 1] = val_22[:, 0, 0]

        MT = T @ MT

    rtemp = -MT[:, 1, 0] / MT[:, 1, 1]
    ttemp = 1.0 / MT[:, 1, 1]
    return rtemp, ttemp