import matplotlib.pyplot as plt
import numpy as np


def complex_qn_normalize(qn: list | np.ndarray):
    """Normalize the complex grating profile by its peak coupling strength.

    The input profile encodes both the coupling coefficient kappa(z) in its
    amplitude and the grating phase phi_G(z) in its argument:
        q(z) = kappa(z) * exp(j * phi_G(z))

    Args:
        qn (list | np.ndarray): Complex grating profile values along the structure.

    Returns:
        tuple[np.ndarray, float]:
            - qn_norm: Normalized complex grating profile (peak amplitude = 1.0).
            - max_qn_amp: Peak amplitude of the original profile (max |qn|).
    """
    qn = np.asarray(qn)
    max_qn_amp = np.max(np.abs(qn))
    qn_norm = qn / max_qn_amp
    return qn_norm, max_qn_amp


def plot_qn(qn_norm: list | np.ndarray, period_um: float):
    """Plot the amplitude and unwrapped phase of the normalized complex grating profile.

    Generates a two-panel diagnostic plot displaying:
      - Left panel: Normalized coupling coefficient (|q_norm(z)|) along z.
      - Right panel: Unwrapped grating phase (phi_G(z) / pi) along z.

    Args:
        qn_norm (list | np.ndarray): Normalized complex grating profile.
        period_um (float): Grating period (spatial sampling pitch) in micrometers.

    Returns:
        tuple[matplotlib.figure.Figure, np.ndarray]: Matplotlib Figure and Axes objects.
    """
    qn_norm = np.asarray(qn_norm)
    z_um = np.arange(len(qn_norm)) * period_um

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout='constrained')
    ax_left, ax_right = axes[0], axes[1]

    color_amp = 'tab:blue'
    qn_norm_amp = np.abs(qn_norm)
    ax_left.plot(z_um, qn_norm_amp, color=color_amp)
    ax_left.set_xlabel(r'z ($\mu$m)')
    ax_left.set_ylabel(r'Norm. $\kappa$', color=color_amp)
    ax_left.tick_params(axis='y', labelcolor=color_amp)
    ax_left.grid(True)

    color_pha = 'tab:red'
    qn_pha = np.unwrap(np.angle(qn_norm))
    ax_right.plot(z_um, qn_pha / np.pi, color=color_pha)
    ax_right.set_xlabel(r'z ($\mu$m)')
    ax_right.set_ylabel(r'Grating Phase ($\pi$)', color=color_pha)
    ax_right.tick_params(axis='y', labelcolor=color_pha)
    ax_right.grid(True)

    return fig, axes