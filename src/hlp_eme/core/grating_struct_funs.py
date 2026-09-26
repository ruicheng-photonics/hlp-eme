"""Complex-Modulated Waveguide Grating Module for HLP-EME Modeling.

This module provides tools to synthesize, slice, and verify complex-modulated
Bragg grating geometries for HLP-EME simulations.

Key Features:
    * Grating Architecture:
        - Symmetric single-mode Bragg gratings (`mc=False`).
        - Asymmetric multimode mode-conversion Bragg gratings (`mc=True`).
    * Apodization Scheme:
        - Lateral Phase-Delay Modulation (LPDM), which decouples the coupling
          magnitude |kappa(z)| and grating phase phi_G(z) into asymmetric upper
          and lower sidewall corrugations.
    * Operation:
        - Discretizes continuous sinusoidal profiles into piecewise-stepped
          rectangular slices suitable for Lumerical EME solvers.
        - Packs repeating periods into periodic block groups (`group_params`)
          for analytical cascading in HLP-EME.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def generate_grating_sinusoidal(
    qn_norm_input: Sequence[complex] | np.ndarray,
    n_periods: int,
    period_um: float,
    w_wg_um: float,
    delta_w_um: float,
    mc: bool | int = 1,
    n_slices_per_period: int = 6,
) -> Tuple[
    List[Dict[str, float]],
    List[List[Dict[str, float]]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Synthesizes a stepped-slice sidewall grating modulated by LPDM.

    Encodes the complex coupling profile $q_{\\mathrm{norm}}(z) = \vert{}q(z)\vert{} e^{j \\phi_G(z)}$
    into physical upper and lower sidewall boundaries.

    Args:
        qn_norm_input: 1D array of normalized complex coupling coefficients sampled per period
        n_periods: Total number of grating periods along the propagation axis.
        period_um: Grating period  in micrometers.
        w_wg_um: Unperturbed base waveguide core width in micrometers.
        delta_w_um: peak-to-peak sidewall corrugation width.
        mc: Mode-conversion flag. If `True` (or `1`), applies asymmetric mode-conversion LPDM
            formulation ($2\\arcsin$); if `False` (or `0`), applies symmetric LPDM ($2\\arccos$).
        n_slices_per_period: Number of uniform rectangular staircased slices per period. Default is 6.

    Returns:
        A 5-element tuple containing:
            - rectangles_all (List[Dict[str, float]]): Flat list of all discretized rectangular slice dictionaries,
              each containing:
                * 'period_index' (int): 0-based period index.
                * 'z_start' (float): Slice upstream coordinate (um).
                * 'z_end' (float): Slice downstream coordinate (um).
                * 'z_length' (float): Longitudinal slice thickness (um).
                * 'x_lower' (float): Lower sidewall transverse boundary coordinate (um).
                * 'x_upper' (float): Upper sidewall transverse boundary coordinate (um).
                * 'width' (float): slide width (um).
            - rectangles_by_period (List[List[Dict[str, float]]]): Slices grouped by period index
              with shape `[n_periods][n_slices_per_period]`.
            - z_sampling (np.ndarray): High-density longitudinal coordinate array (um)
              for continuous profile plotting.
            - phase_plot (np.ndarray): Sampled longitudinal phase values $\\phi_G(z)$ in radians.
            - delay_plot (np.ndarray): Sampled lateral phase delay values $\\Delta\\phi_{\\mathrm{lat}}(z)$ in radians.
    """
    z_total_um = n_periods * period_um

    # 1. Map normalized coupling amplitude to lateral phase delay
    qn_amp_nor = np.abs(qn_norm_input)
    if mc:
        phase_delays = np.arcsin(np.clip(qn_amp_nor, 0.0, 1.0)) * 2.0
    else:
        phase_delays = np.arccos(np.clip(qn_amp_nor, 0.0, 1.0)) * 2.0

    qn_phase_vals = np.angle(qn_norm_input)

    # 2. Define phase mapping closures indexed by local period index
    def longitudinal_phase(z: Union[float, np.ndarray]) -> np.ndarray:
        idx = np.clip(np.floor(z / period_um).astype(int), 0, n_periods - 1)
        return qn_phase_vals[idx]

    def lateral_phase_delay(z: Union[float, np.ndarray]) -> np.ndarray:
        idx = np.clip(np.floor(z / period_um).astype(int), 0, n_periods - 1)
        return phase_delays[idx]

    # 3. Sidewall boundary phase functions (upper and lower offsets)
    def phase_upper(z: Union[float, np.ndarray]) -> np.ndarray:
        return 2.0 * np.pi * z / period_um + longitudinal_phase(z) + lateral_phase_delay(z) / 2.0

    def phase_lower(z: Union[float, np.ndarray]) -> np.ndarray:
        return 2.0 * np.pi * z / period_um + longitudinal_phase(z) - lateral_phase_delay(z) / 2.0

    # Continuous boundary functions evaluated at local coordinate z
    def upper_boundary(z: Union[float, np.ndarray]) -> np.ndarray:
        return w_wg_um / 2.0 + (delta_w_um / 2.0) * np.sin(phase_upper(z))

    def lower_boundary(z: Union[float, np.ndarray]) -> np.ndarray:
        return -w_wg_um / 2.0 - (delta_w_um / 2.0) * np.sin(phase_lower(z))

    # 4. Discretize continuous geometry into stepped rectangular slices
    rectangles_all: List[Dict[str, float]] = []
    rectangles_by_period: List[List[Dict[str, float]]] = []
    slice_length = period_um / n_slices_per_period

    for period_index in range(n_periods):
        z0 = period_index * period_um
        rects_this_period: List[Dict[str, float]] = []

        for slice_idx in range(n_slices_per_period):
            zs = z0 + slice_idx * slice_length
            ze = z0 + (slice_idx + 1) * slice_length

            # Midpoint sampling rule for slice discretization
            zm = 0.5 * (zs + ze)
            xu = float(upper_boundary(zm))
            xl = float(lower_boundary(zm))

            rect = {
                "period_index": period_index,
                "z_start": zs,
                "z_end": ze,
                "z_length": ze - zs,
                "x_lower": xl,
                "x_upper": xu,
                "width": xu - xl,
            }
            rects_this_period.append(rect)
            rectangles_all.append(rect)

        rectangles_by_period.append(rects_this_period)

    # 5. Continuous coordinate grid for diagnostic inspection
    z_sampling = np.linspace(0, z_total_um, n_periods * 50 + 1)
    phase_plot = np.array([longitudinal_phase(z) for z in z_sampling])
    delay_plot = np.array([lateral_phase_delay(z) for z in z_sampling])

    return rectangles_all, rectangles_by_period, z_sampling, phase_plot, delay_plot


def extract_group_params_for_eme(
    rects_by_period: Sequence[Sequence[Dict[str, float]]],
    n_periods: int,
    n_periods_per_block: int,
    period_um: float,
) -> List[Dict[str, Any]]:
    """Generatew block configuration list for `verify_eme_grating()`.

    Args:
        rects_by_period: Nested list of slice geometries indexed by `[period_idx][slice_idx]`.
        n_periods: Total count of physical grating periods.
        n_periods_per_block: number of periods per block.
        period_um: Physical grating pitch (um).

    Returns:
        A list of block parameter dictionaries. Each element contains:
            - 'group_index' (int): 0-based block index.
            - 'n_repeats' (int): Repetition multiplier count assigned to this block.
            - 'period_um' (float): Nominal period length (um).
            - 'base_period_index' (int): Physical period index used as the representative template.
            - 'slices' (List[Dict[str, float]]): List of slice geometries inside the template,
              with coordinates relative to the template origin:
                * 'z_start_rel' (float): Local slice start offset (um).
                * 'z_end_rel' (float): Local slice end offset (um).
                * 'z_length' (float): Longitudinal slice thickness (um).
                * 'x_lower' (float): Lower sidewall boundary position (um).
                * 'x_upper' (float): Upper sidewall boundary position (um).
                * 'width' (float): Slice width along the transverse axis (um).
    """
    group_params: List[Dict[str, Any]] = []
    num_groups = int(np.ceil(n_periods / n_periods_per_block))

    for g_idx in range(num_groups):
        base_period_idx = g_idx * n_periods_per_block
        actual_repeats = min(n_periods_per_block, n_periods - base_period_idx)

        template_rects = rects_by_period[base_period_idx]
        slices_info: List[Dict[str, float]] = []
        z_base_start = base_period_idx * period_um

        for r in template_rects:
            slice_data = {
                "z_start_rel": r["z_start"] - z_base_start,
                "z_end_rel": r["z_end"] - z_base_start,
                "z_length": r["z_length"],
                "x_lower": r["x_lower"],
                "x_upper": r["x_upper"],
                "width": r["width"],
            }
            slices_info.append(slice_data)

        group_data = {
            "group_index": g_idx,
            "n_repeats": actual_repeats,
            "period_um": period_um,
            "base_period_index": base_period_idx,
            "slices": slices_info,
        }
        group_params.append(group_data)

    return group_params


def draw_rects_with_lines(
    ax: plt.Axes,
    rectangles: Sequence[Dict[str, float]],
    color: str,
    period_um: float,
    alpha: float = 0.8,
    label: Optional[str] = None,
) -> None:
    """Renders staircased rectangular grating slices with intra-period interface markers.

    Draws each slice as a filled `matplotlib.patches.Rectangle` and overlays vertical
    dotted lines at sub-period slice transitions to visualize mesh discretization.

    Args:
        ax: Target Matplotlib axes object.
        rectangles: Sequence of slice dictionaries containing keys `'z_start'`,
            `'x_lower'`, `'z_length'`, and `'width'`.
        color: Patch fill color (e.g., `'tab:blue'`, `'#1f77b4'`).
        period_um: Period pitch in micrometers ($\\mu\\text{m}$), used to distinguish intra-period
            slice cuts from outer period boundaries.
        alpha: Transparency factor of the filled rectangle patches ($0.0 \\le \\alpha \\le 1.0$).
        label: Optional legend label assigned exclusively to the first rendered patch.
    """
    for i, rect in enumerate(rectangles):
        patch = Rectangle(
            xy=(rect["z_start"], rect["x_lower"]),
            width=rect["z_length"],
            height=rect["width"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.3,
            alpha=alpha,
            label=label if i == 0 else "",
        )
        ax.add_patch(patch)

    # Identify internal slice interfaces (excluding outer period boundaries)
    internal_cuts = set()
    for rect in rectangles:
        internal_cuts.add(rect["z_start"])
        internal_cuts.add(rect["z_end"])

    for z_cut in internal_cuts:
        if not np.isclose(z_cut % period_um, 0.0):
            ax.axvline(z_cut, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)


def generate_eme_grating_data(
    qn_norm_piecewise: Sequence[complex] | np.ndarray,
    n_periods: int,
    period_um: float,
    w_wg_um: float,
    delta_w_um: float,
    n_periods_per_block: int = 20,
    mc: bool = False,
    qn_norm_org: Optional[Sequence[complex] | np.ndarray] = None,
    n_slices_per_period: int = 6,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compiles grating geometry, performs periodic grouping, and aggregates plotting datasets.

    Generates the stepped-slice coordinates for both piecewise-constant and continuous
    profiles, constructs the EME block definitions, and organizes plotting bundles.

    Args:
        qn_norm_piecewise: Piecewise-constant, complex normalized grating profile.
        n_periods: Total physical grating period count.
        period_um: Grating pitch in micrometers ($\\mu\\text{m}$).
        w_wg_um: Nominal unperturbed core width in micrometers ($\\mu\\text{m}$).
        delta_w_um: Peak-to-peak corrugation width ($2 \\times \\Delta w$) in micrometers ($\\mu\\text{m}$).
        n_periods_per_block: Number of uniform periods clustered into an EME periodic group. Default is 20.
        mc: Mode-conversion flag. If `True`, enables asymmetric mode-conversion LPDM;
            otherwise defaults to symmetric reflection LPDM. Default is `False`.
        qn_norm_org: Original, continuous grating profiles.
            If provided, used for diagnostic baseline comparison (Structure 1).
        n_slices_per_period: Number of uniform staircase steps evaluated inside each period. 

    Returns:
        A 2-element tuple containing:
            - group_params (List[Dict[str, Any]]): Formatted block parameters ready for EME solvers.
            - plot_data (Dict[str, Any]): Cached dictionary storing coordinate arrays and geometries:
                * 'rects_s1': Baseline continuous geometry slices.
                * 'rects_s2': Piecewise discretized geometry slices.
                * 'rects_by_period_s2': Slices partitioned by period index.
                * 'z_samp': Continuous longitudinal sampling coordinates.
                * 'phi_p': Longitudinal phase values along `z_samp`.
                * 'delay_p': Lateral phase delay values along `z_samp`.
                * 'period_um': Period pitch.
                * 'w_wg_um': Unperturbed waveguide width.
                * 'n_periods_per_block': Periodic block size.
                * 'n_slices_per_period': Slices per period.
    """
    # 1. Generate baseline geometry for the piecewise discretized grating
    rects_s2, rects_by_period_s2, z_samp, phi_p, delay_p = generate_grating_sinusoidal(
        qn_norm_piecewise,
        n_periods,
        period_um,
        w_wg_um,
        delta_w_um,
        n_slices_per_period=n_slices_per_period,
        mc=mc,
    )

    # Generate reference continuous grating geometry if provided
    if qn_norm_org is not None:
        rects_s1, _, _, _, _ = generate_grating_sinusoidal(
            qn_norm_org,
            n_periods,
            period_um,
            w_wg_um,
            delta_w_um,
            n_slices_per_period=n_slices_per_period,
            mc=mc,
        )
    else:
        rects_s1 = rects_s2

    # 2. Extract block parameters for EME solver
    group_params = extract_group_params_for_eme(
        rects_by_period_s2, n_periods, n_periods_per_block, period_um
    )

    # 3. Package auxiliary arrays and dictionaries for verification rendering
    plot_data = {
        "rects_s1": rects_s1,
        "rects_s2": rects_s2,
        "rects_by_period_s2": rects_by_period_s2,
        "z_samp": z_samp,
        "phi_p": phi_p,
        "delay_p": delay_p,
        "period_um": period_um,
        "w_wg_um": w_wg_um,
        "n_periods_per_block": n_periods_per_block,
        "n_slices_per_period": n_slices_per_period,
    }

    return group_params, plot_data


def verify_eme_grating(
    group_params: Sequence[Dict[str, Any]],
    plot_data: Dict[str, Any],
    draw_block_range: Optional[Tuple[int, int]] = None,
    plot_only_struct4: bool = False,
) -> None:
    """Renders multi-tier geometric verification and phase diagnostics for HLP-EME gratings.

    Visualizes and compares up to 5 tiers of profiles to validate discretization accuracy:
        1. Phase Profiles: Longitudinal phase $\\phi(z)/\\pi$ and lateral delay $\\Delta\\phi_{\\mathrm{lat}}(z)/\\pi$.
        2. Structure 1: Target continuous grating geometry from $q_{\\mathrm{norm, org}}$.
        3. Structure 2: Piecewise-constant staircased geometry from $q_{\\mathrm{norm, piecewise}}$.
        4. Structure 3: Artificially replicated block geometry (physical representation of the EME model).
        5. Structure 4: EME solver native geometry (cascaded single-period unit templates).

    Args:
        group_params: Block configuration list produced by `extract_group_params_for_eme`.
        plot_data: Auxiliary visualization dictionary generated by `generate_eme_grating_data`.
        draw_block_range: Optional tuple `(start_block_idx, end_block_idx)` defining the block
            index window to visualize. If `None`, renders all blocks.
        plot_only_struct4: If `True`, renders only the compressed single-period template geometry
            (Structure 4) in a compact single-row canvas. If `False`, displays the full 5-row
            comparison diagnostic plot.
    """
    # 0. Unpack rendering dependencies
    rects_s1 = plot_data["rects_s1"]
    rects_s2 = plot_data["rects_s2"]
    rects_by_period_s2 = plot_data["rects_by_period_s2"]
    z_samp = plot_data["z_samp"]
    phi_p = plot_data["phi_p"]
    delay_p = plot_data["delay_p"]
    period_um = plot_data["period_um"]
    w_wg_um = plot_data["w_wg_um"]
    n_periods_per_block = plot_data["n_periods_per_block"]

    # 1. Determine block and coordinate range for visualization
    total_blocks = len(group_params)
    if draw_block_range is None:
        start_block_idx = 0
        end_block_idx = total_blocks
    else:
        start_block_idx = max(0, draw_block_range[0])
        end_block_idx = min(total_blocks, draw_block_range[1])

    n_blocks_drawn = end_block_idx - start_block_idx
    start_period_idx = start_block_idx * n_periods_per_block
    end_period_idx = min(
        end_block_idx * n_periods_per_block,
        group_params[-1]["base_period_index"] + group_params[-1]["n_repeats"],
    )

    z_start_crop = start_period_idx * period_um
    z_end_crop = end_period_idx * period_um

    # 2. Reconstruct Structure 3 (period-replicated geometry) if full comparison is requested
    if not plot_only_struct4:
        rects_s3: List[Dict[str, float]] = []
        for g_data in group_params:
            base_p_idx = g_data["base_period_index"]
            template_rects = rects_by_period_s2[base_p_idx]
            for copy_idx in range(g_data["n_repeats"]):
                shift_z = copy_idx * period_um
                for r in template_rects:
                    new_r = r.copy()
                    new_r["z_start"] = r["z_start"] + shift_z
                    new_r["z_end"] = r["z_end"] + shift_z
                    new_r["period_index"] = base_p_idx + copy_idx
                    rects_s3.append(new_r)

        crop_s1 = [
            r for r in rects_s1
            if (r["z_start"] >= z_start_crop - 1e-9) and (r["z_end"] <= z_end_crop + 1e-9)
        ]
        crop_s2 = [
            r for r in rects_s2
            if (r["z_start"] >= z_start_crop - 1e-9) and (r["z_end"] <= z_end_crop + 1e-9)
        ]
        crop_s3 = [
            r for r in rects_s3
            if (r["z_start"] >= z_start_crop - 1e-9) and (r["z_end"] <= z_end_crop + 1e-9)
        ]

    # Generate Structure 4 (concatenated single-period unit templates in EME space)
    rects_s4_crop: List[Dict[str, float]] = []
    eme_start_z = start_block_idx * period_um
    curr_z_offset = eme_start_z

    for g_data in group_params[start_block_idx:end_block_idx]:
        b_idx = g_data["base_period_index"]
        template_rects = rects_by_period_s2[b_idx]
        z_base_start = b_idx * period_um

        for r in template_rects:
            new_r = r.copy()
            new_r["z_start"] = (r["z_start"] - z_base_start) + curr_z_offset
            new_r["z_end"] = (r["z_end"] - z_base_start) + curr_z_offset
            rects_s4_crop.append(new_r)

        curr_z_offset += period_um

    eme_end_z = curr_z_offset

    # 3. Canvas rendering and subplot styling
    info_str = (
        f"Blocks [{start_block_idx} ~ {end_block_idx - 1}] ({n_blocks_drawn} Blocks) | "
        f"Periods [{start_period_idx} ~ {end_period_idx - 1}]"
    )

    def style_struct4_axis(ax: plt.Axes) -> None:
        """Helper to style boundaries and block label annotations on Structure 4."""
        ax.set_xlim(eme_start_z, eme_end_z)
        ax.set_ylim(-w_wg_um * 1.15, w_wg_um * 1.15)

        for k in range(n_blocks_drawn + 1):
            line_z = eme_start_z + k * period_um
            ax.axvline(line_z, color="red", linestyle="--", linewidth=1.2, alpha=0.85)

            if k < n_blocks_drawn:
                block_idx = start_block_idx + k
                center_z = line_z + period_um / 2.0
                ymin, ymax = ax.get_ylim()
                text_y = ymax - (ymax - ymin) * 0.12
                ax.text(
                    center_z,
                    text_y,
                    f"B {block_idx}",
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                )

    if plot_only_struct4:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        draw_rects_with_lines(
            ax,
            rects_s4_crop,
            "tab:red",
            period_um=period_um,
            label="Struct 4 (EME Native Rendered Geometry)",
        )
        ax.set_title(f"Structure 4: EME Native Geometry | {info_str}")
        ax.set_xlabel(r"$z$ ($\mu$m) [EME Native Compressed Coordinates]")
        ax.set_ylabel(r"$x$ ($\mu$m)")
        ax.grid(True, alpha=0.3)
        style_struct4_axis(ax)
    else:
        fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=False)

        # Row 0: Phase parameters
        mask = (z_samp >= z_start_crop) & (z_samp <= z_end_crop)
        axes[0].plot(
            z_samp[mask],
            phi_p[mask] / np.pi,
            color="tab:purple",
            linewidth=2,
            label=r"Longitudinal phase $\phi(z)/\pi$",
        )
        axes[0].plot(
            z_samp[mask],
            delay_p[mask] / np.pi,
            color="tab:green",
            linewidth=2,
            label=r"Lateral phase delay $\Delta\phi_{\rm lat}(z)/\pi$",
        )
        axes[0].set_title(f"1. Longitudinal Phase & Lateral Delay | {info_str}", fontsize=11)
        axes[0].set_ylabel(r"Phase / $\pi$")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        # Row 1: Target continuous structure
        draw_rects_with_lines(axes[1], crop_s1, "tab:blue", period_um=period_um, label="Struct 1 (qn0 Original)")
        axes[1].set_title("2. Structure 1: Original Continuous Signal qn0")
        axes[1].set_ylabel(r"$x$ ($\mu$m)")
        axes[1].grid(True, alpha=0.3)

        # Row 2: Piecewise-constant discretized structure
        draw_rects_with_lines(axes[2], crop_s2, "tab:orange", period_um=period_um, label="Struct 2 (qn1 Piecewise)")
        axes[2].set_title("3. Structure 2: Piecewise-Constant qn1 (Step Phase)")
        axes[2].set_ylabel(r"$x$ ($\mu$m)")
        axes[2].grid(True, alpha=0.3)

        # Row 3: Artificially replicated block structure
        draw_rects_with_lines(axes[3], crop_s3, "tab:green", period_um=period_um, label="Struct 3 (Artificial EME Replica)")
        axes[3].set_title("4. Structure 3: Block-Replicated Target Structure (Physical View)")
        axes[3].set_ylabel(r"$x$ ($\mu$m)")
        axes[3].grid(True, alpha=0.3)

        # Row 4: Compressed EME native structure
        draw_rects_with_lines(axes[4], rects_s4_crop, "tab:red", period_um=period_um, label="Struct 4 (EME Native Rendered Geometry)")
        axes[4].set_title(f"5. Structure 4: EME Native Geometry ({n_blocks_drawn} Single-Period Templates Rendered)")
        axes[4].set_xlabel(r"$z$ ($\mu$m) [EME Native Compressed Coordinates]")
        axes[4].set_ylabel(r"$x$ ($\mu$m)")
        axes[4].grid(True, alpha=0.3)

        # Configure axis boundaries and grid guides for Rows 0–3 (Physical Space)
        for i in range(4):
            ax = axes[i]
            is_geom_plot = i > 0
            ax.set_xlim(z_start_crop, z_end_crop)
            if is_geom_plot:
                ax.set_ylim(-w_wg_um * 1.15, w_wg_um * 1.15)

            for p_idx in range(start_period_idx, end_period_idx + 1):
                is_block_boundary = p_idx % n_periods_per_block == 0
                if not is_geom_plot:
                    line_color, line_width, line_alpha, line_style = "gray", 0.8, 0.3, "--"
                    draw_label = False
                else:
                    if is_block_boundary:
                        line_color, line_width, line_alpha, line_style = "red", 1.2, 0.85, "--"
                        draw_label = True
                    else:
                        line_color, line_width, line_alpha, line_style = "black", 1.0, 0.4, "--"
                        draw_label = False

                ax.axvline(
                    p_idx * period_um,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=line_alpha,
                )

                if draw_label and p_idx < end_period_idx:
                    block_idx = p_idx // n_periods_per_block
                    center_z = (p_idx + min(n_periods_per_block, end_period_idx - p_idx) / 2.0) * period_um
                    ymin, ymax = ax.get_ylim()
                    text_y = ymax - (ymax - ymin) * 0.12
                    ax.text(
                        center_z,
                        text_y,
                        f"B {block_idx}",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                    )

        # Style template view on Row 4
        style_struct4_axis(axes[4])

    plt.tight_layout()
    plt.show()

    print("=" * 65)
    print("[ EME Sampling Verification Report ]")
    print(f"* Plotting Mode           : {'Single Template Mode (Struct 4)' if plot_only_struct4 else 'Full Comparison Mode'}")
    print(f"* Total EME Groups Defined: {total_blocks} Block(s)")
    print(f"* Rendered Group Range    : Block [{start_block_idx}] ~ [{end_block_idx - 1}]")
    print(f"* Corresponding Periods   : Period [{start_period_idx}] ~ [{end_period_idx - 1}]")
    print("=" * 65)