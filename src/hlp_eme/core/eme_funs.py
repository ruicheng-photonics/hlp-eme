from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from hlp_eme.lum import get_lumapi

if TYPE_CHECKING:
    import lumapi

# Type definition for Lumerical EME boundary conditions
BCType = Literal["PML", "Metal", "Symmetric", "Anti-Symmetric", "Periodic", "PMC"]


def add_wg(
    s: lumapi.MODE,
    name: str = "wg",
    l: float = 20.0,
    w: float = 0.5,
    thick: float = 0.22,
    material: str = "Si (Silicon) - Palik",
) -> None:
    """Adds or updates a rectangular waveguide structure in the simulation environment.

    If an object with the specified name already exists, it will be selected and updated;
    otherwise, a new rectangular geometry is created.

    Args:
        s: Active Lumerical MODE session handle.
        name: Identifier name for the waveguide geometry object.
        l: Waveguide length along the propagation direction (x-axis) in micrometers (μm).
        w: Waveguide width (y-span) in micrometers (μm).
        thick: Waveguide core thickness (z-span) in micrometers (μm).
        material: Optical material model name present in the Lumerical material database.

    Raises:
        lumapi.LumApiError: If an unexpected error occurs querying the object name.
    """
    lumapi = get_lumapi()
    try:
        _ = s.getnamed(name)
        obj_exists = True
    except lumapi.LumApiError as e:
        msg = str(e)
        if f"no items matching the name '{name}'" in msg:
            obj_exists = False
        else:
            raise

    if obj_exists:
        s.select(name)
    else:
        s.addrect()

    s.set("name", name)
    s.set("x min", -5e-6)
    s.set("material", material)
    s.set("x max", l * 1e-6)
    s.set("y", 0.0)
    s.set("y span", w * 1e-6)
    s.set("z", 0.0)
    s.set("z span", thick * 1e-6)


def add_eme(
    s: lumapi.MODE,
    y_span: float = 2.0,
    z_span: float = 1.22,
    z_min_bc: Optional[BCType] = None,
    z_max_bc: Optional[BCType] = None,
    y_min_bc: Optional[BCType] = None,
    y_max_bc: Optional[BCType] = None,
    mesh_cells_y: Optional[int] = None,
    mesh_cells_z: Optional[int] = None,
    background_material: str = "SiO2 (Glass) - Palik",
    n_modes_all_cell_groups: Optional[int] = None,
    center_wavelength_um: float = 1.555,
) -> None:
    """Creates or updates the global Eigenmode Expansion (EME) simulation region.

    Sets physical cross-sectional boundaries, boundary conditions (BCs), background index,
    transverse mesh discretization, mode limits, and nominal source wavelength.

    Args:
        s: Active Lumerical MODE session handle.
        y_span: Total simulation region span along the y-axis in micrometers (μm).
        z_span: Total simulation region span along the z-axis in micrometers (μm).
        z_min_bc: Boundary condition applied to the minimum z boundary.
        z_max_bc: Boundary condition applied to the maximum z boundary.
        y_min_bc: Boundary condition applied to the minimum y boundary.
        y_max_bc: Boundary condition applied to the maximum y boundary.
        mesh_cells_y: Number of transverse mesh cells along the y-axis.
        mesh_cells_z: Number of transverse mesh cells along the z-axis.
        background_material: Background cladding material model from the material database.
        n_modes_all_cell_groups: Uniform global mode capacity count across all cell groups.
        center_wavelength_um: Central calculation wavelength in micrometers (μm).

    Raises:
        lumapi.LumApiError: If an unexpected error occurs querying the EME solver object.
    """
    lumapi = get_lumapi()
    try:
        _ = s.getnamed("EME")
        eme_exists = True
    except lumapi.LumApiError as e:
        msg = str(e)
        if "no items matching the name 'EME'" in msg or "no items matching the name" in msg:
            eme_exists = False
        else:
            raise

    if eme_exists:
        s.select("EME")
    else:
        s.addeme()

    s.set("x min", 0.0)
    s.set("y", 0.0)
    s.set("y span", y_span * 1e-6)
    s.set("z", 0.0)
    s.set("z span", z_span * 1e-6)

    if z_min_bc:
        s.set("z min bc", z_min_bc)
    if z_max_bc:
        s.set("z max bc", z_max_bc)
    if y_min_bc:
        s.set("y min bc", y_min_bc)
    if y_max_bc:
        s.set("y max bc", y_max_bc)

    if mesh_cells_y:
        s.set("mesh cells y", mesh_cells_y)
    if mesh_cells_z:
        s.set("mesh cells z", mesh_cells_z)
    if background_material:
        s.set("background material", background_material)

    if n_modes_all_cell_groups is not None:
        s.set("number of modes for all cell groups", n_modes_all_cell_groups)

    s.set("wavelength", center_wavelength_um * 1e-6)


def set_cell_basic(
    s: lumapi.MODE,
    n_cell_groups: int,
    group_spans: Sequence[float] | np.ndarray,
    cells_per_group: Sequence[int] | np.ndarray,
    subcell_methods: Sequence[bool] | np.ndarray,
    custom_eigensolver_settings: bool = True,
    n_modes: Sequence[int] | np.ndarray | int | None = None,
) -> None:
    """Configures cell group discretization, subcell methods, and modal bases in EME.

    Args:
        s: Active Lumerical MODE session handle.
        n_cell_groups: Total number of discrete longitudinal cell groups.
        group_spans: Physical length of each cell group in micrometers (μm).
        cells_per_group: Number of internal calculation cells allocated per cell group.
        subcell_methods: Subcell method toggle flags (True for CV/staircased, False for none).
        custom_eigensolver_settings: Whether custom per-cell-group solver settings are allowed.
        n_modes: Number of modes calculated per group. Can be a single integer applied to all
            groups or an array matching `n_cell_groups`.

    Raises:
        ValueError: If input array lengths do not match `n_cell_groups`.
    """
    subcell_methods_arr = np.asarray(subcell_methods, dtype=bool)
    cells_per_group_arr = np.asarray(cells_per_group, dtype=int)
    group_spans_arr = np.asarray(group_spans, dtype=float) * 1e-6

    # Dimension validation
    if n_cell_groups != len(group_spans_arr):
        raise ValueError("Length of group_spans should be equal to n_cell_groups!")
    if n_cell_groups != len(cells_per_group_arr):
        raise ValueError("Length of cells_per_group should be equal to n_cell_groups!")
    if n_cell_groups != len(subcell_methods_arr):
        raise ValueError("Length of subcell_methods should be equal to n_cell_groups!")

    s.select("EME")
    s.set("number of cell groups", n_cell_groups)
    s.set("group spans", group_spans_arr)
    s.set("cells", cells_per_group_arr)
    s.set("subcell method", subcell_methods_arr)
    s.set("allow custom eigensolver settings", custom_eigensolver_settings)

    if custom_eigensolver_settings and n_modes is not None:
        if np.isscalar(n_modes) or (isinstance(n_modes, np.ndarray) and n_modes.ndim == 0):
            n_modes_arr = np.full(n_cell_groups, n_modes, dtype=int)
        else:
            n_modes_arr = np.asarray(n_modes, dtype=int)
            if len(n_modes_arr) != n_cell_groups:
                raise ValueError(
                    f"Length of n_modes ({len(n_modes_arr)}) should be equal to n_cell_groups ({n_cell_groups})!"
                )

        s.set("modes", n_modes_arr)


def set_cell_radius(
    s: lumapi.MODE,
    bent: Sequence[bool | int] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
    orientations: Sequence[float] | np.ndarray,
    cells_per_group: int | Sequence[int] | np.ndarray,
) -> None:
    """Configures conformal bent waveguide eigensolver parameters per cell group.

    Sets curvature radius and orientation on the first cell of each group via batched
    script execution. Subsequent cells within each group inherit these settings.

    Args:
        s: Active Lumerical MODE session handle.
        bent: Flags indicating whether each group corresponds to a curved/bent waveguide.
        radii: Curvature bend radii in meters for each cell group.
        orientations: Bend orientation angles in degrees (e.g., 0 or 180).
        cells_per_group: Cell count per group (scalar integer or sequence matching groups).
    """
    bent_arr = np.asarray(bent, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    orientations_arr = np.asarray(orientations, dtype=float)
    n_groups = len(radii_arr)

    if n_groups == 0:
        return

    if np.isscalar(cells_per_group):
        cells_arr = np.full(n_groups, cells_per_group, dtype=int)
    else:
        cells_arr = np.asarray(cells_per_group, dtype=int)

    first_cell_indices = 1 + np.concatenate(([0], np.cumsum(cells_arr)[:-1]))

    print(f"🚀 Fast mode: total {n_groups} Groups, only setting the first cell...")

    script_commands = ["redrawoff;", "switchtolayout;"]

    for i in range(n_groups):
        is_bent = bent_arr[i]
        r = radii_arr[i]
        ori = orientations_arr[i]
        c_idx = first_cell_indices[i]

        script_commands.append(f"select('EME::Cells::cell_{c_idx}');")
        script_commands.append(f"seteigensolver('bent waveguide', {is_bent});")

        if is_bent:
            script_commands.append(f"seteigensolver('bend radius', {r});")
            script_commands.append(f"seteigensolver('bend orientation', {ori});")

    script_commands.append("redrawon;")

    chunk_size = 2000
    for i in range(0, len(script_commands), chunk_size):
        chunk = script_commands[i : i + chunk_size]
        s.eval("\n".join(chunk))

    print("✅ Bending setting has been finished!")


def set_ports(
    s: lumapi.MODE,
    selected_mode_numbers_port1: Sequence[int] | np.ndarray,
    selected_mode_numbers_port2: Sequence[int] | np.ndarray,
) -> None:
    """Configures active modal basis sets on Port 1 and Port 2 of the EME solver.

    Sets full simulation span coverage and activates user-selected mode indices.

    Args:
        s: Active Lumerical MODE session handle.
        selected_mode_numbers_port1: 1-based mode indices selected for Port 1.
        selected_mode_numbers_port2: 1-based mode indices selected for Port 2.
    """
    mode_nums_p1 = np.asarray(selected_mode_numbers_port1, dtype=int)
    mode_nums_p2 = np.asarray(selected_mode_numbers_port2, dtype=int)

    s.select("EME::Ports::port_1")
    s.set("use full simulation span", 1)
    s.set("mode selection", "user select")
    s.set("selected mode numbers", mode_nums_p1)
    s.updateportmodes(mode_nums_p1)

    s.select("EME::Ports::port_2")
    s.set("use full simulation span", 1)
    s.set("mode selection", "user select")
    s.set("selected mode numbers", mode_nums_p2)
    s.updateportmodes(mode_nums_p2)


def add_eme_index_profile(
    s: lumapi.MODE,
    x_max: float,
    name_profile: str = "overall_field",
    name_index: str = "index",
) -> None:
    """Adds or updates 2D Z-normal spatial field and refractive index profile monitors.

    Monitors match the global EME y-span across the propagation span [0, x_max].

    Args:
        s: Active Lumerical MODE session handle.
        x_max: Propagation maximum coordinate boundary in micrometers (μm).
        name_profile: Name identifier for the 2D EME field profile monitor.
        name_index: Name identifier for the 2D EME refractive index profile monitor.
    """
    s.selectall()
    existing = [obj.name for obj in s.getAllSelectedObjects()]

    if name_profile not in existing:
        s.addemeprofile()
    else:
        s.select(name_profile)
    s.set("name", name_profile)
    s.set("monitor type", "2D Z-normal")
    s.set("x min", 0.0)
    s.set("x max", x_max * 1e-6)
    s.set("z", 0.0)
    s.set("y", 0.0)
    y_span = s.getnamed("EME", "y span")
    s.set("y span", y_span)

    if name_index not in existing:
        s.addemeindex()
    else:
        s.select(name_index)
    s.set("name", name_index)
    s.set("monitor type", "2D Z-normal")
    s.set("x min", 0.0)
    s.set("x max", x_max * 1e-6)
    s.set("z", 0.0)
    s.set("y", 0.0)
    s.set("y span", y_span)


def add_xs_field_monitor(
    s: lumapi.MODE,
    x: float,
    name: str = "xs_field",
) -> None:
    """Adds or updates a 2D X-normal cross-sectional field monitor at coordinate x.

    Matches the EME solver cross-sectional dimensions (y-span and z-span).

    Args:
        s: Active Lumerical MODE session handle.
        x: Longitudinal x-position along the propagation direction in micrometers (μm).
        name: Name identifier for the cross-sectional monitor.

    Raises:
        lumapi.LumApiError: If an unexpected error occurs querying the monitor object.
    """
    lumapi = get_lumapi()
    try:
        _ = s.getnamed(name)
        obj_exists = True
    except lumapi.LumApiError as e:
        msg = str(e)
        if "no items matching the name" in msg:
            obj_exists = False
        else:
            raise

    if obj_exists:
        s.select(name)
    else:
        s.addemeprofile()

    s.set("name", name)
    s.set("monitor type", "2D X-normal")
    s.set("x", x * 1e-6)
    z_span = s.getnamed("EME", "z span")
    s.set("z", 0.0)
    s.set("z span", z_span)
    s.set("y", 0.0)
    y_span = s.getnamed("EME", "y span")
    s.set("y span", y_span)


def set_periodic_groups(
    s: lumapi.MODE,
    n_of_periodic_groups: int,
    periods_cell_groups: Sequence[int] | np.ndarray,
    start_cell_group: Sequence[int] | np.ndarray,
    end_cell_group: Sequence[int] | np.ndarray,
) -> None:
    """Configures periodic cascaded repetition blocks for EME cell groups.

    Allows fast cascaded scattering matrix evaluation across periodic sub-structures.

    Args:
        s: Active Lumerical MODE session handle.
        n_of_periodic_groups: Total number of distinct periodic repeat definitions.
        periods_cell_groups: Repetition multiplier count for each periodic group.
        start_cell_group: 1-based start cell group indices for each periodic range.
        end_cell_group: 1-based end cell group indices for each periodic range.
    """
    periods_arr = np.asarray(periods_cell_groups, dtype=int)
    start_arr = np.asarray(start_cell_group, dtype=int)
    end_arr = np.asarray(end_cell_group, dtype=int)

    s.select("EME")
    s.set("number of periodic groups", n_of_periodic_groups)
    s.set("start cell group", start_arr)
    s.set("end cell group", end_arr)
    s.set("periods", periods_arr)


def add_apodized_grating_group_for_eme(
    s: lumapi.MODE,
    group_params: Sequence[Dict[str, Any]],
    x0: float = 0.0,
    thick: float = 0.22,
    lead_length: float = 1.0,
    material: str = "Si (Silicon) - Palik",
    group_name: str = "Apodized_Grating_Group",
) -> float:
    """Constructs a compressed apodized grating structure group for HLP-EME modeling.

    Generates compressed grating composed by cascaded representative periods across all blocks 
           used in the HLP-EME. Periodic repetition is handled analytically downstream by the EME solver.

    Args:
        s: Active Lumerical MODE session handle.
        group_params: Sequence of dictionaries containing slice and geometry parameters
            (keys: 'group_index', 'slices' with 'z_length', 'width', 'x_upper', 'x_lower').
        x0: Origin starting offset along the x-axis in micrometers (μm).
        thick: Waveguide core layer thickness in micrometers (μm).
        lead_length: Length of straight input/output access leads in micrometers (μm).
        material: Core material model name from the Lumerical material database.
        group_name: Name identifier assigned to the created structure group.

    Returns:
        Total longitudinal length of the compressed grating in meters.

    Raises:
        Exception: If an unexpected error occurs querying existing objects.
    """
    try:
        _ = s.getnamed(group_name)
        grating_exists = True
    except Exception as e:
        msg = str(e)
        if "no items matching the name" in msg:
            grating_exists = False
        else:
            raise

    if grating_exists:
        s.select(group_name)
        s.delete(group_name)

    s.addstructuregroup()
    s.set("name", group_name)

    s.set("x", x0 * 1e-6)
    s.set("y", 0.0)
    s.set("z", 0.0)

    s.select(group_name)
    s.adduserprop("thick", 2, thick * 1e-6)
    s.adduserprop("material", 5, material)
    s.adduserprop("lead_length", 2, lead_length * 1e-6)

    script_content = [
        "deleteall;",
        "mat = %material%;",
        "th = %thick%;",
        "lead = %lead_length%;",
    ]

    first_slice = group_params[0]["slices"][0]
    w_in = first_slice["width"] * 1e-6
    c_in = 0.5 * (first_slice["x_upper"] + first_slice["x_lower"]) * 1e-6

    script_content.append(f"w_in = {w_in};")
    script_content.append(f"c_in = {c_in};")
    script_content.append("""
    if (lead > 0) {
        addrect;
        set("name", "input_lead");
        set("x min", -lead);
        set("x max", 0);
        set("y", c_in); set("y span", w_in);
        set("z", 0); set("z span", th);
        set("material", mat);
    }
    """)

    current_x_local = 0.0

    script_content.append("# --- Grating Single-Period Templates for EME ---")

    for g in group_params:
        g_idx = g["group_index"]
        slices = g["slices"]

        for s_idx, sl in enumerate(slices):
            length = sl["z_length"] * 1e-6
            w = sl["width"] * 1e-6
            c = 0.5 * (sl["x_upper"] + sl["x_lower"]) * 1e-6

            x_min = current_x_local
            x_max = current_x_local + length

            cmd = (
                f"addrect; "
                f"set('name', 'g{g_idx}_s{s_idx}'); "
                f"set('x min', {x_min}); "
                f"set('x max', {x_max}); "
                f"set('y', {c}); "
                f"set('y span', {w}); "
                f"set('z', 0); set('z span', th); "
                f"set('material', mat);"
            )
            script_content.append(cmd)

            current_x_local += length

    template_total_length = current_x_local

    last_slice = group_params[-1]["slices"][-1]
    w_out = last_slice["width"] * 1e-6
    c_out = 0.5 * (last_slice["x_upper"] + last_slice["x_lower"]) * 1e-6

    script_content.append(f"grating_len = {template_total_length};")
    script_content.append(f"w_out = {w_out};")
    script_content.append(f"c_out = {c_out};")
    script_content.append("""
    if (lead > 0) {
        addrect;
        set("name", "output_lead");
        set("x min", grating_len);
        set("x max", grating_len + lead);
        set("y", c_out); set("y span", w_out);
        set("z", 0); set("z span", th);
        set("material", mat);
    }
    """)

    final_script = "\n".join(script_content)
    s.set("script", final_script)

    return template_total_length



def add_mesh_grating(
    s: lumapi.MODE,
    x_max: float,
    dy_nm: int,
    dw_nm: int,
    w_wg: float,
    z_span: float = 0.22e-6,
    set: bool = False,
) -> None:
    """Adds localized transverse mesh overrides along sidewall corrugation regions.

    Creates upper and lower mesh override zones to accurately resolve spatial profiles of the grating edges.

    Args:
        s: Active Lumerical MODE session handle.
        x_max: Maximum span boundary along the propagation direction in micrometers (μm).
        dy_nm: Target transverse mesh pitch along the y-axis in nanometers (nm).
        dw_nm: Corrugation width in nanometers (nm) defining the mesh override width.
        w_wg: Nominal unperturbed waveguide width in micrometers (μm).
        z_span: Thickness of the mesh override volume in meters.
        set: Reserved configuration toggle flag.

    Raises:
        lumapi.LumApiError: If an unexpected error occurs querying existing mesh objects.
    """
    lumapi = get_lumapi()
    try:
        _ = s.getnamed("upper_mesh")
        obj_exists = True
    except lumapi.LumApiError as e:
        msg = str(e)
        if "no items matching the name" in msg:
            obj_exists = False
        else:
            raise

    if obj_exists:
        s.select("upper_mesh")
    else:
        s.addmesh()
        s.set("name", "upper_mesh")

    s.set("override x mesh", 0)
    s.set("override y mesh", 1)
    s.set("dy", dy_nm * 1e-9)
    s.set("override z mesh", 0)
    s.set("x min", 0.0)
    s.set("x max", x_max * 1e-6)
    s.set("y min", (w_wg / 2.0 - dw_nm / 2.0 / 1000.0) * 1e-6)
    s.set("y max", (w_wg / 2.0 + dw_nm / 2.0 / 1000.0) * 1e-6)
    s.set("z", 0.0)
    s.set("z span", z_span)

    try:
        _ = s.getnamed("lower_mesh")
        obj_exists = True
    except lumapi.LumApiError as e:
        msg = str(e)
        if "no items matching the name" in msg:
            obj_exists = False
        else:
            raise

    if obj_exists:
        s.select("lower_mesh")
        s.delete("lower_mesh")

    s.select("upper_mesh")
    s.copy(0.0, -w_wg * 1e-6)
    s.set("name", "lower_mesh")


def eme_cell_setting_data_gen(
    group_params: Sequence[Dict[str, Any]],
) -> Tuple[List[float], List[int], List[int], List[int], List[int]]:
    """Converts apodized grating parameters into structured EME cell and periodic inputs.

    Extracts group spans, slice counts per cell group, and periodic cascade ranges
    required by `set_cell_basic` and `set_periodic_groups`.

    Args:
        group_params: Sequence of group dictionaries containing `'period_um'`,
            `'slices'`, and `'n_repeats'`.

    Returns:
        A 5-tuple containing:
            - flat_group_spans (List[float]): Physical length of each unit template in μm.
            - cells_per_group (List[int]): Number of discrete slices within each group.
            - periodic_starts (List[int]): 1-based start group indices for periodicity.
            - periodic_ends (List[int]): 1-based end group indices for periodicity.
            - periodic_periods (List[int]): Periodic repetition counts per group.
    """
    flat_group_spans: List[float] = []
    periodic_periods: List[int] = []
    periodic_starts: List[int] = []
    periodic_ends: List[int] = []
    cells_per_group: List[int] = []
    current_eme_group_idx = 1

    for g in group_params:
        template_span = float(g["period_um"])
        n_slices = len(g["slices"])

        flat_group_spans.append(template_span)
        cells_per_group.append(n_slices)
        periodic_starts.append(current_eme_group_idx)
        periodic_ends.append(current_eme_group_idx)
        periodic_periods.append(int(g["n_repeats"]))

        current_eme_group_idx += 1

    return (
        flat_group_spans,
        cells_per_group,
        periodic_starts,
        periodic_ends,
        periodic_periods,
    )


def save_lms_file(
    s: lumapi.MODE,
    w_wg_um: float,
    period_nm: float,
    design_name: str,
    dw_nm: float,
    n_blocks: int,
    n_periods_per_block: int,
    n_slices_per_period: int,
    n_modes: Optional[int] = None,
    mc: bool = True,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, Optional[str]]:
    """Generates a standardized simulation filename and optionally saves the `.lms` file.

    Encodes device dimensions and solver parameters into the filename, replacing
    decimal points with 'p' to maintain file system compatibility.

    Args:
        s: Active Lumerical MODE session handle.
        w_wg_um: Waveguide width in micrometers (μm).
        period_nm: Grating period in nanometers (nm).
        design_name: Base design tag identifier.
        dw_nm: Corrugation depth offset in nanometers (nm).
        n_blocks: Number of apodization blocks.
        n_periods_per_block: Period count per apodization block.
        n_slices_per_period: Discretized slice count per grating period.
        n_modes: Optional mode capacity count included in the file name.
        mc: If True, prefixes filename with `'mc_'`.
        save: If True, writes the active session to disk.
        save_path: Directory or explicit file path for saving. Defaults to the current
            working directory if None.

    Returns:
        A tuple `(sim_file_name, actual_save_path)` containing:
            - sim_file_name (str): The constructed base simulation filename without suffix.
            - actual_save_path (Optional[str]): Absolute string path to the saved file, or
              None if `save=False`.
    """
    parts = [
        f"{design_name}",
        f"w{w_wg_um}",
        f"p{period_nm}",
        f"dw{dw_nm}",
        f"nb{n_blocks}",
        f"np{n_periods_per_block}",
        f"ns{n_slices_per_period}",
    ]
    if n_modes is not None:
        parts.append(f"nm{n_modes}")

    core_name = "_".join(parts).replace(".", "p")
    sim_file_name = f"mc_{core_name}" if mc else core_name

    actual_save_path: Optional[Path] = None

    if save:
        if save_path is None:
            actual_save_path = Path.cwd() / f"{sim_file_name}.lms"
        else:
            p = Path(save_path)
            if p.is_dir():
                actual_save_path = p / f"{sim_file_name}.lms"
            else:
                actual_save_path = p if p.suffix == ".lms" else p.with_suffix(".lms")

        actual_save_path.parent.mkdir(parents=True, exist_ok=True)
        s.save(str(actual_save_path))
        print(f"[hlp-eme] .lms file has been saved to: \n {actual_save_path}")

    return sim_file_name, str(actual_save_path) if actual_save_path else None


def run_eme_lam_sweep(
    s: lumapi.MODE,
    start_lam_um: float,
    end_lam_um: float,
    n_lams: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs an EME wavelength sweep and extracts S-parameter frequency response data.

    Enables group delay calculations and returns sampled wavelength points along
    with the raw scattering dataset.

    Args:
        s: Active Lumerical MODE session handle.
        start_lam_um: Initial sweep wavelength in micrometers (μm).
        end_lam_um: Final sweep wavelength in micrometers (μm).
        n_lams: Total number of spectral wavelength sample points.

    Returns:
        A tuple `(lams_nm_eme, sweep_data)` containing:
            - lams_nm_eme (np.ndarray): 1D array of sampled wavelengths in nanometers (nm).
            - sweep_data (Dict[str, Any]): Raw dictionary containing the S-matrix sweep dataset
              extracted from `S_wavelength_sweep`.
    """
    s.setemeanalysis("wavelength sweep", 1)
    s.setemeanalysis("start wavelength", start_lam_um * 1e-6)
    s.setemeanalysis("stop wavelength", end_lam_um * 1e-6)
    s.setemeanalysis("number of wavelength points", n_lams)
    s.setemeanalysis("calculate group delays for wavelength sweep", 1)
    s.emesweep("wavelength sweep")
    sweep_data = s.getemesweep("S_wavelength_sweep")
    lams_nm_eme = sweep_data["wavelength"].squeeze() * 1e9
    return lams_nm_eme, sweep_data