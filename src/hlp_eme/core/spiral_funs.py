from typing import List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


class PerfectSpiral:
    """Archimedean spiral waveguide geometry generator for optical EME simulations.

    Generates a continuous trajectory composed of an inward Archimedean spiral arm,
    a central S-bend transition, and an outward Archimedean spiral arm, partitioned
    into discrete slice segments.

    Attributes:
        rmin (float): Minimum inner clearance radius in meters.
        deltaw (float): Radial separation between adjacent waveguide tracks in meters.
        group_spans (np.ndarray): Array of discrete segment lengths along the path in meters.
        n_cells (int): Total number of discrete cells/segments.
        effective_pitch (float): Pitch of the spiral arms in meters ($2 \\times \\delta w$).
        k (float): Archimedean spiral growth constant $dr/d\\theta = \\delta w / \\pi$ in m/rad.
        r_out_start (float): Starting radius of the outward spiral arm in meters.
        r_in_end_radius (float): Ending radius of the inward spiral arm in meters.
        rs_bend (float): Curvature radius of the two circular S-bend arcs in meters.
        xc_s1 (float): Center x-coordinate of the first S-bend arc in meters.
        xc_s2 (float): Center x-coordinate of the second S-bend arc in meters.
        len_s_total (float): Total arc length of the complete central S-bend in meters.
        len_arm (float): Arc length of a single spiral arm (inward or outward) in meters.
        m1 (float): Cumulative arc length milestone at end of inward arm in meters.
        m2 (float): Cumulative arc length milestone at midpoint of S-bend in meters.
        m3 (float): Cumulative arc length milestone at end of S-bend in meters.
        total_len (float): Total path length across all segments in meters.
    """

    def __init__(
        self,
        rmin_um: float,
        deltaw_um: float,
        group_spans_um: Sequence[float],
    ) -> None:
        """Initializes the spiral geometry with structural dimensions and segment spans.

        Args:
            rmin_um: Minimum inner radius in micrometers ($\mu$m).
            deltaw_um: Waveguide track-to-track pitch/spacing in micrometers ($\mu$m).
            group_spans_um: Sequence of individual slice lengths along the path in micrometers ($\mu$m).
        """
        # 1. Unit conversion: um -> m
        self.rmin: float = float(rmin_um) * 1e-6
        self.deltaw: float = float(deltaw_um) * 1e-6

        # Convert input segment lengths from um to meters and store as an array
        self.group_spans: np.ndarray = np.asarray(group_spans_um, dtype=np.float64) * 1e-6
        self.n_cells: int = len(self.group_spans)

        # 2. Core parameter calculations
        self.effective_pitch: float = 2.0 * self.deltaw
        self.k: float = self.deltaw / np.pi

        self.r_out_start: float = self.rmin
        self.r_in_end_radius: float = self.rmin

        # Dynamic calculation of S-bend radius
        self.rs_bend: float = (self.r_in_end_radius + self.r_out_start) / 4.0

        # Center coordinates of the two circular arcs forming the S-bend
        self.xc_s1: float = -(self.r_in_end_radius) + self.rs_bend
        self.xc_s2: float = self.r_out_start - self.rs_bend
        self.len_s_total: float = 2.0 * np.pi * self.rs_bend

        # 3. Path lengths and transition milestone coordinates
        total_target_len: float = float(np.sum(self.group_spans))
        remaining_len: float = total_target_len - self.len_s_total

        if remaining_len < 0:
            print("Warning: Spiral target length is shorter than S-bend. Adjusting...")
            self.len_arm: float = 0.0
            self.m1: float = 0.0
            self.m2: float = self.len_s_total / 2.0
            self.m3: float = self.len_s_total
            self.total_len: float = self.len_s_total
        else:
            self.len_arm = remaining_len / 2.0
            self.m1 = self.len_arm
            self.m2 = self.m1 + (self.len_s_total / 2.0)
            self.m3 = self.m2 + (self.len_s_total / 2.0)
            self.total_len = self.m3 + self.len_arm

    def _radius(self, theta: float, r0: float) -> float:
        """Calculates radial distance on the Archimedean spiral for a given angle.

        Args:
            theta: Polar angle in radians.
            r0: Base inner radius at $\theta = 0$ in meters.

        Returns:
            Radial distance $r(\\theta) = r_0 + k \\cdot \\theta$ in meters.
        """
        return r0 + self.k * theta

    def _ds_dtheta(self, theta: float, r0: float) -> float:
        """Computes the differential arc length $ds/d\\theta$ for the spiral.

        Args:
            theta: Polar angle in radians.
            r0: Base inner radius at $\theta = 0$ in meters.

        Returns:
            Differential rate of change of arc length $\\sqrt{r(\\theta)^2 + k^2}$ in m/rad.
        """
        return np.sqrt((self._radius(theta, r0)) ** 2 + self.k**2)

    def _arc_len(self, theta: float, r0: float) -> float:
        """Integrates differential arc length from 0 to $\theta$ using numerical quadrature.

        Args:
            theta: Upper integration limit polar angle in radians.
            r0: Base inner radius at $\theta = 0$ in meters.

        Returns:
            Integrated arc length in meters.
        """
        length, _ = quad(self._ds_dtheta, 0, theta, args=(r0,))
        return float(length)

    def _solve_theta(self, length: float, r0: float) -> float:
        """Solves the inverse arc-length problem using Brent's root-finding method.

        Args:
            length: Target arc length along the spiral arm in meters.
            r0: Base inner radius at $\theta = 0$ in meters.

        Returns:
            Polar angle $\theta$ (in radians) corresponding to the requested arc length.
        """
        if length < 1e-9:
            return 0.0
        upper = length / r0 * 2.0 + 20.0
        return float(brentq(lambda t: self._arc_len(t, r0) - length, 0.0, upper))

    def _get_curvature_radius_spiral(self, r_local: float) -> float:
        """Calculates local radius of curvature for an Archimedean spiral.

        Args:
            r_local: Local radial distance $r$ from the coordinate origin in meters.

        Returns:
            Radius of curvature $R_c = (r^2 + k^2)^{1.5} / (r^2 + 2k^2)$ in meters.
        """
        num = (r_local**2 + self.k**2) ** 1.5
        den = r_local**2 + 2.0 * self.k**2
        return float(num / den)

    def get_point(self, s: float) -> Tuple[float, float, int]:
        """Evaluates Cartesian coordinates and region identifier at a specified arc length.

        Args:
            s: Cumulative arc length from start of the waveguide in meters ($0 \\le s \\le \\text{total\\_len}$).

        Returns:
            A tuple `(x, y, rid)` containing:
                - x (float): x-coordinate in meters.
                - y (float): y-coordinate in meters.
                - rid (int): Region identifier:
                    - `0`: Inward Archimedean spiral arm
                    - `1`: First half of central S-bend (arc centered at `xc_s1`)
                    - `2`: Second half of central S-bend (arc centered at `xc_s2`)
                    - `3`: Outward Archimedean spiral arm
        """
        if s <= self.m1:
            dist = self.m1 - s
            theta = self._solve_theta(dist, self.r_in_end_radius)
            r = self._radius(theta, self.r_in_end_radius)
            return -r * np.cos(theta), -r * np.sin(theta), 0
        elif s <= self.m2:
            dist = s - self.m1
            angle = np.pi - (dist / self.rs_bend)
            return self.xc_s1 + self.rs_bend * np.cos(angle), self.rs_bend * np.sin(angle), 1
        elif s <= self.m3:
            dist = s - self.m2
            angle = np.pi - (dist / self.rs_bend)
            return self.xc_s2 + self.rs_bend * np.cos(angle), -self.rs_bend * np.sin(angle), 2
        else:
            dist = s - self.m3
            theta = self._solve_theta(dist, self.r_out_start)
            r = self._radius(theta, self.r_out_start)
            return r * np.cos(theta), r * np.sin(theta), 3

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, None]:
        """Discretizes trajectory into cell midpoints according to `group_spans`.

        Returns:
            A tuple `(xs, ys, ids, bends, None)` where:
                - xs (np.ndarray): Center x-coordinates for each slice in meters.
                - ys (np.ndarray): Center y-coordinates for each slice in meters.
                - ids (np.ndarray): Region IDs (0, 1, 2, or 3) for each slice.
                - bends (np.ndarray): Bend flag array initialized to zeros.
                - None: Reserved placeholder for downstream layout metadata.
        """
        xs: List[float] = []
        ys: List[float] = []
        ids: List[int] = []
        bends: List[int] = []
        s_current = 0.0

        for span in self.group_spans:
            s_end = min(s_current + span, self.total_len)
            actual_span = s_end - s_current
            if actual_span < 1e-15:
                break

            s_mid = s_current + actual_span / 2.0
            x, y, rid = self.get_point(s_mid)
            xs.append(x)
            ys.append(y)
            ids.append(rid)
            bends.append(0)
            s_current = s_end

        return np.array(xs), np.array(ys), np.array(ids), np.array(bends), None

    def get_vis_path(self, num_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a high-density trajectory coordinate array for smooth path plotting.

        Args:
            num_points: Number of evaluation sample points along the entire path length.

        Returns:
            A tuple `(px, py)` containing:
                - px (np.ndarray): Array of sampled x-coordinates in meters.
                - py (np.ndarray): Array of sampled y-coordinates in meters.
        """
        ss = np.linspace(0, self.total_len, num_points)
        pts = [self.get_point(s) for s in ss]
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    def get_eme_parameters_with_coords(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts optical Eigenmode Expansion (EME) cell parameters across all slices.

        Computes section span, local bend radius, curvature orientation angle, and
        midpoint Cartesian coordinates for each slice segment.

        Returns:
            A tuple `(group_spans, radii_curv, orientations, coords)` where:
                - group_spans (np.ndarray): Actual arc length of each slice in meters.
                - radii_curv (np.ndarray): Local radius of curvature for each slice in meters.
                - orientations (np.ndarray): Bend orientation angles in degrees (`180` or `0`).
                - coords (np.ndarray): Array of shape `(N, 2)` containing `[x, y]` coordinates in meters.
        """
        group_spans: List[float] = []
        radii_curv: List[float] = []
        orientations: List[int] = []
        coords: List[List[float]] = []
        s_current = 0.0

        for span in self.group_spans:
            s_end = min(s_current + span, self.total_len)
            actual_span = s_end - s_current

            if actual_span < 1e-15:
                break

            s_mid = s_current + actual_span / 2.0

            # 1. Obtain center point coordinates (x, y) in meters
            cx, cy, rid = self.get_point(s_mid)
            coords.append([cx, cy])

            # 2. Determine radius of curvature and orientation angle
            r_c: float = np.inf
            orient: int = 0
            if rid == 0:
                dist = self.m1 - s_mid
                theta = self._solve_theta(dist, self.r_in_end_radius)
                r_local = self._radius(theta, self.r_in_end_radius)
                r_c = self._get_curvature_radius_spiral(r_local)
                orient = 180
            elif rid == 1:
                r_c = self.rs_bend
                orient = 180
            elif rid == 2:
                r_c = self.rs_bend
                orient = 0
            elif rid == 3:
                dist = s_mid - self.m3
                theta = self._solve_theta(dist, self.r_out_start)
                r_local = self._radius(theta, self.r_out_start)
                r_c = self._get_curvature_radius_spiral(r_local)
                orient = 0

            group_spans.append(actual_span)
            radii_curv.append(r_c)
            orientations.append(orient)

            s_current = s_end

        return (
            np.array(group_spans),
            np.array(radii_curv),
            np.array(orientations),
            np.array(coords),
        )


def plot_analytical_spiral(
    generator: PerfectSpiral,
    fig_num: Optional[Union[int, str]] = None,
) -> None:
    """Renders the analytical spiral waveguide trajectory and discretized cell midpoints.

    Converts internal SI meter coordinates into micrometers ($\mu$m) for display.
    Discretized cell midpoints are color-coded based on their geometric segment ID.

    Args:
        generator: Configured `PerfectSpiral` instance providing analytical paths and midpoints.
        fig_num: Optional figure number or title passed to `matplotlib.pyplot.figure`.
    """
    xs, ys, ids, _, _ = generator.generate()
    px, py = generator.get_vis_path()

    if fig_num:
        plt.figure(fig_num, figsize=(8, 8))
    else:
        plt.figure(figsize=(8, 8))

    # Convert meters back to micrometers for plotting
    plt.plot(px * 1e6, py * 1e6, "k-", alpha=0.2, linewidth=3, label="Analytical Path")

    # Subsample scatter points to avoid overplotting
    idx_sample = np.arange(0, len(xs), max(1, len(xs) // 200))
    plt.scatter(
        xs[idx_sample] * 1e6,
        ys[idx_sample] * 1e6,
        s=10,
        c=ids[idx_sample],
        cmap="brg",
        label="Cell Centers",
    )

    plt.title(r"Analytical Spiral Geometry ($\mu$m)")
    plt.xlabel(r"$x$ ($\mu$m)")
    plt.ylabel(r"$y$ ($\mu$m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()