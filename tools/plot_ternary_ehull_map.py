#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram, order_phase_diagram
from pymatgen.core import Composition

from tools.pymatgen_ehull import _build_pd_entry, load_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a ternary composition map colored by energy above hull."
    )
    parser.add_argument("--ehull-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--chemical-system", required=True, help="Element order, e.g. Mn-Bi-F.")
    parser.add_argument("--single-csv", type=Path, nargs="+")
    parser.add_argument("--binary-csv", type=Path, nargs="+")
    parser.add_argument("--ternary-csv", type=Path, nargs="+")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument(
        "--energy-mode",
        choices=("per-atom", "total"),
        default="per-atom",
    )
    parser.add_argument(
        "--replace-cif-prefix",
        nargs=2,
        action="append",
        metavar=("OLD_PREFIX", "NEW_PREFIX"),
        help="Rewrite CIF paths on the fly if the source CSV stores an outdated root.",
    )
    parser.add_argument(
        "--ehull-below",
        type=float,
        default=None,
        help="Optional strict upper threshold; only keep rows with Ehull below this value.",
    )
    parser.add_argument(
        "--ehull-max",
        type=float,
        default=0.2,
        help="Upper end of the color scale in eV/atom; larger values are clipped.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.5,
        help="Scatter marker area in points^2.",
    )
    parser.add_argument(
        "--hull-linewidth",
        type=float,
        default=1.2,
        help="Line width for the stable-hull overlay from pymatgen.",
    )
    return parser.parse_args()


def ternary_to_cartesian(left: np.ndarray, right: np.ndarray, top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = right + 0.5 * top
    y = (math.sqrt(3.0) / 2.0) * top
    return x, y


def load_hull_lines(args: argparse.Namespace, elements: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not (args.single_csv and args.binary_csv and args.ternary_csv):
        return []

    replace_prefixes = tuple(tuple(item) for item in args.replace_cif_prefix) if args.replace_cif_prefix else ()
    all_entries = []
    for csv_path in args.single_csv:
        all_entries.extend(load_entries(csv_path, "single", args.jobs, args.energy_mode, replace_prefixes))
    for csv_path in args.binary_csv:
        all_entries.extend(load_entries(csv_path, "binary", args.jobs, args.energy_mode, replace_prefixes))
    for csv_path in args.ternary_csv:
        all_entries.extend(load_entries(csv_path, "ternary", args.jobs, args.energy_mode, replace_prefixes))

    target_elements = set(elements)
    compatible_entries = [
        entry for entry in all_entries if set(entry.elements).issubset(target_elements)
    ]
    phase_diagram = PhaseDiagram([_build_pd_entry(entry) for entry in compatible_entries])
    plotter = PDPlotter(phase_diagram, backend="matplotlib")
    lines, stable_entries, unstable_entries = plotter.pd_plot_data
    ordered_lines, _, _, _ = order_phase_diagram(
        lines,
        stable_entries,
        unstable_entries,
        [elements[2], elements[0], elements[1]],
    )
    return [(np.asarray(x), np.asarray(y)) for x, y in ordered_lines]


def main() -> None:
    args = parse_args()
    elements = args.chemical_system.split("-")
    if len(elements) != 3:
        raise ValueError(f"Expected a ternary system, got {args.chemical_system!r}")
    target_system = "-".join(sorted(elements))

    xs: list[float] = []
    ys: list[float] = []
    ehulls: list[float] = []
    stable_xs: list[float] = []
    stable_ys: list[float] = []
    hull_lines = load_hull_lines(args, elements)
    filtered_out = 0

    with args.ehull_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["chemical_system"] != target_system:
                continue
            ehull = float(row["ehull"])
            if args.ehull_below is not None and ehull < args.ehull_below:
                pass
            elif args.ehull_below is not None:
                filtered_out += 1
                continue
            if ehull > args.ehull_max:
                filtered_out += 1
                continue
            composition = Composition(row["formula"])
            fractions = np.array(
                [composition.get_atomic_fraction(el) for el in elements],
                dtype=float,
            )
            x, y = ternary_to_cartesian(fractions[0], fractions[1], fractions[2])
            xs.append(float(x))
            ys.append(float(y))
            ehulls.append(ehull)
            if abs(ehull) <= 1e-8:
                stable_xs.append(float(x))
                stable_ys.append(float(y))

    if not xs:
        threshold_text = (
            f" with Ehull < {args.ehull_below}"
            if args.ehull_below is not None
            else f" with Ehull <= {args.ehull_max}"
        )
        raise ValueError(
            f"No rows found for chemical system {args.chemical_system}{threshold_text}. "
            f"Filtered out {filtered_out} rows."
        )

    fig, ax = plt.subplots(figsize=(10, 9))
    norm = Normalize(vmin=0.0, vmax=args.ehull_max)
    cmap = plt.get_cmap("viridis_r")

    scatter = ax.scatter(
        xs,
        ys,
        c=ehulls,
        s=args.point_size,
        cmap=cmap,
        norm=norm,
        linewidths=0,
        rasterized=True,
    )
    if stable_xs:
        ax.scatter(
            stable_xs,
            stable_ys,
            s=8,
            facecolors="none",
            edgecolors="black",
            linewidths=0.4,
            rasterized=True,
        )

    for x_coords, y_coords in hull_lines:
        ax.plot(
            x_coords,
            y_coords,
            color="black",
            linewidth=args.hull_linewidth,
            zorder=2,
        )

    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=1.0)

    ax.text(-0.03, -0.04, elements[0], ha="right", va="top", fontsize=12)
    ax.text(1.03, -0.04, elements[1], ha="left", va="top", fontsize=12)
    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.04, elements[2], ha="center", va="bottom", fontsize=12)

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3.0) / 2.0 + 0.08)
    ax.axis("off")
    ax.set_title(f"{args.chemical_system} composition map colored by Ehull")

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85, pad=0.04)
    colorbar.set_label("Ehull (eV/atom)")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
