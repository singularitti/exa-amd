#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from pymatgen.core import Composition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a ternary hull-style phase diagram from an ehull CSV."
    )
    parser.add_argument("--ehull-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--chemical-system", required=True, help="Element order, e.g. Mn-Bi-O.")
    parser.add_argument("--unstable-max", type=float, default=0.1)
    parser.add_argument("--label-stable", action="store_true", default=True)
    return parser.parse_args()


def to_xy(fractions: np.ndarray) -> tuple[float, float]:
    x = float(fractions[0] + 0.5 * fractions[1])
    y = float((math.sqrt(3.0) / 2.0) * fractions[1])
    return x, y


def main() -> None:
    args = parse_args()
    elements = args.chemical_system.split("-")
    target_system = "-".join(sorted(elements))

    stable = []
    unstable = []

    with args.ehull_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["chemical_system"] != target_system:
                continue
            comp = Composition(row["formula"])
            # Map to left, top, right to match previous exa-amd-style layout:
            # left=elements[0], top=elements[2], right=elements[1]
            frac = np.array(
                [
                    comp.get_atomic_fraction(elements[0]),
                    comp.get_atomic_fraction(elements[2]),
                    comp.get_atomic_fraction(elements[1]),
                ],
                dtype=float,
            )
            x, y = to_xy(frac)
            entry = {
                "formula": comp.reduced_formula,
                "x": x,
                "y": y,
                "ehull": float(row["ehull"]),
            }
            if abs(entry["ehull"]) <= 1e-8:
                stable.append(entry)
            elif entry["ehull"] <= args.unstable_max:
                unstable.append(entry)

    if not stable:
        raise ValueError(f"No stable entries found for {args.chemical_system}")

    fig, ax = plt.subplots(figsize=(11, 9))

    # Outer triangle
    triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
            [0.0, 0.0],
        ]
    )
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=2.0)

    stable_xy = np.array([[e["x"], e["y"]] for e in stable])
    if len(stable_xy) >= 3:
        hull = ConvexHull(stable_xy)
        for simplex in hull.simplices:
            pts = stable_xy[simplex]
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.2, zorder=1)

    ax.scatter(
        stable_xy[:, 0],
        stable_xy[:, 1],
        s=90,
        c="#53b64c",
        edgecolors="#1f1f1f",
        linewidths=1.0,
        zorder=3,
    )

    if unstable:
        unstable_xy = np.array([[e["x"], e["y"]] for e in unstable])
        ax.scatter(
            unstable_xy[:, 0],
            unstable_xy[:, 1],
            s=55,
            marker="s",
            c="red",
            edgecolors="#1f1f1f",
            linewidths=0.8,
            zorder=2,
        )

    if args.label_stable:
        for entry in stable:
            ax.text(
                entry["x"] + 0.008,
                entry["y"] + 0.008,
                entry["formula"],
                fontsize=16,
                weight="bold",
                ha="left",
                va="center",
            )

    ax.text(-0.02, -0.025, elements[0], fontsize=18, weight="bold", ha="right", va="top")
    ax.text(1.02, -0.025, elements[1], fontsize=18, weight="bold", ha="left", va="top")
    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.02, elements[2], fontsize=18, weight="bold", ha="center", va="bottom")

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.06, math.sqrt(3.0) / 2.0 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
