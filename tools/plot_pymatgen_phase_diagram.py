#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pymatgen.analysis.phase_diagram import PDEntry, PDPlotter, PhaseDiagram
from pymatgen.core import Composition

from tools.pymatgen_ehull import ParsedEntry, _build_pd_entry, load_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a ternary phase diagram with pymatgen.PDPlotter."
    )
    parser.add_argument("--single-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--binary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--ternary-csv", type=Path, nargs="+")
    parser.add_argument("--ehull-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-data-csv", type=Path)
    parser.add_argument("--chemical-system", default="Bi-Mn-O")
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
        help="Rewrite CIF paths on the fly if the CSV stores an outdated root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--show-unstable",
        type=float,
        default=0.05,
        help="Only plot unstable ternary entries with Ehull <= this cutoff.",
    )
    parser.add_argument(
        "--min-ehull",
        type=float,
        default=0.0,
        help="Only plot unstable ternary entries with Ehull strictly greater than this value.",
    )
    parser.add_argument(
        "--color-by-ehull",
        action="store_true",
        help="Overlay displayed entries with colors based on the Ehull values from --ehull-csv.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=10.0,
        help="Scatter marker area for the colored point overlay.",
    )
    parser.add_argument(
        "--ehull-cmap",
        default="inferno",
        help="Matplotlib colormap name for the Ehull overlay.",
    )
    parser.add_argument(
        "--use-ehull-csv-for-ternary",
        action="store_true",
        help="Load ternary formulas, num_atoms, energies, and Ehull directly from --ehull-csv instead of parsing ternary CIFs.",
    )
    parser.add_argument(
        "--keep-element-labels",
        action="store_true",
        help="Keep only the three elemental corner labels.",
    )
    parser.add_argument(
        "--hide-pd-mesh",
        action="store_true",
        help="Remove PDPlotter linework and redraw only the outer triangle boundary.",
    )
    parser.add_argument(
        "--label-stable",
        action="store_true",
        help="Show labels for stable entries.",
    )
    parser.add_argument(
        "--label-unstable",
        action="store_true",
        help="Show labels for unstable entries.",
    )
    return parser.parse_args()


def read_ehulls(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values[row["entry_id"]] = float(row["ehull"])
    return values


@dataclass(frozen=True)
class EhullRow:
    entry_id: str
    source: str
    chemical_system: str
    formula: str
    num_atoms: float
    input_energy: float
    energy_per_atom: float
    ehull: float


def read_ehull_rows(path: Path) -> list[EhullRow]:
    rows: list[EhullRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                EhullRow(
                    entry_id=row["entry_id"],
                    source=row["source"],
                    chemical_system=row["chemical_system"],
                    formula=row["formula"],
                    num_atoms=float(row["num_atoms"]),
                    input_energy=float(row["input_energy"]),
                    energy_per_atom=float(row["energy_per_atom"]),
                    ehull=float(row["ehull"]),
                )
            )
    return rows


def select_ternary_ids(
    ehull_rows: list[EhullRow],
    chemical_system: str,
    min_ehull: float,
    show_unstable: float,
) -> set[str]:
    target_system = "-".join(sorted(chemical_system.split("-")))
    return {
        row.entry_id
        for row in ehull_rows
        if row.chemical_system == target_system
        and (
            abs(row.ehull) <= 1e-8
            or (row.ehull > min_ehull and row.ehull <= show_unstable)
        )
    }


def load_ternary_entries_from_ehull_csv(
    ehull_rows: list[EhullRow],
    chemical_system: str,
    min_ehull: float,
    show_unstable: float,
) -> list[ParsedEntry]:
    target_system = "-".join(sorted(chemical_system.split("-")))
    entries: list[ParsedEntry] = []
    for row in ehull_rows:
        if row.chemical_system != target_system:
            continue
        if not (abs(row.ehull) <= 1e-8 or (row.ehull > min_ehull and row.ehull <= show_unstable)):
            continue
        composition = Composition(row.formula)
        entries.append(
            ParsedEntry(
                entry_id=row.entry_id,
                source=row.source,
                formula=row.formula,
                reduced_formula=row.formula,
                chemical_system=row.chemical_system,
                elements=tuple(sorted(str(el) for el in composition.elements)),
                num_atoms=row.num_atoms,
                input_energy=row.input_energy,
                energy_per_atom=row.energy_per_atom,
            )
        )
    return entries


def build_display_entries(
    all_entries: list[ParsedEntry],
    ehulls: dict[str, float],
    chemical_system: str,
    min_ehull: float,
    show_unstable: float,
) -> list[ParsedEntry]:
    target_elements = set(chemical_system.split("-"))
    compatible_entries = [
        entry for entry in all_entries if set(entry.elements).issubset(target_elements)
    ]
    full_phase_diagram = PhaseDiagram([_build_pd_entry(entry) for entry in compatible_entries])
    stable_ids = {entry.name for entry in full_phase_diagram.stable_entries}

    display_entries: list[ParsedEntry] = []
    for entry in compatible_entries:
        if entry.entry_id in stable_ids:
            display_entries.append(entry)
            continue
        if entry.source != "ternary":
            continue
        ehull = ehulls.get(entry.entry_id)
        if ehull is not None and ehull > min_ehull and ehull <= show_unstable:
            display_entries.append(entry)
    return display_entries


def build_plot_entry(parsed: ParsedEntry) -> PDEntry:
    composition = Composition(parsed.formula)
    return PDEntry(
        composition,
        parsed.energy_per_atom * parsed.num_atoms,
        name=parsed.reduced_formula,
        attribute=parsed.entry_id,
    )


def overlay_ehull_colors(
    ax: plt.Axes,
    plotter: PDPlotter,
    ehulls: dict[str, float],
    marker_size: float,
    cmap_name: str,
) -> None:
    _, stable_entries, unstable_entries = plotter.pd_plot_data

    stable_x: list[float] = []
    stable_y: list[float] = []
    stable_ehulls: list[float] = []
    for coords, entry in stable_entries.items():
        entry_id = entry.attribute
        if entry_id is None or entry_id not in ehulls:
            continue
        stable_x.append(coords[0])
        stable_y.append(coords[1])
        stable_ehulls.append(ehulls[entry_id])

    unstable_x: list[float] = []
    unstable_y: list[float] = []
    unstable_ehulls: list[float] = []
    for entry, coords in unstable_entries.items():
        entry_id = entry.attribute
        if entry_id is None or entry_id not in ehulls:
            continue
        unstable_x.append(coords[0])
        unstable_y.append(coords[1])
        unstable_ehulls.append(ehulls[entry_id])

    values = stable_ehulls + unstable_ehulls
    if not values:
        return

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    if stable_x:
        ax.scatter(
            stable_x,
            stable_y,
            c=stable_ehulls,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            edgecolors="none",
            linewidths=0,
            zorder=4,
        )
    if unstable_x:
        ax.scatter(
            unstable_x,
            unstable_y,
            c=unstable_ehulls,
            cmap=cmap,
            norm=norm,
            s=marker_size,
            marker="o",
            edgecolors="none",
            linewidths=0,
            zorder=4,
        )

    colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.82, pad=0.04)
    colorbar.set_label("Ehull (eV/atom)")


def restyle_pdplotter_axes(
    ax: plt.Axes,
    plotter: PDPlotter,
    chemical_system: str,
    keep_element_labels: bool,
    hide_pd_mesh: bool,
) -> None:
    _, stable_entries, _ = plotter.pd_plot_data
    element_names = set(chemical_system.split("-"))
    element_coords: list[tuple[float, float, str]] = []
    for coords, entry in stable_entries.items():
        if len(entry.composition.elements) != 1:
            continue
        symbol = str(entry.composition.elements[0])
        if symbol in element_names:
            element_coords.append((coords[0], coords[1], symbol))

    if hide_pd_mesh:
        for line in list(ax.lines):
            if isinstance(line, Line2D):
                line.remove()
        if len(element_coords) == 3:
            ordered = sorted(element_coords, key=lambda item: (item[1], item[0]))
            left = ordered[0]
            right = ordered[1]
            top = ordered[2]
            ax.plot(
                [left[0], right[0], top[0], left[0]],
                [left[1], right[1], top[1], left[1]],
                color="black",
                linewidth=1.5,
                zorder=2,
            )

    for text in list(ax.texts):
        text.remove()

    if keep_element_labels:
        for x, y, symbol in element_coords:
            dx = 0.0
            dy = 0.0
            ha = "center"
            va = "center"
            if y > 0.5:
                dy = 0.035
                va = "bottom"
            elif x < 0.2:
                dx = -0.055
                ha = "right"
            else:
                dx = 0.03
                ha = "left"
            ax.text(x + dx, y + dy, symbol, fontsize=16, ha=ha, va=va, zorder=5)


def write_display_data(
    output_csv: Path,
    display_entries: list[ParsedEntry],
    ehulls: dict[str, float],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entry_id",
        "source",
        "formula",
        "chemical_system",
        "num_atoms",
        "input_energy",
        "energy_per_atom",
        "ehull",
        "is_stable",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted(display_entries, key=lambda item: (item.source, item.entry_id)):
            ehull = ehulls.get(entry.entry_id)
            writer.writerow(
                {
                    "entry_id": entry.entry_id,
                    "source": entry.source,
                    "formula": entry.reduced_formula,
                    "chemical_system": entry.chemical_system,
                    "num_atoms": entry.num_atoms,
                    "input_energy": entry.input_energy,
                    "energy_per_atom": entry.energy_per_atom,
                    "ehull": ehull,
                    "is_stable": ehull is not None and abs(ehull) <= 1e-8,
                }
            )


def main() -> None:
    args = parse_args()
    if not args.use_ehull_csv_for_ternary and not args.ternary_csv:
        raise ValueError("--ternary-csv is required unless --use-ehull-csv-for-ternary is set.")

    replace_prefixes = tuple(tuple(item) for item in args.replace_cif_prefix) if args.replace_cif_prefix else ()
    ehull_rows = read_ehull_rows(args.ehull_csv)
    ehulls = {row.entry_id: row.ehull for row in ehull_rows}
    ternary_ids = select_ternary_ids(ehull_rows, args.chemical_system, args.min_ehull, args.show_unstable)

    all_entries: list[ParsedEntry] = []
    for csv_path in args.single_csv:
        all_entries.extend(load_entries(csv_path, "single", args.jobs, args.energy_mode, replace_prefixes))
    for csv_path in args.binary_csv:
        all_entries.extend(load_entries(csv_path, "binary", args.jobs, args.energy_mode, replace_prefixes))
    if args.use_ehull_csv_for_ternary:
        all_entries.extend(
            load_ternary_entries_from_ehull_csv(
                ehull_rows,
                args.chemical_system,
                args.min_ehull,
                args.show_unstable,
            )
        )
    else:
        for csv_path in args.ternary_csv or []:
            all_entries.extend(
                load_entries(
                    csv_path,
                    "ternary",
                    args.jobs,
                    args.energy_mode,
                    replace_prefixes,
                    ternary_ids,
                )
            )

    display_entries = build_display_entries(
        all_entries=all_entries,
        ehulls=ehulls,
        chemical_system=args.chemical_system,
        min_ehull=args.min_ehull,
        show_unstable=args.show_unstable,
    )
    display_phase_diagram = PhaseDiagram([build_plot_entry(entry) for entry in display_entries])
    if args.output_data_csv is not None:
        write_display_data(args.output_data_csv, display_entries, ehulls)

    plotter = PDPlotter(
        display_phase_diagram,
        show_unstable=args.show_unstable,
        backend="matplotlib",
        ternary_style="2d",
    )
    ax = plotter.get_plot(
        label_stable=args.label_stable,
        label_unstable=args.label_unstable,
        fill=True,
    )
    restyle_pdplotter_axes(
        ax=ax,
        plotter=plotter,
        chemical_system=args.chemical_system,
        keep_element_labels=args.keep_element_labels,
        hide_pd_mesh=args.hide_pd_mesh,
    )
    if args.color_by_ehull:
        overlay_ehull_colors(ax, plotter, ehulls, args.marker_size, args.ehull_cmap)
    fig = ax.figure
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
