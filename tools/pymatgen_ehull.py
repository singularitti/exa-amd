#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shlex
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser


@dataclass(frozen=True)
class ParsedEntry:
    entry_id: str
    source: str
    formula: str
    reduced_formula: str
    chemical_system: str
    elements: tuple[str, ...]
    num_atoms: float
    input_energy: float
    energy_per_atom: float


def _resolve_cif_path(cif_path: str, replace_prefixes: tuple[tuple[str, str], ...]) -> str:
    resolved = cif_path
    for old_prefix, new_prefix in replace_prefixes:
        if not cif_path.startswith(old_prefix):
            continue
        mapped = new_prefix + cif_path[len(old_prefix):]
        if Path(mapped).exists():
            return mapped
        flat_mapped = str(Path(new_prefix) / Path(cif_path).name)
        if Path(flat_mapped).exists():
            return flat_mapped
        resolved = mapped
        break
    return resolved


def _clean_formula_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        parts = shlex.split(stripped)
    except ValueError:
        parts = []
    if parts:
        return " ".join(parts)
    return stripped.strip("'\"")


def _formula_from_cif_header(cif_file: Path) -> tuple[str | None, float | None]:
    structural_formula: str | None = None
    sum_formula: str | None = None
    header_formula: str | None = None

    with cif_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("data_") and header_formula is None:
                header_formula = stripped[5:].strip() or None
                continue
            if stripped.startswith("_chemical_formula_sum"):
                sum_formula = _clean_formula_text(stripped.partition("_chemical_formula_sum")[2])
                break
            if stripped.startswith("_chemical_formula_structural") and structural_formula is None:
                structural_formula = _clean_formula_text(stripped.partition("_chemical_formula_structural")[2])
            if stripped.startswith("loop_"):
                break

    for formula in (sum_formula, structural_formula, header_formula):
        if not formula:
            continue
        composition = Composition(formula)
        num_atoms = float(composition.num_atoms)
        if num_atoms > 0:
            return composition.formula, num_atoms
    return None, None


def _formula_from_cif(cif_path: str, replace_prefixes: tuple[tuple[str, str], ...]) -> tuple[str, float]:
    resolved_path = Path(_resolve_cif_path(cif_path, replace_prefixes))
    formula, num_atoms = _formula_from_cif_header(resolved_path)
    if formula is not None and num_atoms is not None:
        return formula, num_atoms

    parser = CifParser(resolved_path)
    structures = parser.parse_structures(primitive=False, on_error="raise")
    if not structures:
        raise ValueError(f"Could not parse a structure from {cif_path}")
    composition = structures[0].composition
    fallback_num_atoms = float(composition.num_atoms)
    if fallback_num_atoms <= 0:
        raise ValueError(f"Invalid atom count for {cif_path}: {fallback_num_atoms}")
    return composition.formula, fallback_num_atoms


def _parse_row(args: tuple[str, str, str, str, tuple[tuple[str, str], ...]]) -> ParsedEntry:
    source, entry_id, energy_text, cif_path, replace_prefixes = args
    formula, num_atoms = _formula_from_cif(cif_path, replace_prefixes)
    composition = Composition(formula)
    reduced_formula = composition.reduced_formula
    elements = tuple(sorted(str(el) for el in composition.elements))
    chemical_system = "-".join(elements)
    input_energy = float(energy_text)
    return ParsedEntry(
        entry_id=entry_id,
        source=source,
        formula=formula,
        reduced_formula=reduced_formula,
        chemical_system=chemical_system,
        elements=elements,
        num_atoms=num_atoms,
        input_energy=input_energy,
        energy_per_atom=math.nan,
    )


def _iter_csv_rows(
    csv_path: Path,
    source: str,
    replace_prefixes: tuple[tuple[str, str], ...],
    allowed_entry_ids: set[str] | None = None,
) -> Iterable[tuple[str, str, str, str, tuple[tuple[str, str], ...]]]:
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row_num, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"{csv_path}:{row_num} has fewer than 3 columns")
            entry_id = row[0].strip()
            if allowed_entry_ids is not None and entry_id not in allowed_entry_ids:
                continue
            yield source, entry_id, row[1].strip(), row[2].strip(), replace_prefixes


def load_entries(
    csv_path: Path,
    source: str,
    jobs: int,
    energy_mode: str,
    replace_prefixes: tuple[tuple[str, str], ...],
    allowed_entry_ids: set[str] | None = None,
) -> list[ParsedEntry]:
    entries: list[ParsedEntry] = []

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for parsed in executor.map(
            _parse_row,
            _iter_csv_rows(csv_path, source, replace_prefixes, allowed_entry_ids),
            chunksize=256,
        ):
            energy_per_atom = (
                parsed.input_energy / parsed.num_atoms
                if energy_mode == "total"
                else parsed.input_energy
            )
            entries.append(
                ParsedEntry(
                    entry_id=parsed.entry_id,
                    source=parsed.source,
                    formula=parsed.formula,
                    reduced_formula=parsed.reduced_formula,
                    chemical_system=parsed.chemical_system,
                    elements=parsed.elements,
                    num_atoms=parsed.num_atoms,
                    input_energy=parsed.input_energy,
                    energy_per_atom=energy_per_atom,
                )
            )
    return entries


def _build_pd_entry(parsed: ParsedEntry) -> PDEntry:
    composition = Composition(parsed.formula)
    return PDEntry(composition, parsed.energy_per_atom * parsed.num_atoms, name=parsed.entry_id)


def _compute_system_ehull(args: tuple[str, list[ParsedEntry]]) -> list[dict[str, str | float]]:
    chemical_system, entries = args
    pd_entries = [_build_pd_entry(entry) for entry in entries]
    phase_diagram = PhaseDiagram(pd_entries)
    results: list[dict[str, str | float]] = []

    for parsed, pd_entry in zip(entries, pd_entries, strict=True):
        decomp, ehull = phase_diagram.get_decomp_and_e_above_hull(
            pd_entry,
            allow_negative=True,
            on_error="raise",
        )
        decomposition = ";".join(
            f"{entry.composition.reduced_formula}:{amount:.8f}"
            for entry, amount in sorted(
                decomp.items(),
                key=lambda item: item[0].composition.reduced_formula,
            )
        )
        results.append(
            {
                "entry_id": parsed.entry_id,
                "source": parsed.source,
                "chemical_system": chemical_system,
                "formula": parsed.reduced_formula,
                "num_atoms": parsed.num_atoms,
                "input_energy": parsed.input_energy,
                "energy_per_atom": parsed.energy_per_atom,
                "ehull": ehull,
                "is_stable": abs(ehull) <= 1e-8,
                "decomposition": decomposition,
            }
        )
    return results


def compute_ehulls(entries: list[ParsedEntry], jobs: int, target_sources: set[str]) -> list[dict[str, str | float]]:
    target_systems = sorted(
        {
            entry.chemical_system
            for entry in entries
            if entry.source in target_sources
        }
    )
    entries_by_target: list[tuple[str, list[ParsedEntry]]] = []
    for target_system in target_systems:
        target_elements = set(target_system.split("-"))
        compatible_entries = [
            entry
            for entry in entries
            if set(entry.elements).issubset(target_elements)
        ]
        entries_by_target.append((target_system, compatible_entries))

    work_items = entries_by_target
    results: list[dict[str, str | float]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for chunk in executor.map(_compute_system_ehull, work_items, chunksize=1):
            results.extend(chunk)
    return results


def write_results(rows: list[dict[str, str | float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "entry_id",
        "source",
        "chemical_system",
        "formula",
        "num_atoms",
        "input_energy",
        "energy_per_atom",
        "ehull",
        "is_stable",
        "decomposition",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    str(row["chemical_system"]),
                    float(row["ehull"]),
                    str(row["source"]),
                    str(row["entry_id"]),
                ),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute energy above hull with pymatgen for unary, binary, and ternary datasets."
    )
    parser.add_argument("--single-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--binary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--ternary-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=10,
        help="Worker processes for CIF parsing and per-system hull evaluation.",
    )
    parser.add_argument(
        "--energy-mode",
        choices=("per-atom", "total"),
        default="per-atom",
        help="Interpret the input CSV energy column as formation energy per atom or total formation energy.",
    )
    parser.add_argument(
        "--output-sources",
        default="ternary",
        choices=("all", "ternary"),
        help="Write all entries or only ternary-candidate rows to the output CSV.",
    )
    parser.add_argument(
        "--replace-cif-prefix",
        nargs=2,
        action="append",
        metavar=("OLD_PREFIX", "NEW_PREFIX"),
        help="Rewrite CIF paths on the fly if the CSV stores an outdated root. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_entries: list[ParsedEntry] = []
    replace_prefixes = tuple(tuple(item) for item in args.replace_cif_prefix) if args.replace_cif_prefix else ()
    for csv_path in args.single_csv:
        all_entries.extend(load_entries(csv_path, "single", args.jobs, args.energy_mode, replace_prefixes))
    for csv_path in args.binary_csv:
        all_entries.extend(load_entries(csv_path, "binary", args.jobs, args.energy_mode, replace_prefixes))
    for csv_path in args.ternary_csv:
        all_entries.extend(load_entries(csv_path, "ternary", args.jobs, args.energy_mode, replace_prefixes))

    target_sources = {"ternary"} if args.output_sources == "ternary" else {"single", "binary", "ternary"}
    results = compute_ehulls(all_entries, args.jobs, target_sources)
    if args.output_sources == "ternary":
        results = [row for row in results if row["source"] == "ternary"]
    write_results(results, args.output_csv)


if __name__ == "__main__":
    main()
