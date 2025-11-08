from parsl import bash_app, python_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL


def cmd_compile_vasp_hull(total_calcs, output_file, prefix):
    '''
    collect convex hull for all the structures
    '''
    import os

    total_calcs = int(total_calcs)
    pairs = []

    for calc_idx in range(1, total_calcs + 1):
        calc_dir = f"{prefix}{calc_idx}"
        incar = os.path.join(calc_dir, "INCAR")
        out = os.path.join(calc_dir, "output")
        outcar = os.path.join(calc_dir, "OUTCAR")

        formula = ""
        energy = 0.0
        natoms = None

        try:
            last = None
            with open(incar, "r") as f:
                for ln in f:
                    if "SYSTEM" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 3:
                    formula = toks[2]
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(out, "r") as f:
                for ln in f:
                    if "F=" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 5:
                    energy = float(toks[4])
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(outcar, "r") as f:
                for ln in f:
                    if "NIONS" in ln:
                        last = ln
            if last:
                natoms = int(last.split()[-1])
        except FileNotFoundError:
            pass

        if natoms is not None:
            tenergy = float(f"{energy:.6f}")
            epa = float(f"{tenergy / natoms:.6f}")
            pairs.append((formula, epa))

    pairs.sort(key=lambda x: (x[0], x[1]))
    seen = set()
    best = {}
    for formula, epa in pairs:
        if (formula not in seen) or (epa < best.get(formula, float("inf"))):
            best[formula] = epa
            seen.add(formula)
    with open(output_file, "w") as f:
        for formula in sorted(best.keys()):
            f.write(f"{formula} {best[formula]:.6f}\n")


@python_app(executors=[POSTPROCESSING_LABEL])
def compile_vasp_hull(total_calcs, output_file, prefix):
    return cmd_compile_vasp_hull(total_calcs, output_file, prefix)
