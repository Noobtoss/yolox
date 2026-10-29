import argparse
import ast
import re
import shutil
import sys
from pathlib import Path


IGNORED_FIELDS = {"seed", "exp_name", "output_dir"}


class IgnoreRunFields(ast.NodeTransformer):
    def visit_Assign(self, node):
        if all(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
               and target.value.id == "self" and target.attr in IGNORED_FIELDS
               for target in node.targets):
            return None
        return node

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self" and node.func.attr == "merge"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Dict)):
            config = node.args[0]
            items = [(key, value) for key, value in zip(config.keys, config.values)
                     if not isinstance(key, ast.Constant) or key.value not in IGNORED_FIELDS]
            config.keys = [key for key, value in items]
            config.values = [value for key, value in items]
        return self.generic_visit(node)


def read_exp(exp_path):
    tree = ast.parse(exp_path.read_text(encoding="utf-8"))
    return ast.dump(IgnoreRunFields().visit(tree), include_attributes=False)


def group_runs(runs_dir, target_dir=None):
    runs_dir = Path(runs_dir).resolve()
    target_dir = Path(target_dir).resolve() if target_dir is not None else runs_dir.parent / "results"
    if not runs_dir.is_dir():
        raise ValueError(f"Not a directory: {runs_dir}")

    groups = {}
    for run in sorted(runs_dir.iterdir()):
        match = re.fullmatch(r"(.+)_\d+_\d+", run.name)
        if run.is_dir() and match:
            groups.setdefault(match[1], []).append(run)
    if not groups:
        raise ValueError(f"No run directories ending in _<number>_<number> in {runs_dir}")

    # Validate every group before moving anything.
    plans = []
    for group, runs in groups.items():
        target = target_dir / group
        reference_path = target / "exp.py" if target.exists() or target.is_symlink() else None
        reference = read_exp(reference_path) if reference_path else None
        moves = {}
        for run in runs:
            exp_path = run / "exp.py"
            checkpoints = sorted(path for path in run.rglob("*.pth") if path.is_file())
            if not exp_path.is_file() or not checkpoints:
                print(f"Warning: skipping {run}: missing exp.py or .pth checkpoints")
                continue
            exp = read_exp(exp_path)
            if reference is None:
                reference, reference_path = exp, exp_path
            elif exp != reference:
                raise ValueError(f"Experiments differ: {reference_path} vs {exp_path}")

            for checkpoint in checkpoints:
                name = f"{run.name}.pth" if len(checkpoints) == 1 else f"{run.name}_{checkpoint.stem}.pth"
                existing = moves.get(name, target / name)
                if existing.exists() or existing.is_symlink():
                    print(f"Skipping {checkpoint}: destination name already in use: {name}")
                    continue
                moves[name] = checkpoint
        if moves:
            plans.append((target, {"exp.py": reference_path, **moves}))

    for target, moves in plans:
        print(f"Organizing into {target}")
        target.mkdir(parents=True, exist_ok=True)
        for name, source in moves.items():
            if source != target / name:
                print(f"  {source} -> {name}")
                shutil.move(str(source), target / name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="Directory containing the individual run folders")
    parser.add_argument("--target_dir", type=Path, help="Output directory; defaults to results beside runs_dir")
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = argparse.Namespace(
            runs_dir=Path(__file__).resolve().parents[1] / "runs",
            target_dir=None,
        )
    group_runs(args.runs_dir, target_dir=args.target_dir)
