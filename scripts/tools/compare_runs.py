# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Compare agent.yaml and env.yaml between two training runs.

train.py dumps these files to <log_dir>/params/ at the start of each run.

Usage:
    python scripts/tools/compare_runs.py <path_run1> <path_run2>

Each path should be a run directory (e.g. logs/rsl_rl/deeprobotics_lite3_rough/2025-01-01_12-00-00)
containing params/agent.yaml and params/env.yaml, or the yaml files directly.
"""

import argparse
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Custom YAML loader that converts Isaac Lab Python-tagged values to plain
# Python objects so they can be diffed without importing Isaac Lab.
# ---------------------------------------------------------------------------

class _IslabLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


def _construct_python_object_apply(loader, tag_suffix, node):
    args = loader.construct_sequence(node)
    return f"{tag_suffix}({', '.join(str(a) for a in args)})"


_IslabLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", _construct_python_tuple
)
_IslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _construct_python_object_apply
)
_IslabLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:", _construct_python_object_apply
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_IslabLoader) or {}


# ---------------------------------------------------------------------------
# Flatten nested dict/list to dot-notation keys
# ---------------------------------------------------------------------------

def _flatten(d, prefix="") -> dict:
    items = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.update(_flatten(v, key))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            items.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = d
    return items


# ---------------------------------------------------------------------------
# Pretty diff
# ---------------------------------------------------------------------------

def _print_diff(flat1: dict, flat2: dict, name1: str, name2: str) -> None:
    keys1, keys2 = set(flat1), set(flat2)
    only1   = sorted(keys1 - keys2)
    only2   = sorted(keys2 - keys1)
    changed = [(k, flat1[k], flat2[k]) for k in sorted(keys1 & keys2) if flat1[k] != flat2[k]]

    if not (only1 or only2 or changed):
        print("  [identical]")
        return

    col_w = max(len(name1), len(name2), 6)

    if only1:
        print(f"\n  Keys only in {name1}:")
        for k in only1:
            print(f"    {k} = {flat1[k]!r}")

    if only2:
        print(f"\n  Keys only in {name2}:")
        for k in only2:
            print(f"    {k} = {flat2[k]!r}")

    if changed:
        key_w = max(len(k) for k, *_ in changed)
        print(f"\n  Changed values:")
        print(f"    {'key':<{key_w}}  {name1:<{col_w}}  {name2:<{col_w}}")
        print(f"    {'-'*key_w}  {'-'*col_w}  {'-'*col_w}")
        for k, v1, v2 in changed:
            print(f"    {k:<{key_w}}  {str(v1):<{col_w}}  {str(v2):<{col_w}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _find_yaml(run_dir: Path, filename: str) -> Path:
    """Search for filename directly in run_dir or in run_dir/params/."""
    for candidate in [run_dir / filename, run_dir / "params" / filename]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in {run_dir} or {run_dir / 'params'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two training run configs.")
    parser.add_argument("run1", help="Path to first run directory")
    parser.add_argument("run2", help="Path to second run directory")
    args = parser.parse_args()

    run1, run2 = Path(args.run1), Path(args.run2)
    name1, name2 = run1.name, run2.name

    for yaml_name in ("agent.yaml", "env.yaml"):
        print(f"\n{'='*70}")
        print(f" {yaml_name}")
        print(f"{'='*70}")

        try:
            path1 = _find_yaml(run1, yaml_name)
            path2 = _find_yaml(run2, yaml_name)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        flat1 = _flatten(_load_yaml(path1))
        flat2 = _flatten(_load_yaml(path2))
        _print_diff(flat1, flat2, name1, name2)

    print()


if __name__ == "__main__":
    main()
