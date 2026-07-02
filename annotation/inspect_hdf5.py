"""Print the structure of an HDF5 file: groups, datasets (shape/dtype), and attrs.

Usage:
    python inspect_hdf5.py path/to/file.hdf5
    python inspect_hdf5.py path/to/file.hdf5 --stats   # also print min/max/mean for numeric datasets
    python inspect_hdf5.py path/to/file.hdf5 --group data/demo_00  # only walk one subtree
"""
import argparse

import h5py
import numpy as np


def format_attrs(obj):
    if not obj.attrs:
        return ""
    parts = [f"{k}={v}" for k, v in obj.attrs.items()]
    return "  [attrs: " + ", ".join(parts) + "]"


def format_dataset(ds, show_stats):
    info = f"{ds.shape} {ds.dtype}"
    if show_stats and np.issubdtype(ds.dtype, np.number) and ds.size > 0 and ds.size < 5_000_000:
        arr = ds[()]
        info += f"  min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g}"
    return info


def walk(group, show_stats, prefix=""):
    for name in group.keys():
        item = group[name]
        indent = prefix + "  "
        if isinstance(item, h5py.Group):
            print(f"{prefix}{name}/{format_attrs(item)}")
            walk(item, show_stats, indent)
        else:
            print(f"{prefix}{name}: {format_dataset(item, show_stats)}{format_attrs(item)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to .hdf5 file")
    parser.add_argument("--group", default=None, help="Only walk this subgroup (e.g. data/demo_00)")
    parser.add_argument("--stats", action="store_true", help="Print min/max/mean for numeric datasets")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"{args.path}{format_attrs(f)}")
        root = f[args.group] if args.group else f
        walk(root, args.stats)


if __name__ == "__main__":
    main()
