"""Command-line entry point: run the G3Point pipeline on a point cloud + .ini config.

    g3point CLOUD.ply CONFIG.ini [--version matlab_dbscan] [--seed 42] [--save] [--json]

Runs denoise -> segment -> cluster -> clean -> per-grain ellipsoid fit, then prints the
grain count and D16/D50/D84 of the b-axis grain-size distribution. `--save` also writes the
labelled `_G3POINT.laz` / `_G3POINT_SINKS.laz` next to the input cloud.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

from .G3Point import G3Point
from .grains import percentiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g3point", description=__doc__.splitlines()[0])
    parser.add_argument("cloud", help="input point cloud (.ply or .laz)")
    parser.add_argument("ini", help="G3Point parameter file (.ini)")
    parser.add_argument("--version", default="matlab_dbscan",
                        choices=["matlab_dbscan", "matlab", "cpp", "custom"],
                        help="merge mode; 'matlab_dbscan' is the verified MATLAB-parity path")
    parser.add_argument("--seed", type=int, default=42,
                        help="base seed for the deterministic per-grain Acover RNG")
    parser.add_argument("--no-min-shift", action="store_true",
                        help="do not min-shift the cloud for numerical conditioning")
    parser.add_argument("--save", action="store_true",
                        help="write labelled _G3POINT.laz / _G3POINT_SINKS.laz next to the input")
    parser.add_argument("--json", action="store_true",
                        help="emit the result (percentiles, counts, provenance) as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # The constructor and pipeline print progress to stdout; when emitting JSON, route ALL of that
    # to stderr so stdout carries ONLY the JSON document (otherwise it is not parseable).
    with contextlib.redirect_stdout(sys.stderr if args.json else sys.stdout):
        g = G3Point(args.cloud, args.ini, remove_mins=not args.no_min_shift)
        result = g.run(version=args.version, run_seed=args.seed)
        gsd = g.grain_size_distribution()
        if args.save:
            g.save()

    n_grains = sum(1 for grain in result.grains if grain.fitok)
    n_kept = len(gsd)
    pct = percentiles(gsd)

    if args.json:
        # NaN (empty percentiles) is not valid JSON -> emit null instead.
        pct_json = {k: (v if v == v else None) for k, v in pct.items()}
        payload = {
            "n_grains_fit": n_grains,
            "n_grains_in_gsd": n_kept,
            "percentiles_m": pct_json,
            "provenance": dict(result.provenance),   # plain dict for the JSON encoder
        }
        json.dump(payload, sys.stdout, indent=2, allow_nan=False, default=float)
        sys.stdout.write("\n")
    else:
        print(f"grains fit: {n_grains}   in GSD (fitok & aqualityok): {n_kept}")
        if n_kept:
            print("b-axis grain size (metres): "
                  + "  ".join(f"{k}={v:.4f}" for k, v in pct.items()))
        else:
            print("no grains passed the fit-quality filter -- empty GSD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
