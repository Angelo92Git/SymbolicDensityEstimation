#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dill
import numpy as np
import torch
import runpy

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from config_management.data_config_dijet import DataConfig  # noqa: E402


@dataclass(frozen=True)
class Square:
    x0: float
    y0: float
    size: float

    @property
    def x1(self) -> float:
        return self.x0 + self.size

    @property
    def y1(self) -> float:
        return self.y0 + self.size

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        return (self.x0 <= x) & (x <= self.x1) & (self.y0 <= y) & (y <= self.y1)


def _invalid_mask(points_xy: np.ndarray) -> np.ndarray:
    mask = np.zeros(points_xy.shape[0], dtype=bool)
    reflection_lines = DataConfig.reflection_lines
    for i in range(reflection_lines.shape[0]):
        m = reflection_lines[i, 0]
        b = reflection_lines[i, 1]
        gt_or_lt = reflection_lines[i, 2]
        mask = mask | (
            (points_xy[:, 1] < m * points_xy[:, 0] + b)
            if gt_or_lt == 1
            else (points_xy[:, 1] > m * points_xy[:, 0] + b)
        )
    return mask


def _make_square_grid(square: Square, n: int) -> tuple[np.ndarray, float, float]:
    if n < 2:
        raise ValueError("n must be >= 2")
    xs = np.linspace(square.x0, square.x1, n, dtype=float)
    ys = np.linspace(square.y0, square.y1, n, dtype=float)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xv.ravel(), yv.ravel()], axis=1)
    return pts, dx, dy


def _load_kde_model(models_dir: Path) -> object:
    with (models_dir / "dijet_kde_wrapped.pkl").open("rb") as f:
        return dill.load(f)


def _load_nf_model(models_dir: Path) -> torch.nn.Module:
    with (models_dir / "dijet_neural.pkl").open("rb") as f:
        model = dill.load(f)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Loaded dijet_neural.pkl is not a torch.nn.Module")
    model.eval()
    return model


def _load_samples(samples_path: Path) -> np.ndarray:
    def _try_load(skiprows: int) -> np.ndarray:
        return np.loadtxt(samples_path, delimiter=",", skiprows=skiprows)

    try:
        samples = _try_load(skiprows=0)
    except Exception:
        samples = _try_load(skiprows=1)

    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(f"Expected Nx(>=2) numeric samples in {samples_path}, got {samples.shape}")
    return samples[:, :2].astype(float, copy=False)


def _load_density_scale_factor() -> float:
    candidates = [
        _REPO_ROOT / "data/processed_data/dijet_neural_scale_factor.txt",
        _REPO_ROOT / "data/processed_data/dijet_scale_factor.txt",
    ]
    for p in candidates:
        if p.exists():
            return float(np.loadtxt(p))
    return 1.0


def _load_sr_best(
    sr_results_path: Path, sr_index: int
) -> tuple[callable, dict[str, object]]:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    g = runpy.run_path(str(sr_results_path))
    raw_equations = list(g["raw_equations"])
    loss = list(g["loss"])
    complexity = list(g["complexity"])

    if sr_index < 0:
        idx = len(raw_equations) + sr_index
    else:
        idx = sr_index
    if idx < 0 or idx >= len(raw_equations):
        raise IndexError(f"sr_index={sr_index} out of range for {len(raw_equations)} equations")

    eq_str = str(raw_equations[idx])

    x1, x2 = sp.symbols("x1 x2")
    s_processed = (
        eq_str.replace("pow2", "pow2_func")
        .replace("pow3", "pow3_func")
        .replace("pow4", "pow4_func")
        .replace("pow5", "pow5_func")
        .replace("pow_int", "pow_int_func")
    )
    locals_dict = {
        "x1": x1,
        "x2": x2,
        "pow2_func": sp.Function("pow2"),
        "pow3_func": sp.Function("pow3"),
        "pow4_func": sp.Function("pow4"),
        "pow5_func": sp.Function("pow5"),
        "pow_int_func": sp.Function("pow_int"),
        "exp": sp.exp,
        "log": sp.log,
    }
    expr = parse_expr(s_processed, local_dict=locals_dict, evaluate=False)

    def _replace_pow(expr_in: sp.Expr) -> sp.Expr:
        def _repl(f):
            if not isinstance(f, sp.Function):
                return f
            name = f.func.__name__
            if name == "pow_int":
                if len(f.args) != 2:
                    return f
                base, exp = f.args
                return sp.Pow(_replace_pow(base), _replace_pow(exp), evaluate=False)
            if not name.startswith("pow"):
                return f
            try:
                n = int(name[len("pow") :])
            except ValueError:
                return f
            if len(f.args) != 1:
                return f
            return sp.Pow(_replace_pow(f.args[0]), n, evaluate=False)

        return expr_in.replace(
            lambda x: isinstance(x, sp.Function)
            and (x.func.__name__.startswith("pow") or x.func.__name__ == "pow_int"),
            _repl,
            map=False,
        )

    expr = _replace_pow(expr)
    sr_func = sp.lambdify((x1, x2), expr, modules="numpy")

    meta = {
        "equation": eq_str,
        "loss": float(loss[idx]),
        "complexity": int(complexity[idx]),
        "index": int(sr_index),
    }
    return sr_func, meta


def _load_raw_and_scale(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(raw_path, usecols=DataConfig.columns)
    samples = df[DataConfig.columns].to_numpy(dtype=float, copy=False)
    samples_min = np.min(samples, axis=0)
    samples_max = np.max(samples, axis=0)
    if DataConfig.min_max_scaling:
        scaled = (samples - samples_min) / (samples_max - samples_min)
    else:
        scaled = samples
    return scaled, samples_min, samples_max


def _integrate_kde_on_square(kde_model: object, square: Square, n: int) -> float:
    pts, dx, dy = _make_square_grid(square, n)
    densities = np.asarray(kde_model.evaluate(pts), dtype=float)
    densities[_invalid_mask(pts)] = 0.0
    return float(np.sum(densities) * dx * dy)


def _integrate_nf_on_square(nf_model: torch.nn.Module, square: Square, n: int, batch_size: int) -> float:
    pts, dx, dy = _make_square_grid(square, n)
    mask = _invalid_mask(pts)

    param = next(nf_model.parameters())
    device = param.device
    dtype = param.dtype
    pts_t = torch.from_numpy(pts).to(device=device, dtype=dtype)

    densities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in torch.split(pts_t, batch_size):
            densities.append(torch.exp(nf_model.log_prob(batch)).detach().cpu().numpy())

    dens = np.concatenate(densities, axis=0).astype(float, copy=False)
    dens[mask] = 0.0
    return float(np.sum(dens) * dx * dy)


def _integrate_sr_on_square(sr_func, square: Square, n: int, density_scale_factor: float) -> float:
    pts, dx, dy = _make_square_grid(square, n)
    mask = _invalid_mask(pts)
    dens = np.asarray(sr_func(pts[:, 0], pts[:, 1]), dtype=float)
    if dens.shape != (pts.shape[0],):
        dens = np.asarray(dens).reshape(-1)
    dens = np.maximum(dens / density_scale_factor, 0.0)
    dens[mask] = 0.0
    return float(np.sum(dens) * dx * dy)


def _square_report(
    *,
    name: str,
    square: Square,
    samples_xy: np.ndarray,
    kde_model: object,
    nf_model: torch.nn.Module,
    sr_func,
    sr_meta: dict[str, object],
    density_scale_factor: float,
    n_integral: int,
    batch_size: int,
) -> dict[str, float]:
    inside = square.contains(samples_xy)
    n_inside = int(np.sum(inside))
    n_total = int(samples_xy.shape[0])
    empirical = float(np.mean(inside))

    kde_mass = _integrate_kde_on_square(kde_model, square, n=n_integral)
    nf_mass = _integrate_nf_on_square(nf_model, square, n=n_integral, batch_size=batch_size)
    sr_mass = _integrate_sr_on_square(sr_func, square, n=n_integral, density_scale_factor=density_scale_factor)

    n_integral_fine = int(n_integral) * 2
    kde_mass_fine = _integrate_kde_on_square(kde_model, square, n=n_integral_fine)
    nf_mass_fine = _integrate_nf_on_square(nf_model, square, n=n_integral_fine, batch_size=batch_size)
    sr_mass_fine = _integrate_sr_on_square(
        sr_func, square, n=n_integral_fine, density_scale_factor=density_scale_factor
    )

    return {
        "n_inside": float(n_inside),
        "n_total": float(n_total),
        "empirical_prob": empirical,
        "kde_prob": kde_mass,
        "nf_prob": nf_mass,
        "sr_prob": sr_mass,
        "n_integral": float(n_integral),
        "n_integral_fine": float(n_integral_fine),
        "kde_prob_fine": kde_mass_fine,
        "nf_prob_fine": nf_mass_fine,
        "sr_prob_fine": sr_mass_fine,
        "sr_loss": float(sr_meta["loss"]),
        "sr_complexity": float(sr_meta["complexity"]),
        "kde_abs_err": abs(kde_mass - empirical),
        "nf_abs_err": abs(nf_mass - empirical),
        "sr_abs_err": abs(sr_mass - empirical),
        "kde_rel_err": abs(kde_mass - empirical) / (empirical + 1e-15),
        "nf_rel_err": abs(nf_mass - empirical) / (empirical + 1e-15),
        "sr_rel_err": abs(sr_mass - empirical) / (empirical + 1e-15),
        "kde_abs_err_fine": abs(kde_mass_fine - empirical),
        "nf_abs_err_fine": abs(nf_mass_fine - empirical),
        "sr_abs_err_fine": abs(sr_mass_fine - empirical),
        "kde_rel_err_fine": abs(kde_mass_fine - empirical) / (empirical + 1e-15),
        "nf_rel_err_fine": abs(nf_mass_fine - empirical) / (empirical + 1e-15),
        "sr_rel_err_fine": abs(sr_mass_fine - empirical) / (empirical + 1e-15),
    }


def _format_row(label: str, results: dict[str, float]) -> str:
    n_inside = int(results["n_inside"])
    n_total = int(results["n_total"])
    emp = results["empirical_prob"]
    kde = results["kde_prob"]
    nf = results["nf_prob"]
    sr = results["sr_prob"]
    kde_f = results["kde_prob_fine"]
    nf_f = results["nf_prob_fine"]
    sr_f = results["sr_prob_fine"]
    n_int = int(results["n_integral"])
    n_int_f = int(results["n_integral_fine"])
    sr_loss = results["sr_loss"]
    sr_complexity = int(results["sr_complexity"])
    return "\n".join(
        [
            f"{label}",
            f"  count: {n_inside}/{n_total}  (empirical={emp:.6e})",
            f"  kde:   {kde:.6e}  (n={n_int})    abs_err={results['kde_abs_err']:.3e}   rel_err={results['kde_rel_err']:.3e}",
            f"         {kde_f:.6e}  (n={n_int_f})  abs_err={results['kde_abs_err_fine']:.3e}   rel_err={results['kde_rel_err_fine']:.3e}",
            f"  nf:    {nf:.6e}  (n={n_int})    abs_err={results['nf_abs_err']:.3e}   rel_err={results['nf_rel_err']:.3e}",
            f"         {nf_f:.6e}  (n={n_int_f})  abs_err={results['nf_abs_err_fine']:.3e}   rel_err={results['nf_rel_err_fine']:.3e}",
            f"  sr:    {sr:.6e}  (n={n_int})    abs_err={results['sr_abs_err']:.3e}   rel_err={results['sr_rel_err']:.3e}",
            f"         {sr_f:.6e}  (n={n_int_f})  abs_err={results['sr_abs_err_fine']:.3e}   rel_err={results['sr_rel_err_fine']:.3e}",
            f"         (sr complexity={sr_complexity}, loss={sr_loss:.6g})",
        ]
    )


def _iter_squares(square_size: float) -> Iterable[tuple[str, Square]]:
    yield "square_(0.01,0.01)", Square(x0=0.01, y0=0.01, size=square_size)
    yield "square_(0.03,0.03)", Square(x0=0.03, y0=0.03, size=square_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate empirical mass vs numerical integral on two small squares, "
            "using the same dijet KDE wrapper + normalizing flow model as the notebook."
        )
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help=(
            "CSV of samples in pipeline space (expects >=2 columns; uses first two). "
            "If not set, uses --samples-raw."
        ),
    )
    parser.add_argument(
        "--samples-raw",
        type=Path,
        default=Path("data/Dijets.csv"),
        help=(
            "Optional raw dijet CSV (e.g. data/Dijets.csv). When set, reads DataConfig.columns "
            "and applies min-max scaling using the raw file's min/max."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing dijet_kde_wrapped.pkl and dijet_neural.pkl (default: models).",
    )
    parser.add_argument(
        "--sr-results",
        type=Path,
        default=Path("data/pareto_results/dijet_neural_results.py"),
        help="SR pareto results python file (default: data/pareto_results/dijet_neural_results.py).",
    )
    parser.add_argument(
        "--sr-index",
        type=int,
        default=-1,
        help="Index into SR equation list (default: -1 for the last / best-loss equation in the notebook).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.02,
        help="Square side length (default: 0.02).",
    )
    parser.add_argument(
        "--n-integral",
        type=int,
        default=250,
        help="Grid resolution per axis for numeric integration (default: 250).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for NF evaluation (default: 8192).",
    )
    args = parser.parse_args()

    def _resolve_repo_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else (_REPO_ROOT / p)

    samples_path = _resolve_repo_path(args.samples)
    samples_raw_path = _resolve_repo_path(args.samples_raw)
    models_dir = _resolve_repo_path(args.models_dir) or _REPO_ROOT / "models"
    sr_results_path = _resolve_repo_path(args.sr_results) or (_REPO_ROOT / "data/pareto_results/dijet_neural_results.py")

    if samples_raw_path is not None and not samples_raw_path.exists():
        raise FileNotFoundError(f"Raw samples file not found: {samples_raw_path}")
    if samples_raw_path is None and (samples_path is None or not samples_path.exists()):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")
    if not sr_results_path.exists():
        raise FileNotFoundError(f"SR results file not found: {sr_results_path}")

    kde_model = _load_kde_model(models_dir)
    nf_model = _load_nf_model(models_dir)
    density_scale_factor = _load_density_scale_factor()
    sr_func, sr_meta = _load_sr_best(sr_results_path, sr_index=args.sr_index)
    if samples_raw_path is not None:
        samples_xy, smin, smax = _load_raw_and_scale(samples_raw_path)
        samples_path_label = str(samples_raw_path)
        print(f"raw scaling mins: {smin} maxs: {smax}")
    else:
        if samples_path is None:
            raise ValueError("--samples is required when --samples-raw is not provided")
        samples_xy = _load_samples(samples_path)
        samples_path_label = str(samples_path)

    print(f"samples: {samples_path_label}  (N={samples_xy.shape[0]})")
    print(f"models: {models_dir}")
    print(f"sr_results: {sr_results_path}  (sr_index={args.sr_index}, complexity={int(sr_meta['complexity'])}, loss={float(sr_meta['loss']):.6g})")
    print(f"density_scale_factor: {density_scale_factor}")
    print(f"square_size: {args.square_size}  n_integral: {args.n_integral}")
    print(f"samples_min: {samples_xy.min(axis=0)}  samples_max: {samples_xy.max(axis=0)}")
    print("")

    for idx, (name, square) in enumerate(_iter_squares(args.square_size)):
        results = _square_report(
            name=name,
            square=square,
            samples_xy=samples_xy,
            kde_model=kde_model,
            nf_model=nf_model,
            sr_func=sr_func,
            sr_meta=sr_meta,
            density_scale_factor=density_scale_factor,
            n_integral=args.n_integral,
            batch_size=args.batch_size,
        )
        if idx > 0:
            print("")
        print(_format_row(f"{name}  (x∈[{square.x0:.3f},{square.x1:.3f}], y∈[{square.y0:.3f},{square.y1:.3f}])", results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
