"""
Train and compare KDE, normalizing flow, and diffusion models on MuonDecay data.

Run from project root:
    python -m data_processing_scripts.test_nf_df
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity

if __name__ == "__main__" and "data_processing_scripts" not in sys.modules:
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.abspath(os.path.join(here, os.pardir))
    if parent not in sys.path:
        sys.path.insert(0, parent)

from data_processing_scripts import df_model, nf_model


def _resolve_data_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.abspath(os.path.join(here, os.pardir)), path)
    return candidate


def load_data(path: str, columns: list[str]) -> np.ndarray:
    resolved = _resolve_data_path(path)
    df = pd.read_csv(resolved, usecols=columns)
    data = df.to_numpy().astype(np.float32)
    return data


def kde_model(train: np.ndarray, bandwidth: float = 0.5) -> tuple[KernelDensity, np.ndarray, np.ndarray, float]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    train_std = (train - mean) / std
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(train_std)
    return kde, mean, std, bandwidth


def kde_log_prob(kde: KernelDensity, mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_std = (x - mean) / std
    return kde.score_samples(x_std) - np.log(std).sum()


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    columns = ["m12^2", "m13^2"]
    data = load_data("data/MuonDecay.csv", columns)

    # Prepare evaluation grid over data bounds
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    grid_res = 40
    g1 = np.linspace(mins[0], maxs[0], grid_res)
    g2 = np.linspace(mins[1], maxs[1], grid_res)
    G1, G2 = np.meshgrid(g1, g2)
    grid_points = np.stack([G1.ravel(), G2.ravel()], axis=1).astype(np.float32)
    # For compact display, also keep a 10x10 grid
    disp_res = 10
    dg1 = np.linspace(mins[0], maxs[0], disp_res)
    dg2 = np.linspace(mins[1], maxs[1], disp_res)
    dG1, dG2 = np.meshgrid(dg1, dg2)
    disp_points = np.stack([dG1.ravel(), dG2.ravel()], axis=1).astype(np.float32)
    np.set_printoptions(precision=3, suppress=False, formatter={"float_kind": lambda x: f"{x:.3e}"})

    # KDE baseline with fixed bandwidth
    kde, mean, std, bw = kde_model(data, bandwidth=0.5)
    kde_disp = np.exp(kde_log_prob(kde, mean, std, disp_points)).reshape(disp_res, disp_res)
    print(f"KDE density on 10x10 grid (bandwidth={bw}):")
    print(kde_disp)

    # Normalizing flow
    nf = nf_model.model(dim=2, hidden=256, num_coupling_layers=8)
    nf.fit(data, epochs=20, batch_size=4096, lr=5e-4, verbose=True)
    nf_disp = nf.eval(disp_points).reshape(disp_res, disp_res)
    print("NF density on 10x10 grid:")
    print(nf_disp)

    # Diffusion model
    dfm = df_model.model(dim=2, hidden=256, timesteps=200)
    dfm.fit(data, epochs=20, batch_size=4096, lr=1e-3, num_generated=20000, verbose=True)
    df_disp = dfm.eval(disp_points).reshape(disp_res, disp_res)
    print("Diffusion density on 10x10 grid:")
    print(df_disp)


if __name__ == "__main__":
    main()
