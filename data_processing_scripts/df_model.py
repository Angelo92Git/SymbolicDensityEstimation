"""Simple diffusion-based density estimator with a KDE-like API."""

from typing import Iterable, List, Optional

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_array(x: np.ndarray | Iterable, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"Expected data with dimension {dim}, got {arr.shape[1]}")
    return arr


class DiffusionMLP(nn.Module):
    """Small MLP that conditions on (scaled) time."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


class DiffusionModel(nn.Module):
    """Minimal DDPM for vector-valued data."""

    def __init__(self, dim: int, hidden: int = 128, timesteps: int = 200):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.net = DiffusionMLP(dim, hidden)

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bars[t].unsqueeze(1)
        noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        t_scaled = t.float() / (self.timesteps - 1)
        noise_pred = self.net(noisy, t_scaled)
        return (noise_pred - noise).pow(2).mean()

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device if device is not None else next(self.parameters()).device
        x = torch.randn(num_samples, self.dim, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), float(i) / (self.timesteps - 1), device=device)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alpha_bars[i]
            eps = self.net(x, t)
            coef1 = 1.0 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1 - alpha_bar)
            if i > 0:
                noise = torch.randn_like(x)
                x = coef1 * (x - coef2 * eps) + torch.sqrt(beta) * noise
            else:
                x = coef1 * (x - coef2 * eps)
        return x


class model:
    """Wrapper exposing fit/log_prob/eval/sample, using KDE on generated samples."""

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        timesteps: int = 200,
        standardize: bool = True,
        device: Optional[str] = None,
    ):
        self.dim = dim
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DiffusionModel(dim, hidden=hidden, timesteps=timesteps).to(self.device)
        self.standardize = standardize
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._kde: Optional[KernelDensity] = None

    def _prep(self, samples: np.ndarray | Iterable) -> np.ndarray:
        return _as_array(samples, self.dim)

    def _normalize(self, arr: torch.Tensor) -> torch.Tensor:
        if not self.standardize or self._mean is None or self._std is None:
            return arr
        mean = torch.from_numpy(self._mean).to(arr.device)
        std = torch.from_numpy(self._std).to(arr.device)
        return (arr - mean) / std

    def _fit_kde(self, generated: np.ndarray) -> None:
        # Scott's rule of thumb for bandwidth; clamp to a small positive minimum.
        bandwidth = 1.06 * np.std(generated) * np.power(generated.shape[0], -1 / (self.dim + 4))
        bandwidth = float(max(bandwidth, 1e-3))
        self._kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self._kde.fit(generated)

    def fit(
        self,
        samples: np.ndarray | Iterable,
        epochs: int = 25,
        batch_size: int = 512,
        lr: float = 2e-3,
        num_generated: int = 4000,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> list[float]:
        """
        Train the diffusion model on provided samples.

        Returns
        -------
        List of per-epoch noise-prediction losses.
        """
        data = self._prep(samples)
        if self.standardize:
            self._mean = data.mean(axis=0, keepdims=True)
            self._std = data.std(axis=0, keepdims=True) + 1e-6
            data = (data - self._mean) / self._std

        # Shuffle once before building the loader for reproducibility.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(data))
        data = data[perm]

        tensor_data = torch.from_numpy(data).to(self.device)
        loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.diffusion.parameters(), lr=lr)

        history: List[float] = []
        self.diffusion.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = self.diffusion.loss(batch)
                loss.backward()
                opt.step()
                running_loss += loss.item() * len(batch)
            epoch_loss = running_loss / len(tensor_data)
            history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"[Diffusion] epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")

        self.diffusion.eval()
        with torch.no_grad():
            gen = self.diffusion.sample(num_generated, device=self.device).cpu().numpy()
        if self.standardize and self._mean is not None and self._std is not None:
            gen = gen * self._std + self._mean
        self._fit_kde(gen)
        return history

    def log_prob(self, x: np.ndarray | Iterable) -> np.ndarray:
        if self._kde is None:
            raise RuntimeError("Model must be fitted before evaluating densities.")
        arr = self._prep(x)
        logp = self._kde.score_samples(arr)
        return logp

    def eval(self, x: np.ndarray | Iterable) -> np.ndarray:
        """Density values at query points."""
        return np.exp(self.log_prob(x))

    def sample(self, num_samples: int) -> np.ndarray:
        self.diffusion.eval()
        with torch.no_grad():
            samples = self.diffusion.sample(num_samples, device=self.device).cpu().numpy()
        if self.standardize and self._mean is not None and self._std is not None:
            samples = samples * self._std + self._mean
        return samples


__all__ = ["model"]
