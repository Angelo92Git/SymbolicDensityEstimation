"""
Lightweight normalizing-flow density estimator for tabular data.

Usage
-----
>>> from data_processing_scripts import nf_model
>>> flow = nf_model.model(dim=2)
>>> flow.fit(samples)
>>> density = flow.eval(query_points)
"""

from typing import Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_array(x: np.ndarray | Iterable, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"Expected data with dimension {dim}, got {arr.shape[1]}")
    return arr


class AffineCoupling(nn.Module):
    """Single RealNVP-style affine coupling layer."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim * 2),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * mask
        s, t = self.net(x_masked).chunk(2, dim=1)
        s = torch.tanh(s)  # keep scale stable
        y = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        log_det = ((1 - mask) * s).sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * mask
        s, t = self.net(y_masked).chunk(2, dim=1)
        s = torch.tanh(s)
        x = y_masked + (1 - mask) * ((y - t) * torch.exp(-s))
        log_det = -((1 - mask) * s).sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    """Minimal RealNVP normalizing flow with alternating binary masks."""

    def __init__(self, dim: int, hidden: int = 64, num_layers: int = 6):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden) for _ in range(num_layers)])
        masks: List[torch.Tensor] = []
        for i in range(num_layers):
            mask_vals = [(i + j) % 2 for j in range(dim)]
            masks.append(torch.tensor(mask_vals, dtype=torch.float32))
        self.register_buffer("masks", torch.stack(masks))
        self.base_dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for mask, layer in zip(self.masks, self.layers):
            mask = mask.to(x.device)
            z, ld = layer(z, mask)
            log_det += ld
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.size(0), device=z.device)
        x = z
        for mask, layer in reversed(list(zip(self.masks, self.layers))):
            mask = mask.to(z.device)
            x, ld = layer.inverse(x, mask)
            log_det += ld
        return x, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = self.base_dist.log_prob(z).sum(dim=1)
        return log_pz + log_det

    def sample(self, num_samples: int) -> torch.Tensor:
        z = self.base_dist.sample((num_samples, self.dim))
        x, _ = self.inverse(z)
        return x


class model:
    """
    Wrapper exposing a KDE-like interface.

    Parameters
    ----------
    dim:
        Dimensionality of the data.
    hidden:
        Hidden width for the coupling networks.
    num_coupling_layers:
        Number of affine coupling layers.
    standardize:
        Whether to internally standardize the data before training/evaluation.
    device:
        Torch device override.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 128,
        num_coupling_layers: int = 6,
        standardize: bool = True,
        device: Optional[str] = None,
    ):
        self.dim = dim
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flow = RealNVP(dim, hidden=hidden, num_layers=num_coupling_layers).to(self.device)
        self.standardize = standardize
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def _prep(self, samples: np.ndarray | Iterable) -> np.ndarray:
        return _as_array(samples, self.dim)

    def _normalize(self, arr: torch.Tensor) -> torch.Tensor:
        if not self.standardize or self._mean is None or self._std is None:
            return arr
        mean = torch.from_numpy(self._mean).to(arr.device)
        std = torch.from_numpy(self._std).to(arr.device)
        return (arr - mean) / std

    def fit(
        self,
        samples: np.ndarray | Iterable,
        epochs: int = 25,
        batch_size: int = 512,
        lr: float = 1e-3,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> list[float]:
        """
        Train the normalizing flow on provided samples.

        Returns
        -------
        List of per-epoch negative log-likelihood values.
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
        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)

        history: List[float] = []
        self.flow.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                log_prob = self.flow.log_prob(batch)
                loss = -log_prob.mean()
                loss.backward()
                opt.step()
                running_loss += loss.item() * len(batch)
            epoch_loss = running_loss / len(tensor_data)
            history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"[NF] epoch {epoch + 1}/{epochs} nll={epoch_loss:.4f}")
        return history

    def log_prob(self, x: np.ndarray | Iterable) -> np.ndarray:
        arr = self._prep(x)
        xt = torch.from_numpy(arr).to(self.device)
        xt = self._normalize(xt)
        self.flow.eval()
        with torch.no_grad():
            logp = self.flow.log_prob(xt)
            if self.standardize and self._std is not None:
                adjust = -torch.log(torch.from_numpy(self._std).to(self.device)).sum()
                logp = logp + adjust
        return logp.cpu().numpy()

    def eval(self, x: np.ndarray | Iterable) -> np.ndarray:
        """Density values at query points."""
        dens = np.exp(self.log_prob(x))
        return np.maximum(dens, 1e-12)

    def sample(self, num_samples: int) -> np.ndarray:
        self.flow.eval()
        with torch.no_grad():
            samples = self.flow.sample(num_samples).cpu().numpy()
        if self.standardize and self._mean is not None and self._std is not None:
            samples = samples * self._std + self._mean
        return samples


__all__ = ["model"]
