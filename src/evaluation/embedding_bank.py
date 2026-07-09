"""Embedding bank for storing and managing evaluation embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBankConfig:
    """Configuration for the EmbeddingBank."""
    embedding_dim: int
    max_size: Optional[int] = None
    device: Union[str, torch.device] = "cpu"
    normalize: bool = False
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


class EmbeddingBank:
    """Manages a bank of embeddings for evaluation and inference.
    
    Supports incremental updates, persistence, configurable memory limits,
    nearest-neighbor search, and descriptive statistics caching.
    """

    def __init__(
        self,
        config: Optional[EmbeddingBankConfig] = None,
        *,
        embedding_dim: Optional[int] = None,
        max_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        normalize: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            if embedding_dim is None:
                raise ValueError("Must provide either 'config' or 'embedding_dim'.")
            self.config = EmbeddingBankConfig(
                embedding_dim=embedding_dim,
                max_size=max_size,
                device=device,
                normalize=normalize,
                dtype=dtype,
            )

        self.embeddings: Optional[torch.Tensor] = None
        self.metadata: Dict[str, list[Any]] = {}
        self._stats_cache: Dict[str, torch.Tensor] = {}

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def max_size(self) -> Optional[int]:
        return self.config.max_size

    @property
    def device(self) -> torch.device:
        return self.config.device

    @property
    def normalize(self) -> bool:
        return self.config.normalize

    @property
    def dtype(self) -> torch.dtype:
        return self.config.dtype

    def __len__(self) -> int:
        """Returns the number of embeddings in the bank."""
        if self.embeddings is None:
            return 0
        return self.embeddings.shape[0]

    def __getitem__(self, idx: Union[int, slice, List[int], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Allows tensor-like indexing, returning embeddings and matching metadata."""
        if self.embeddings is None:
            raise IndexError("Cannot index an empty EmbeddingBank.")
            
        indexed_embeddings = self.embeddings[idx]
        indexed_metadata = {}
        
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                indices = idx.nonzero(as_tuple=False).squeeze(1).tolist()
            else:
                indices = idx.tolist()
        elif isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = list(range(len(self)))[idx]
        elif isinstance(idx, list):
            indices = idx
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

        for k, v in self.metadata.items():
            if isinstance(idx, int):
                indexed_metadata[k] = v[idx]
            else:
                indexed_metadata[k] = [v[i] for i in indices]
                
        return indexed_embeddings, indexed_metadata

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Yields individual (embedding, metadata_dict) pairs."""
        for i in range(len(self)):
            yield self[i]

    def _invalidate_cache(self) -> None:
        """Clears the cached statistics."""
        self._stats_cache.clear()
        logger.debug("EmbeddingBank statistics cache invalidated.")

    def _validate_embeddings(self, embeddings: torch.Tensor) -> None:
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Embeddings must be a torch.Tensor, got {type(embeddings)}")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        if embeddings.dtype != self.dtype:
            raise TypeError(f"Expected dtype {self.dtype}, got {embeddings.dtype}")
        
        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
            raise ValueError("Embeddings contain NaN or Inf values")

    def _validate_metadata(self, num_embeddings: int, metadata: Optional[Dict[str, list[Any]]]) -> None:
        if metadata is None:
            return
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")
        for k, v in metadata.items():
            if not isinstance(v, list):
                raise TypeError(f"Metadata value for key '{k}' must be a list.")
            if len(v) != num_embeddings:
                raise ValueError(
                    f"Metadata list '{k}' length ({len(v)}) does not match "
                    f"number of embeddings ({num_embeddings})."
                )

    def _process_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(self.device, dtype=self.dtype)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def build(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[Dict[str, list[Any]]] = None,
    ) -> None:
        """Initializes the embedding bank with a set of embeddings."""
        self._validate_embeddings(embeddings)
        self._validate_metadata(embeddings.shape[0], metadata)
        
        processed = self._process_embeddings(embeddings)
        
        if self.max_size is not None and processed.shape[0] > self.max_size:
            processed = processed[-self.max_size:]
            if metadata is not None:
                metadata = {k: v[-self.max_size:] for k, v in metadata.items()}

        self.embeddings = processed
        self.metadata = metadata.copy() if metadata else {}
        self._invalidate_cache()
        logger.info("EmbeddingBank built with %d embeddings.", self.embeddings.shape[0])

    def add(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[Dict[str, list[Any]]] = None,
    ) -> None:
        """Adds new embeddings to the bank."""
        self._validate_embeddings(embeddings)
        self._validate_metadata(embeddings.shape[0], metadata)
        processed = self._process_embeddings(embeddings)
        
        if self.embeddings is None:
            self.build(processed, metadata)
            return

        new_embeddings = torch.cat([self.embeddings, processed], dim=0)
        new_metadata = self.metadata.copy()
        
        if metadata is not None:
            for k, v in metadata.items():
                if k in new_metadata:
                    new_metadata[k] = new_metadata[k] + v
                else:
                    new_metadata[k] = [None] * self.embeddings.shape[0] + v

        if self.max_size is not None and new_embeddings.shape[0] > self.max_size:
            new_embeddings = new_embeddings[-self.max_size:]
            for k in new_metadata:
                new_metadata[k] = new_metadata[k][-self.max_size:]

        self.embeddings = new_embeddings
        self.metadata = new_metadata
        self._invalidate_cache()
        logger.info("Added %d embeddings. Total size: %d.", processed.shape[0], self.embeddings.shape[0])

    def remove(self, indices: Union[int, List[int], torch.Tensor]) -> None:
        """Removes embeddings by index."""
        if self.embeddings is None:
            return

        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        if not indices:
            return

        keep_mask = torch.ones(self.embeddings.shape[0], dtype=torch.bool, device=self.device)
        keep_mask[indices] = False
        
        self.embeddings = self.embeddings[keep_mask]
        
        if self.embeddings.shape[0] == 0:
            self.clear()
            return
            
        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        
        for k, v in self.metadata.items():
            if isinstance(v, list):
                self.metadata[k] = [v[i] for i in keep_indices]
            else:
                self.metadata[k] = [v[i] for i in keep_indices]

        self._invalidate_cache()
        logger.info("Removed embeddings. New size: %d.", self.embeddings.shape[0])

    def clear(self) -> None:
        """Clears all embeddings and metadata."""
        self.embeddings = None
        self.metadata = {}
        self._invalidate_cache()
        logger.info("EmbeddingBank cleared.")

    def mean(self) -> torch.Tensor:
        """Computes and caches the mean of the embeddings."""
        if self.embeddings is None:
            raise RuntimeError("Cannot compute mean of empty EmbeddingBank.")
        if "mean" not in self._stats_cache:
            self._stats_cache["mean"] = self.embeddings.mean(dim=0).cpu()
        return self._stats_cache["mean"].to(self.device)

    def std(self) -> torch.Tensor:
        """Computes and caches the standard deviation of the embeddings."""
        if self.embeddings is None:
            raise RuntimeError("Cannot compute std of empty EmbeddingBank.")
        if "std" not in self._stats_cache:
            self._stats_cache["std"] = self.embeddings.std(dim=0).cpu()
        return self._stats_cache["std"].to(self.device)

    def covariance(self) -> torch.Tensor:
        """Computes and caches the covariance matrix of the embeddings."""
        if self.embeddings is None or self.embeddings.shape[0] < 2:
            raise RuntimeError("Cannot compute covariance with fewer than 2 embeddings.")
        if "covariance" not in self._stats_cache:
            centered = self.embeddings - self.embeddings.mean(dim=0)
            cov = (centered.T @ centered) / (self.embeddings.shape[0] - 1)
            self._stats_cache["covariance"] = cov.cpu()
        return self._stats_cache["covariance"].to(self.device)

    def nearest_neighbors(
        self, query: torch.Tensor, k: int = 1, metric: str = "cosine"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the k-nearest neighbors in the bank for the given query.
        
        Returns:
            Tuple of (distances, indices).
        """
        if self.embeddings is None:
            raise RuntimeError("Cannot compute nearest neighbors on empty EmbeddingBank.")
        
        query = query.to(self.device, dtype=self.dtype)
        if query.ndim == 1:
            query = query.unsqueeze(0)
            
        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Query dim {query.shape[1]} doesn't match embedding dim {self.embedding_dim}")

        k = min(k, len(self))
        
        if metric == "cosine":
            q_norm = torch.nn.functional.normalize(query, p=2, dim=1)
            e_norm = self.embeddings if self.normalize else torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
            sim = torch.mm(q_norm, e_norm.T)
            dist = 1.0 - sim
            values, indices = torch.topk(dist, k, dim=1, largest=False)
            return values, indices
        elif metric in ("l2", "euclidean"):
            dist = torch.cdist(query, self.embeddings, p=2)
            values, indices = torch.topk(dist, k, dim=1, largest=False)
            return values, indices
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'l2'.")

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary of the embedding bank. Cache is omitted."""
        return {
            "config": {
                "embedding_dim": self.embedding_dim,
                "max_size": self.max_size,
                "device": str(self.device),
                "normalize": self.normalize,
                "dtype": self.dtype,
            },
            "embeddings": self.embeddings,
            "metadata": self.metadata,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state dictionary into the embedding bank."""
        if "config" in state_dict:
            self.config = EmbeddingBankConfig(**state_dict["config"])
        else:
            self.config = EmbeddingBankConfig(
                embedding_dim=state_dict["embedding_dim"],
                max_size=state_dict["max_size"],
                device=state_dict.get("device", "cpu"),
                normalize=state_dict["normalize"],
                dtype=state_dict["dtype"],
            )
        
        if state_dict.get("embeddings") is not None:
            self.embeddings = state_dict["embeddings"].to(self.device)
        else:
            self.embeddings = None
            
        self.metadata = state_dict.get("metadata", {})
        self._invalidate_cache()

    def save(self, path: Union[str, Path]) -> None:
        """Saves the embedding bank to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, Path]) -> None:
        """Loads the embedding bank from disk."""
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def summary(self) -> Dict[str, Any]:
        """Returns a summary of the current state of the embedding bank."""
        return {
            "embedding_dim": self.embedding_dim,
            "current_size": len(self),
            "max_size": self.max_size,
            "device": str(self.device),
            "normalize": self.normalize,
            "dtype": str(self.dtype),
            "metadata_keys": list(self.metadata.keys()),
            "cached_stats": list(self._stats_cache.keys()),
        }
